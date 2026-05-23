/* ═══════════════════════════════════════════════════════════
   Desktop Comic Strip Client — pure fetch polling, no Socket.IO
   Polls /api/story every second for state updates.
═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────
  var storyState   = { title: 'oh-my-comic', segments: [], current_index: 0 };
  var currentIndex = 0;
  var comicVisible = true;
  var lastStatusMsg = '';
  var POLL_INTERVAL = 1000;

  // ── Layout constants ───────────────────────────────────
  // Character image aspect ratio (width / height).
  // Change this when switching to a different generation resolution.
  // 758x1024 → 0.7402   |   768x1024 → 0.75   |   1080x1920 → 0.5625
  var CHARACTER_ASPECT    = 758 / 1024;
  var CENTER_HEIGHT_RATIO = 0.86;   // center image height as fraction of viewport height
  var SIDE_HEIGHT_RATIO   = 0.46;   // max side image height as fraction of viewport height

  // Track what we've already rendered to avoid unnecessary DOM updates
  var renderedIndex = -1;
  var renderedSegCount = 0;
  var renderedImageVersions = {};  // seg_id -> JSON string of image statuses

  // ── DOM refs ───────────────────────────────────────────
  var bgCurrent      = document.getElementById('bg-current');
  var bgNext         = document.getElementById('bg-next');
  var comicOverlay   = document.getElementById('comic-overlay');
  var toggleLabel    = document.getElementById('toggle-label');
  var desktopStatus  = document.getElementById('desktop-status');
  var segmentCounter = document.getElementById('segment-counter');
  var prevImages     = document.getElementById('prev-images');
  var currImages     = document.getElementById('curr-images');
  var nextImages     = document.getElementById('next-images');
  var currLabel      = document.getElementById('curr-label');

  // ── Restore toggle state ───────────────────────────────
  var savedVisible = localStorage.getItem('comic_visible');
  if (savedVisible === 'false') {
    comicVisible = false;
    applyComicVisibility();
  }

  // ── Polling ────────────────────────────────────────────
  function poll() {
    fetch('/api/story')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        handleStoryUpdate(data);
        setTimeout(poll, POLL_INTERVAL);
      })
      .catch(function () {
        setDesktopStatus('服务器连接失败，正在重试...', 'error');
        setTimeout(poll, 3000);
      });
  }

  // ── Elapsed seconds helper ─────────────────────────────
  function formatStatusWithElapsed(status, busy) {
    if (!busy || !status.timestamp) return status.message;
    var elapsed = Math.floor(Date.now() / 1000 - status.timestamp);
    if (elapsed < 3) return status.message;
    return status.message + ' (' + elapsed + 's)';
  }

  // ── Handle story update ────────────────────────────────
  function handleStoryUpdate(data) {
    storyState = data;

    // Status — refresh every poll so elapsed seconds update
    // busy = generating OR batch imaging (both show elapsed timer)
    var status = data.latest_status || {};
    var busy   = !!data.is_generating || !!data.is_batch_imaging;
    if (status.message) {
      lastStatusMsg = status.message;
      setDesktopStatus(
        formatStatusWithElapsed(status, busy),
        status.type || 'info'
      );
    }

    var newIndex    = data.current_index || 0;
    var segCount    = (data.segments || []).length;
    var indexChanged = newIndex !== currentIndex;
    var countChanged = segCount !== renderedSegCount;

    // Check if any image statuses changed for visible segments
    var imageChanged = checkImageChanges(data.segments || [], newIndex);

    if (indexChanged) {
      currentIndex = newIndex;
      updateBackground(newIndex, true);
    }

    if (indexChanged || countChanged || imageChanged) {
      renderComicStrip();
      renderedIndex    = newIndex;
      renderedSegCount = segCount;
    }
  }

  function checkImageChanges(segments, idx) {
    // Check prev, curr, next segments for image status changes
    var toCheck = [idx - 1, idx, idx + 1];
    for (var i = 0; i < toCheck.length; i++) {
      var si = toCheck[i];
      if (si < 0 || si >= segments.length) continue;
      var seg = segments[si];
      var key = String(seg.id);
      var version = JSON.stringify({
        bg: seg.background_image ? seg.background_image.status : null,
        chars: (seg.character_images || []).map(function (c) { return c.id + ':' + c.status; }).join(','),
      });
      if (renderedImageVersions[key] !== version) {
        renderedImageVersions[key] = version;
        return true;
      }
    }
    return false;
  }

  // ── Background Transition ──────────────────────────────
  var bgTransitionTimer = null;

  function updateBackground(idx, animate) {
    var seg = findSegment(idx);
    var url = seg && seg.background_image && seg.background_image.status === 'done'
      ? seg.background_image.url : null;

    if (!url) {
      if (!animate) {
        bgCurrent.style.backgroundImage = '';
        bgCurrent.classList.add('bg-active');
      }
      return;
    }

    if (!animate) {
      bgCurrent.style.backgroundImage = 'url("' + url + '")';
      bgCurrent.classList.add('bg-active');
      bgNext.classList.remove('bg-active');
      return;
    }

    clearTimeout(bgTransitionTimer);
    bgNext.style.backgroundImage = 'url("' + url + '")';
    bgNext.classList.add('bg-active');
    bgTransitionTimer = setTimeout(function () {
      bgCurrent.style.backgroundImage = 'url("' + url + '")';
      bgNext.classList.remove('bg-active');
    }, 1300);
  }

  // ── Comic Strip Render ─────────────────────────────────
  function renderComicStrip() {
    var segments = storyState.segments || [];
    var total    = segments.length;

    segmentCounter.textContent = total === 0
      ? '等待故事开始…'
      : '第 ' + (currentIndex + 1) + ' 段 / 共 ' + total + ' 段';

    currLabel.textContent = total > 0 ? '第 ' + (currentIndex + 1) + ' 段' : '当前段';

    renderGroup(prevImages, findSegment(currentIndex - 1), 'side');
    renderGroup(currImages, findSegment(currentIndex),     'center');
    renderGroup(nextImages, findSegment(currentIndex + 1), 'side');

    // Recalculate sizing in case character count changed for current segment
    updateComicSizing();

    // Also update background if it just became available
    var seg = findSegment(currentIndex);
    if (seg && seg.background_image && seg.background_image.status === 'done') {
      var currentBg = bgCurrent.style.backgroundImage;
      var expectedUrl = 'url("' + seg.background_image.url + '")';
      if (currentBg !== expectedUrl) {
        updateBackground(currentIndex, true);
      }
    }
  }

  function findSegment(id) {
    return (storyState.segments || []).find(function (s) { return s.id === id; });
  }

  function renderGroup(container, seg, size) {
    container.innerHTML = '';

    // Apply layout class so CSS can handle stacking vs flex
    if (size === 'side') {
      container.classList.add('group-images-side');
      container.classList.remove('group-images-center');
    } else {
      container.classList.add('group-images-center');
      container.classList.remove('group-images-side');
    }

    if (!seg) {
      var empty = document.createElement('div');
      empty.className = 'group-empty';
      empty.textContent = '—';
      container.appendChild(empty);
      return;
    }

    var images = seg.character_images || [];

    if (images.length === 0) {
      var card = document.createElement('div');
      card.className = size === 'center'
        ? 'comic-card comic-card-center'
        : 'comic-card comic-card-side';
      var ph = document.createElement('div');
      ph.className = 'comic-placeholder';
      ph.innerHTML = '<span class="ph-icon">🎭</span><span>本段无角色图</span>';
      card.appendChild(ph);
      container.appendChild(card);
      return;
    }

    images.forEach(function (ci) {
      container.appendChild(createComicCard(ci, size));
    });
  }

  function createComicCard(ci, size) {
    var card = document.createElement('div');
    card.className = size === 'center'
      ? 'comic-card comic-card-center'
      : 'comic-card comic-card-side';

    if (ci.status === 'done' && ci.url) {
      var img = document.createElement('img');
      img.src = ci.url;
      img.alt = ci.id;
      img.loading = 'lazy';
      card.appendChild(img);
    } else if (ci.status === 'failed') {
      var ph = document.createElement('div');
      ph.className = 'comic-placeholder';
      ph.innerHTML = '<span class="ph-icon">⚠️</span><span>生成失败</span>';
      card.appendChild(ph);
    } else {
      var ph2 = document.createElement('div');
      ph2.className = 'comic-placeholder loading';
      ph2.innerHTML = '<span class="ph-icon">🖼️</span><span>' + ci.id + '</span><span>生成中…</span>';
      card.appendChild(ph2);
    }
    return card;
  }

  // ── Status ─────────────────────────────────────────────
  function setDesktopStatus(message, type) {
    desktopStatus.textContent = message;
    desktopStatus.className = 'desktop-status';
    if (type === 'success') desktopStatus.classList.add('status-success');
    if (type === 'error')   desktopStatus.classList.add('status-error');
  }

  // ── Toggle Comic ───────────────────────────────────────
  window.toggleComic = function () {
    comicVisible = !comicVisible;
    localStorage.setItem('comic_visible', comicVisible ? 'true' : 'false');
    applyComicVisibility();
  };

  function applyComicVisibility() {
    if (comicVisible) {
      comicOverlay.classList.remove('hidden-overlay');
      toggleLabel.textContent = '隐藏连环画';
    } else {
      comicOverlay.classList.add('hidden-overlay');
      toggleLabel.textContent = '显示连环画';
    }
  }

  // ── Dynamic Comic Sizing ───────────────────────────────
  function clamp(val, lo, hi) {
    return Math.max(lo, Math.min(hi, val));
  }

  function getVisibleCharacterCount(idx) {
    var seg = findSegment(idx);
    if (!seg) return 1;
    var chars = seg.character_images || [];
    return Math.max(1, Math.min(2, chars.length || 1));
  }

  function updateComicSizing() {
    var vw = window.innerWidth;
    var vh = window.innerHeight;

    var groupGap = clamp(Math.round(vw * 0.018), 20, 56);
    var innerGap = clamp(Math.round(vw * 0.005), 6, 16);

    var centerCount = getVisibleCharacterCount(currentIndex);

    // ── Step 1: Center image — height-first ──────────────
    var maxCenterH  = vh * CENTER_HEIGHT_RATIO;
    var centerWByH  = maxCenterH * CHARACTER_ASPECT;

    // ── Step 2: Ensure room for side groups ──────────────
    // Side container pre-reserves hover-expanded width = 2*sideW + innerGap.
    // Minimum sideW = 60 px.
    var minSideW          = 60;
    var minSideContainerW = 2 * minSideW + innerGap;
    var maxCenterTotalW   = vw * 0.96 - 2 * minSideContainerW - 2 * groupGap;

    var centerTotalWByH = centerCount * centerWByH + (centerCount - 1) * innerGap;
    var centerTotalW    = Math.min(centerTotalWByH, maxCenterTotalW);

    var centerW = Math.floor(
      centerCount === 1
        ? centerTotalW
        : (centerTotalW - innerGap) / centerCount
    );
    centerW = clamp(centerW, 120, 900);
    var centerH  = Math.round(centerW / CHARACTER_ASPECT);
    centerTotalW = centerCount * centerW + (centerCount - 1) * innerGap;

    // ── Step 3: Side images — fill remaining space ───────
    // Layout: [sideContainer] gap [centerContainer] gap [sideContainer]
    // sideContainer width = 2*sideW + innerGap  (pre-reserved for hover expand)
    // Total = 2*(2*sideW + innerGap) + 2*groupGap + centerTotalW
    //       = 4*sideW + 2*innerGap + 2*groupGap + centerTotalW
    var sideWByW = (vw * 0.96 - 2 * innerGap - 2 * groupGap - centerTotalW) / 4;
    var sideWByH = vh * SIDE_HEIGHT_RATIO * CHARACTER_ASPECT;
    var sideW    = Math.floor(Math.min(sideWByW, sideWByH));
    sideW = clamp(sideW, 60, 500);
    var sideH = Math.round(sideW / CHARACTER_ASPECT);

    var root = document.documentElement;
    root.style.setProperty('--comic-center-w',  centerW  + 'px');
    root.style.setProperty('--comic-center-h',  centerH  + 'px');
    root.style.setProperty('--comic-side-w',    sideW    + 'px');
    root.style.setProperty('--comic-side-h',    sideH    + 'px');
    root.style.setProperty('--comic-gap',       groupGap + 'px');
    root.style.setProperty('--comic-inner-gap', innerGap + 'px');
  }

  window.addEventListener('resize', updateComicSizing);
  updateComicSizing();

  // ── Init ───────────────────────────────────────────────
  setDesktopStatus('正在连接...', 'info');
  poll();

})();
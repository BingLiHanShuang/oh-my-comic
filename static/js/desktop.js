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
  function formatStatusWithElapsed(status, generating) {
    if (!generating || !status.timestamp) return status.message;
    var elapsed = Math.floor(Date.now() / 1000 - status.timestamp);
    if (elapsed < 3) return status.message;
    return status.message + ' (' + elapsed + 's)';
  }

  // ── Handle story update ────────────────────────────────
  function handleStoryUpdate(data) {
    storyState = data;

    // Status — refresh every poll so elapsed seconds update
    var status = data.latest_status || {};
    if (status.message) {
      lastStatusMsg = status.message;
      setDesktopStatus(
        formatStatusWithElapsed(status, !!data.is_generating),
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

  function updateComicSizing() {
    var vw = window.innerWidth;
    var vh = window.innerHeight;

    // Group gap between left / center / right groups
    var groupGap  = clamp(Math.round(vw * 0.014), 20, 48);
    // Inner gap between two cards within the same group
    var innerGap  = clamp(Math.round(vw * 0.005), 8, 18);

    // Worst case: 2 side cards + 2 center cards + 2 side cards
    // total = 2*sideW + innerGap + 2*centerW + innerGap + 2*sideW + innerGap + 2*groupGap
    // sideW = centerW * 2/3
    // total = 4*(2/3)*centerW + 2*centerW + 3*innerGap + 2*groupGap
    //       = (8/3 + 2)*centerW + 3*innerGap + 2*groupGap
    //       = (14/3)*centerW + 3*innerGap + 2*groupGap
    var usableW   = vw * 0.96;
    var centerByW = (usableW - 3 * innerGap - 2 * groupGap) / (14 / 3);

    // Height constraint: center card should not exceed 62% of viewport height
    var centerByH = vh * 0.62 * (3 / 4);   // 3:4 portrait ratio → width = height * 3/4

    var centerW = Math.floor(Math.min(centerByW, centerByH));
    centerW = clamp(centerW, 160, 640);

    var centerH = Math.round(centerW * 4 / 3);
    var sideW   = Math.round(centerW * 2 / 3);
    var sideH   = Math.round(sideW   * 4 / 3);

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
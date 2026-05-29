/* ═══════════════════════════════════════════════════════════
   Desktop Comic Strip Client — pure fetch polling, no Socket.IO
   Polls /api/story every second for state updates.
   Supports UI_LANGUAGE=zh-CN|en-US via /api/story response.

   Keyboard shortcuts:
     ←  / →   : previous / next segment
     Space    : toggle comic strip visibility
     R        : regenerate all images in current segment
                (only shown/active when segment images are all done/failed
                 and no generation/regen is in progress)
═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── i18n dictionary ────────────────────────────────────
  var I18N = {
    'zh-CN': {
      connecting:      '正在连接...',
      connFailed:      '服务器连接失败，正在重试...',
      waitingStory:    '等待故事开始…',
      segCounter:      function (cur, total) { return '第 ' + cur + ' 段 / 共 ' + total + ' 段'; },
      segLabel:        function (n) { return '第 ' + n + ' 段'; },
      prevLabel:       '上一段',
      currLabel:       '当前段',
      nextLabel:       '下一段',
      noCharImg:       '本段无角色图',
      imgFailed:       '生成失败',
      imgGenerating:   '生成中…',
      imgUploaded:     '参考图',
      hideComic:       '隐藏连环画',
      showComic:       '显示连环画',
      toggleTitle:     '隐藏/显示连环画',
      regenTitle:      '重新生成此图',
      regenQueued:     '已加入重生图队列',
      regenSegTitle:   '重新生成本段全部图片 (R)',
      shortcutsBase:   '←/→ 切换段落　Space 隐藏/显示　点击图片放大',
      shortcutsRegen:  '　R 重生本段',
    },
    'en-US': {
      connecting:      'Connecting...',
      connFailed:      'Server connection failed, retrying...',
      waitingStory:    'Waiting for story to begin…',
      segCounter:      function (cur, total) { return 'Segment ' + cur + ' / ' + total; },
      segLabel:        function (n) { return 'Segment ' + n; },
      prevLabel:       'Previous',
      currLabel:       'Current',
      nextLabel:       'Next',
      noCharImg:       'No character image',
      imgFailed:       'Generation failed',
      imgGenerating:   'Generating…',
      imgUploaded:     'Reference',
      hideComic:       'Hide comic strip',
      showComic:       'Show comic strip',
      toggleTitle:     'Hide / Show comic strip',
      regenTitle:      'Regenerate this image',
      regenQueued:     'Added to regen queue',
      regenSegTitle:   'Regenerate all images in this segment (R)',
      shortcutsBase:   '←/→ Segment　Space Hide/Show　Click image to zoom',
      shortcutsRegen:  '　R Regen Segment',
    },
  };

  var LANG = 'zh-CN';

  function t(key) {
    var dict = I18N[LANG] || I18N['zh-CN'];
    return dict[key] !== undefined ? dict[key] : (I18N['zh-CN'][key] || key);
  }

  function applyI18n() {
    var prevGroups = document.querySelectorAll('.comic-group-prev .group-label');
    prevGroups.forEach(function (el) { el.textContent = t('prevLabel'); });
    var nextGroups = document.querySelectorAll('.comic-group-next .group-label');
    nextGroups.forEach(function (el) { el.textContent = t('nextLabel'); });
    var toggleBtn = document.getElementById('toggle-comic-btn');
    if (toggleBtn) toggleBtn.title = t('toggleTitle');
    applyComicVisibility();
    updateShortcutsHint();
  }

  // ── State ──────────────────────────────────────────────
  var storyState   = { title: 'oh-my-comic', segments: [], current_index: 0 };
  var currentIndex = 0;
  var comicVisible = true;
  var lastStatusMsg = '';
  var POLL_INTERVAL = 1000;
  var isGenerating   = false;
  var isBatchImaging = false;
  var isRegenImaging = false;

  var CHARACTER_ASPECT    = 758 / 1024;
  var CENTER_HEIGHT_RATIO = 0.86;
  var SIDE_HEIGHT_RATIO   = 0.46;

  var renderedIndex = -1;
  var renderedSegCount = 0;
  var renderedImageVersions = {};

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
  var shortcutsEl    = document.getElementById('desktop-shortcuts');
  var lightbox       = document.getElementById('image-lightbox');
  var lightboxImg    = document.getElementById('image-lightbox-img');

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
        var serverLang = data.ui_language || 'zh-CN';
        if (serverLang !== LANG) {
          LANG = serverLang;
          applyI18n();
        }
        handleStoryUpdate(data);
        setTimeout(poll, POLL_INTERVAL);
      })
      .catch(function () {
        setDesktopStatus(t('connFailed'), 'error');
        setTimeout(poll, 3000);
      });
  }

  function formatStatusWithElapsed(status, busy) {
    if (!busy || !status.timestamp) return status.message;
    var elapsed = Math.floor(Date.now() / 1000 - status.timestamp);
    if (elapsed < 3) return status.message;
    return status.message + ' (' + elapsed + 's)';
  }

  // ── Handle story update ────────────────────────────────
  function handleStoryUpdate(data) {
    storyState = data;
    isGenerating   = !!data.is_generating;
    isBatchImaging = !!data.is_batch_imaging;
    isRegenImaging = !!data.is_regen_imaging;

    var status = data.latest_status || {};
    var busy   = isGenerating || isBatchImaging || isRegenImaging;
    if (status.message) {
      lastStatusMsg = status.message;
      setDesktopStatus(formatStatusWithElapsed(status, busy), status.type || 'info');
    }

    var newIndex    = data.current_index || 0;
    var segCount    = (data.segments || []).length;
    var indexChanged = newIndex !== currentIndex;
    var countChanged = segCount !== renderedSegCount;
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

    // Update shortcuts hint whenever busy state or current segment changes
    updateShortcutsHint();
  }

  function checkImageChanges(segments, idx) {
    var toCheck = [idx - 1, idx, idx + 1];
    for (var i = 0; i < toCheck.length; i++) {
      var si = toCheck[i];
      if (si < 0 || si >= segments.length) continue;
      var seg = segments[si];
      var key = String(seg.id);
      var version = JSON.stringify({
        bg:    seg.background_image ? (seg.background_image.status + '|' + (seg.background_image.url || '')) : null,
        chars: (seg.character_images || []).map(function (c) { return c.id + ':' + c.status + ':' + (c.url || ''); }).join(','),
      });
      if (renderedImageVersions[key] !== version) {
        renderedImageVersions[key] = version;
        return true;
      }
    }
    return false;
  }

  // ── Shortcuts hint ─────────────────────────────────────
  function canRegenCurrentSegment() {
    if (isGenerating || isBatchImaging || isRegenImaging) return false;
    var seg = findSegment(currentIndex);
    if (!seg) return false;
    return isSegmentImagesFinished(seg);
  }

  function isSegmentImagesFinished(seg) {
    if (!seg) return false;
    var bi = seg.background_image;
    if (bi && bi.status !== 'done' && bi.status !== 'failed') return false;
    var chars = seg.character_images || [];
    for (var i = 0; i < chars.length; i++) {
      if (chars[i].status !== 'done' && chars[i].status !== 'failed') return false;
    }
    return true;
  }

  function updateShortcutsHint() {
    if (!shortcutsEl) return;
    var text = t('shortcutsBase');
    if (canRegenCurrentSegment()) {
      text += t('shortcutsRegen');
    }
    shortcutsEl.textContent = text;
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
      ? t('waitingStory')
      : t('segCounter')(currentIndex + 1, total);

    currLabel.textContent = total > 0 ? t('segLabel')(currentIndex + 1) : t('currLabel');

    renderGroup(prevImages, findSegment(currentIndex - 1), 'side');
    renderGroup(currImages, findSegment(currentIndex),     'center');
    renderGroup(nextImages, findSegment(currentIndex + 1), 'side');

    updateComicSizing();

    // Refresh background if URL changed (e.g. after regen with cache-buster)
    var seg = findSegment(currentIndex);
    if (seg && seg.background_image && seg.background_image.status === 'done') {
      var expectedUrl = 'url("' + seg.background_image.url + '")';
      if (bgCurrent.style.backgroundImage !== expectedUrl) {
        updateBackground(currentIndex, true);
      }
    }
  }

  function findSegment(id) {
    return (storyState.segments || []).find(function (s) { return s.id === id; });
  }

  function renderGroup(container, seg, size) {
    container.innerHTML = '';

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
      card.className = size === 'center' ? 'comic-card comic-card-center' : 'comic-card comic-card-side';
      var ph = document.createElement('div');
      ph.className = 'comic-placeholder';
      ph.innerHTML = '<span class="ph-icon">🎭</span><span>' + t('noCharImg') + '</span>';
      card.appendChild(ph);
      container.appendChild(card);
      return;
    }

    // Only show regen buttons when the segment is fully finished and not busy
    var segFinished = isSegmentImagesFinished(seg);
    var canRegen    = segFinished && !isGenerating && !isBatchImaging && !isRegenImaging;

    images.forEach(function (ci) {
      container.appendChild(createComicCard(ci, seg, size, canRegen));
    });
  }

  function createComicCard(ci, seg, size, canRegen) {
    var card = document.createElement('div');
    card.className = size === 'center' ? 'comic-card comic-card-center' : 'comic-card comic-card-side';

    if (ci.status === 'done' && ci.url) {
      var img = document.createElement('img');
      img.src = ci.url;
      img.alt = ci.id;
      img.loading = 'lazy';
      img.addEventListener('click', function () { openLightbox(ci.url); });
      card.appendChild(img);

      if (ci.source !== 'uploaded') {
        if (canRegen) {
          var regenBtn = document.createElement('button');
          regenBtn.className = 'regen-btn';
          regenBtn.title = t('regenTitle');
          regenBtn.innerHTML = '&#8635;';
          regenBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            regenImage(seg.id, 'character', ci.id, regenBtn);
          });
          card.appendChild(regenBtn);
        }
      } else {
        var refBadge = document.createElement('span');
        refBadge.className = 'ref-badge';
        refBadge.textContent = t('imgUploaded');
        card.appendChild(refBadge);
      }
    } else if (ci.status === 'failed') {
      var ph = document.createElement('div');
      ph.className = 'comic-placeholder';
      ph.innerHTML = '<span class="ph-icon">⚠️</span><span>' + t('imgFailed') + '</span>';
      card.appendChild(ph);
      if (ci.source !== 'uploaded' && canRegen) {
        var regenBtn2 = document.createElement('button');
        regenBtn2.className = 'regen-btn regen-btn-failed';
        regenBtn2.title = t('regenTitle');
        regenBtn2.innerHTML = '&#8635;';
        regenBtn2.addEventListener('click', function (e) {
          e.stopPropagation();
          regenImage(seg.id, 'character', ci.id, regenBtn2);
        });
        card.appendChild(regenBtn2);
      }
    } else {
      var ph2 = document.createElement('div');
      ph2.className = 'comic-placeholder loading';
      ph2.innerHTML = '<span class="ph-icon">🖼️</span><span>' + ci.id + '</span><span>' + t('imgGenerating') + '</span>';
      card.appendChild(ph2);
    }
    return card;
  }

  // ── Regen ──────────────────────────────────────────────
  function regenImage(segId, imageType, itemId, btnEl) {
    if (btnEl) {
      btnEl.disabled = true;
      btnEl.style.opacity = '0.4';
    }
    fetch('/api/regenerate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segment_id: segId, image_type: imageType, item_id: itemId }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.ok) {
          setDesktopStatus(t('regenQueued'), 'info');
        } else {
          setDesktopStatus(data.message || 'Regen failed', 'error');
          if (btnEl) { btnEl.disabled = false; btnEl.style.opacity = ''; }
        }
      })
      .catch(function (err) {
        setDesktopStatus('Regen error: ' + err.message, 'error');
        if (btnEl) { btnEl.disabled = false; btnEl.style.opacity = ''; }
      });
  }

  function regenCurrentSegment() {
    if (!canRegenCurrentSegment()) return;
    fetch('/api/regenerate-segment-images', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ segment_id: currentIndex }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        setDesktopStatus(data.message || t('regenQueued'), data.ok ? 'info' : 'error');
      })
      .catch(function (err) {
        setDesktopStatus('Regen error: ' + err.message, 'error');
      });
  }

  // ── Keyboard Navigation ────────────────────────────────
  document.addEventListener('keydown', function (e) {
    var tag = document.activeElement ? document.activeElement.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;

    if (e.key === 'Escape') {
      closeLightbox();
      return;
    }

    if (lightbox && !lightbox.classList.contains('hidden')) return;

    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      navigateTo(currentIndex - 1);
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      navigateTo(currentIndex + 1);
    } else if (e.code === 'Space') {
      e.preventDefault();
      toggleComic();
    } else if (e.key.toLowerCase() === 'r') {
      e.preventDefault();
      regenCurrentSegment();
    }
  });

  function navigateTo(idx) {
    var total = (storyState.segments || []).length;
    if (total === 0) return;
    if (idx < 0) idx = 0;
    if (idx >= total) idx = total - 1;
    if (idx === currentIndex) return;

    currentIndex = idx;
    renderComicStrip();
    updateBackground(idx, true);
    updateShortcutsHint();

    fetch('/api/current-index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_index: idx }),
    }).catch(function () {});
  }

  // ── Lightbox ───────────────────────────────────────────
  function openLightbox(url) {
    if (!lightbox || !lightboxImg) return;
    lightboxImg.src = url;
    lightbox.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    if (!lightbox) return;
    lightbox.classList.add('hidden');
    if (lightboxImg) lightboxImg.src = '';
    document.body.style.overflow = '';
  }

  window.closeLightbox = closeLightbox;

  if (lightbox) {
    lightbox.addEventListener('click', function (e) {
      if (e.target === lightbox) closeLightbox();
    });
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
      if (toggleLabel) toggleLabel.textContent = t('hideComic');
    } else {
      comicOverlay.classList.add('hidden-overlay');
      if (toggleLabel) toggleLabel.textContent = t('showComic');
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

    var maxCenterH  = vh * CENTER_HEIGHT_RATIO;
    var centerWByH  = maxCenterH * CHARACTER_ASPECT;

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
  applyI18n();
  setDesktopStatus(t('connecting'), 'info');
  poll();

})();
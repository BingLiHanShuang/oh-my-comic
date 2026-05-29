/* ═══════════════════════════════════════════════════════════
   Mobile Story Client — pure fetch polling, no Socket.IO
   Polls /api/story every second for state updates.
   Supports UI_LANGUAGE=zh-CN|en-US via /api/story response.
   支持通过 /api/story 返回的 ui_language 切换界面语言。
═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── i18n dictionary ────────────────────────────────────
  // 界面文案字典，支持 zh-CN 和 en-US。
  // UI text dictionary supporting zh-CN and en-US.
  var I18N = {
    'zh-CN': {
      connecting:          '正在连接服务器...',
      connFailed:          '服务器连接失败，正在重试...',
      sending:             '正在发送请求...',
      generating:          '生成中，请稍候...',
      reqFailed:           '请求失败',
      netError:            '网络错误: ',
      generate:            '生成',
      creativeOff:         '创作模式',
      creativeOn:          '创作模式 开',
      creativeTitle:       '创作模式：开启后连续写文本，关闭后统一生图',
      creativeTitleOn:     '创作模式已开启：连续写文本，关闭后统一生图',
      creativeTitleWait:   '等待 Qwen 就绪后可开启',
      creativeTitleReady:  '开启创作模式（需要 Qwen 就绪）',
      creativeToggleFail:  '切换失败',
      creativeOffStatus:   '创作模式已关闭，正在启动批量生图...',
      emptyTitle:          '故事还没有开始。',
      emptyHint:           '在下方输入你希望故事如何开始，然后点击「生成」。',
      placeholder:         '输入你希望下一段故事如何发展...',
      serveOnly:           '当前为只读模式，无法生成新内容。',
      segLabel:            function (n) { return '第 ' + n + ' 段'; },
      segStreaming:        function (n) { return '第 ' + n + ' 段 · 生成中…'; },
      defaultDirection:    '继续故事',
      imgPreviewAlt:       '预览',
    },
    'en-US': {
      connecting:          'Connecting to server...',
      connFailed:          'Server connection failed, retrying...',
      sending:             'Sending request...',
      generating:          'Generating, please wait...',
      reqFailed:           'Request failed',
      netError:            'Network error: ',
      generate:            'Generate',
      creativeOff:         'Creative Mode',
      creativeOn:          'Creative Mode ON',
      creativeTitle:       'Creative Mode: keep LLM running; images generated when turned off',
      creativeTitleOn:     'Creative Mode ON: write multiple segments; images generated when turned off',
      creativeTitleWait:   'Waiting for Qwen LLM to be ready',
      creativeTitleReady:  'Enable Creative Mode (requires Qwen ready)',
      creativeToggleFail:  'Toggle failed',
      creativeOffStatus:   'Creative mode off, starting batch imaging...',
      emptyTitle:          'The story has not started yet.',
      emptyHint:           'Enter how you want the story to begin below, then click Generate.',
      placeholder:         'Enter how you want the next segment to develop...',
      serveOnly:           'Read-only mode: generation is disabled.',
      segLabel:            function (n) { return 'Segment ' + n; },
      segStreaming:        function (n) { return 'Segment ' + n + ' · Generating…'; },
      defaultDirection:    'Continue the story',
      imgPreviewAlt:       'Preview',
    },
  };

  // Current UI language — updated from /api/story response.
  // 当前界面语言，从 /api/story 返回值更新。
  var LANG = 'zh-CN';

  function t(key) {
    var dict = I18N[LANG] || I18N['zh-CN'];
    return dict[key] !== undefined ? dict[key] : (I18N['zh-CN'][key] || key);
  }

  // Apply i18n to static HTML elements.
  // 将 i18n 应用到静态 HTML 元素。
  function applyI18n() {
    var emptyHintEl = document.getElementById('empty-hint');
    if (emptyHintEl) {
      var ps = emptyHintEl.querySelectorAll('p');
      if (ps[0]) ps[0].textContent = t('emptyTitle');
      if (ps[1]) ps[1].textContent = t('emptyHint');
    }
    var inp = document.getElementById('direction-input');
    if (inp) inp.placeholder = t('placeholder');
    var btnTxt = document.getElementById('btn-text');
    if (btnTxt) btnTxt.textContent = t('generate');
    var soHint = document.getElementById('serve-only-hint');
    if (soHint) soHint.textContent = t('serveOnly');
    var imgThumb = document.getElementById('image-preview-thumb');
    if (imgThumb) imgThumb.alt = t('imgPreviewAlt');
    updateCreativeToggle();
  }

  // Apply i18n immediately on load so static text is never blank.
  // 页面加载时立即应用 i18n，避免静态文案空白。
  applyI18n();

  // ── State ──────────────────────────────────────────────
  var storyState = { title: 'oh-my-comic', segments: [], mode: 'generate', current_index: 0 };
  var isGenerating    = false;
  var isBatchImaging  = false;
  var creativeModeOn  = false;
  var llmReady        = false;
  var lastSegmentCount = 0;
  var lastEmittedIndex = -1;
  var lastStatusMsg    = '';
  var scrollDebounceTimer = null;
  var pollTimer = null;
  var POLL_INTERVAL    = 1000;
  var SCROLL_DEBOUNCE_MS = 150;

  // ── Pending user bubble ────────────────────────────────
  // Shown immediately after the user hits Send, before the server responds.
  // 用户点击发送后立即显示，服务器响应前的占位气泡。
  var pendingBubbleEl = null;
  var pendingBubbleText = '';

  // ── Selected image ─────────────────────────────────────
  var selectedImageFile = null;

  // ── DOM refs ───────────────────────────────────────────
  var titleEl           = document.getElementById('story-title');
  var segmentsList      = document.getElementById('segments-list');
  var emptyHint         = document.getElementById('empty-hint');
  var statusBar         = document.getElementById('status-bar');
  var statusText        = document.getElementById('status-text');
  // header-status element removed — status is shown in the bottom status bar only.
  // header-status 元素已移除，状态仅在底部状态栏显示。
  var generateBtn       = document.getElementById('generate-btn');
  var btnText           = document.getElementById('btn-text');
  var btnSpinner        = document.getElementById('btn-spinner');
  var directionInput    = document.getElementById('direction-input');
  var serveOnlyHint     = document.getElementById('serve-only-hint');
  var storyContainer    = document.getElementById('story-container');
  var creativeCb        = document.getElementById('creative-mode-checkbox');
  var creativeLabel     = document.getElementById('creative-toggle-label');
  var creativeText      = document.getElementById('creative-toggle-text');
  var imageFileInput    = document.getElementById('image-file-input');
  var imagePreviewArea  = document.getElementById('image-preview-area');
  var imagePreviewThumb = document.getElementById('image-preview-thumb');
  var imagePreviewName  = document.getElementById('image-preview-name');
  var uploadBtnLabel    = document.getElementById('upload-btn-label');

  // ── Polling ────────────────────────────────────────────
  function startPolling() { poll(); }

  function poll() {
    fetch('/api/story')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        // Update language from server if changed.
        // 如果服务器返回的语言有变化则更新。
        var serverLang = data.ui_language || 'zh-CN';
        if (serverLang !== LANG) {
          LANG = serverLang;
          applyI18n();
        }
        handleStoryUpdate(data);
        pollTimer = setTimeout(poll, POLL_INTERVAL);
      })
      .catch(function () {
        setStatus(t('connFailed'), 'error');
        pollTimer = setTimeout(poll, 3000);
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
    var generating   = !!data.is_generating;
    var batchImaging = !!data.is_batch_imaging;
    var creative     = !!data.creative_mode;
    var ready        = !!data.llm_ready;
    var busy         = generating || batchImaging;
    var status       = data.latest_status || {};

    // Status bar — refresh every poll so elapsed seconds update.
    // 每次轮询刷新状态栏，使计时秒数实时更新。
    if (status.message) {
      lastStatusMsg = status.message;
      setStatus(formatStatusWithElapsed(status, busy), status.type || 'info');
    }

    // Generating / batch imaging state
    if (generating !== isGenerating || batchImaging !== isBatchImaging) {
      isGenerating   = generating;
      isBatchImaging = batchImaging;
      setGenerating(busy);
    }

    // Creative mode state
    if (creative !== creativeModeOn || ready !== llmReady) {
      creativeModeOn = creative;
      llmReady       = ready;
      updateCreativeToggle();
    }

    // Title
    if (data.title) {
      titleEl.textContent = data.title;
      document.title = data.title;
    }

    // Mode
    if (data.mode === 'serve-only') {
      serveOnlyHint.classList.remove('hidden');
      generateBtn.disabled = true;
    }

    // Streaming segment (kept for compatibility; currently always null)
    updateStreamingCard(data.streaming_segment || null);

    // Render segments
    var segments = data.segments || [];
    if (segments.length === 0 && !data.streaming_segment) {
      // If there is a pending user bubble, keep it visible and don't wipe the list.
      // 如果有待显示的用户气泡，保留它，不清空列表。
      if (pendingBubbleEl && pendingBubbleEl.parentNode === segmentsList) {
        emptyHint.style.display = 'none';
        lastSegmentCount = 0;
        return;
      }
      emptyHint.style.display = 'flex';
      segmentsList.innerHTML  = '';
      lastSegmentCount = 0;
      return;
    }
    emptyHint.style.display = 'none';

    if (segments.length > lastSegmentCount) {
      var existingIds = new Set(
        Array.from(segmentsList.querySelectorAll('[data-seg-id]'))
          .map(function (el) { return parseInt(el.dataset.segId, 10); })
      );
      segments.forEach(function (seg) {
        if (!existingIds.has(seg.id)) {
          appendSegmentPair(seg);
        }
      });

      var allCards = segmentsList.querySelectorAll('.segment-card:not(.streaming)');
      if (allCards.length > 0) {
        setTimeout(function () {
          allCards[allCards.length - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 80);
      }
      lastSegmentCount = segments.length;
    }

    updateActiveCard(data.current_index || 0);
  }

  // ── Pending user bubble helpers ────────────────────────
  function showPendingUserBubble(text, hasImage) {
    removePendingUserBubble();
    var bubble = document.createElement('div');
    bubble.className = 'user-bubble user-bubble-pending';
    bubble.dataset.pending = 'true';
    var label = text;
    if (hasImage) label = '📎 ' + label;
    bubble.textContent = label;
    segmentsList.appendChild(bubble);
    pendingBubbleEl   = bubble;
    pendingBubbleText = text;
    emptyHint.style.display = 'none';
    setTimeout(function () {
      bubble.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 60);
  }

  function removePendingUserBubble() {
    if (pendingBubbleEl && pendingBubbleEl.parentNode) {
      pendingBubbleEl.parentNode.removeChild(pendingBubbleEl);
    }
    pendingBubbleEl   = null;
    pendingBubbleText = '';
  }

  // ── Segment pair (user bubble + system card) ───────────
  function appendSegmentPair(seg) {
    // Remove the pending bubble for this message (if it matches).
    // 如果待显示气泡与当前段落匹配，则移除。
    if (pendingBubbleEl && seg.user_text &&
        seg.user_text.trim() === pendingBubbleText.trim()) {
      removePendingUserBubble();
    }

    // User message bubble (if user_text is present)
    if (seg.user_text && seg.user_text.trim()) {
      var bubble = document.createElement('div');
      bubble.className = 'user-bubble';
      bubble.dataset.segId = seg.id;
      bubble.textContent = seg.user_text;
      segmentsList.appendChild(bubble);
    }

    // System story card
    segmentsList.appendChild(createSegmentCard(seg));
  }

  // ── Streaming card ─────────────────────────────────────
  var streamingCard = null;

  function updateStreamingCard(seg) {
    if (!seg) {
      if (streamingCard && streamingCard.parentNode) {
        streamingCard.parentNode.removeChild(streamingCard);
      }
      streamingCard = null;
      return;
    }
    emptyHint.style.display = 'none';
    if (!streamingCard) {
      streamingCard = document.createElement('div');
      streamingCard.className = 'segment-card streaming';
      streamingCard.dataset.segId = seg.id;
      var numEl = document.createElement('div');
      numEl.className = 'segment-number';
      numEl.textContent = t('segStreaming')(seg.id + 1);
      streamingCard.appendChild(numEl);
      var textEl = document.createElement('div');
      textEl.className = 'segment-text';
      streamingCard.appendChild(textEl);
      segmentsList.appendChild(streamingCard);
      setTimeout(function () {
        streamingCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 60);
    }
    var textEl = streamingCard.querySelector('.segment-text');
    if (textEl) textEl.textContent = seg.text || '';
  }

  // ── Segment card ───────────────────────────────────────
  function createSegmentCard(seg) {
    var card = document.createElement('div');
    card.className = 'segment-card';
    card.dataset.segId = seg.id;

    var numEl = document.createElement('div');
    numEl.className = 'segment-number';
    numEl.textContent = t('segLabel')(seg.id + 1);

    var textEl = document.createElement('div');
    textEl.className = 'segment-text';
    textEl.textContent = seg.text || '';

    card.appendChild(numEl);
    card.appendChild(textEl);
    return card;
  }

  function updateActiveCard(idx) {
    var cards = segmentsList.querySelectorAll('.segment-card');
    cards.forEach(function (card) {
      card.classList.toggle('active', parseInt(card.dataset.segId, 10) === idx);
    });
  }

  // ── Status ─────────────────────────────────────────────
  function setStatus(message, type) {
    statusText.textContent = message;
    statusBar.className = 'status-bar status-' + (type || 'info');
  }

  // ── Generating state ───────────────────────────────────
  function setGenerating(val) {
    generateBtn.disabled = val;
    if (val) {
      btnText.classList.add('hidden');
      btnSpinner.classList.remove('hidden');
    } else {
      btnText.classList.remove('hidden');
      btnSpinner.classList.add('hidden');
    }
    // Also disable upload button while busy.
    // 生成中时禁用上传按钮。
    if (uploadBtnLabel) {
      uploadBtnLabel.style.opacity = val ? '0.4' : '';
      uploadBtnLabel.style.pointerEvents = val ? 'none' : '';
    }
  }

  // ── Creative mode toggle ───────────────────────────────
  function updateCreativeToggle() {
    if (!creativeCb) return;

    var isMock = storyState.mode === 'mock';
    var canToggle = !isGenerating && !isBatchImaging;
    var canEnable = canToggle && (llmReady || isMock);

    // Hide the entire toggle when LLM is not yet ready and creative mode is off.
    // This avoids showing a toggle with no label text before the LLM warms up.
    // 当 LLM 未就绪且创作模式未开启时，隐藏整个开关，避免显示空白文字。
    if (creativeLabel) {
      var shouldShow = creativeModeOn || llmReady || isMock;
      creativeLabel.style.display = shouldShow ? '' : 'none';
    }

    creativeCb.checked = creativeModeOn;
    creativeCb.disabled = creativeModeOn ? !canToggle : !canEnable;

    if (creativeText) {
      creativeText.textContent = creativeModeOn ? t('creativeOn') : t('creativeOff');
    }
    if (creativeLabel) {
      creativeLabel.title = creativeModeOn
        ? t('creativeTitleOn')
        : (canEnable ? t('creativeTitleReady') : t('creativeTitleWait'));
    }
  }

  window.onCreativeModeChange = function (checkbox) {
    var enable = checkbox.checked;
    // Revert immediately; will be updated by next poll.
    // 立即回退，等待下次轮询更新。
    checkbox.checked = creativeModeOn;
    checkbox.disabled = true;

    fetch('/api/creative-mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enable: enable }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          setStatus(data.message || t('creativeToggleFail'), 'error');
        } else {
          creativeModeOn = !!data.creative_mode;
          checkbox.checked = creativeModeOn;
          if (!enable) {
            setStatus(t('creativeOffStatus'), 'info');
          }
        }
        updateCreativeToggle();
      })
      .catch(function (err) {
        setStatus(t('netError') + err.message, 'error');
        updateCreativeToggle();
      });
  };

  // ── Image upload ───────────────────────────────────────
  window.onImageSelected = function (input) {
    if (!input.files || !input.files[0]) return;
    selectedImageFile = input.files[0];
    if (imagePreviewThumb) {
      imagePreviewThumb.src = URL.createObjectURL(selectedImageFile);
    }
    if (imagePreviewName) {
      imagePreviewName.textContent = selectedImageFile.name;
    }
    if (imagePreviewArea) {
      imagePreviewArea.classList.remove('hidden');
    }
  };

  window.removeSelectedImage = function () {
    selectedImageFile = null;
    if (imageFileInput) imageFileInput.value = '';
    if (imagePreviewArea) imagePreviewArea.classList.add('hidden');
    if (imagePreviewThumb) imagePreviewThumb.src = '';
    if (imagePreviewName) imagePreviewName.textContent = '';
  };

  // ── Generate ───────────────────────────────────────────
  window.sendGenerate = function () {
    if (isGenerating || isBatchImaging) return;
    if (storyState.mode === 'serve-only') return;

    var direction = directionInput.value.trim() || t('defaultDirection');
    var hasImage  = !!selectedImageFile;

    // Show user bubble immediately.
    // 立即显示用户气泡。
    showPendingUserBubble(direction, hasImage);

    setGenerating(true);
    setStatus(t('sending'), 'info');

    var promise;

    if (hasImage) {
      // Multipart upload
      var formData = new FormData();
      formData.append('direction', direction);
      formData.append('image', selectedImageFile);
      promise = fetch('/api/generate', {
        method: 'POST',
        body: formData,
      });
    } else {
      // JSON request
      promise = fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ direction: direction }),
      });
    }

    promise
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          // Request rejected — remove the pending bubble.
          // 请求被拒绝，移除待显示气泡。
          removePendingUserBubble();
          setStatus(data.message || t('reqFailed'), 'error');
          setGenerating(false);
        } else {
          directionInput.value = '';
          removeSelectedImage();
          setStatus(t('generating'), 'info');
        }
      })
      .catch(function (err) {
        removePendingUserBubble();
        setStatus(t('netError') + err.message, 'error');
        setGenerating(false);
      });
  };

  // Enter (without Shift) submits.
  // Enter（不含 Shift）提交。
  directionInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      window.sendGenerate();
    }
  });

  // ── Scroll Detection ───────────────────────────────────
  storyContainer.addEventListener('scroll', function () {
    clearTimeout(scrollDebounceTimer);
    scrollDebounceTimer = setTimeout(detectCurrentSegment, SCROLL_DEBOUNCE_MS);
  });

  function detectCurrentSegment() {
    var cards = segmentsList.querySelectorAll('.segment-card:not(.streaming)');
    if (cards.length === 0) return;

    var containerRect = storyContainer.getBoundingClientRect();
    var containerMid  = containerRect.top + containerRect.height / 2;
    var bestIdx  = 0;
    var bestDist = Infinity;

    cards.forEach(function (card) {
      var rect    = card.getBoundingClientRect();
      var cardMid = rect.top + rect.height / 2;
      var dist    = Math.abs(cardMid - containerMid);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx  = parseInt(card.dataset.segId, 10);
      }
    });

    if (bestIdx !== lastEmittedIndex) {
      lastEmittedIndex = bestIdx;
      updateActiveCard(bestIdx);
      fetch('/api/current-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_index: bestIdx }),
      }).catch(function () {});
    }
  }

  // ── Init ───────────────────────────────────────────────
  setStatus(t('connecting'), 'info');
  startPolling();

})();
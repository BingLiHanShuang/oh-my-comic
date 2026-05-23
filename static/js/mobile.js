/* ═══════════════════════════════════════════════════════════
   Mobile Story Client — pure fetch polling, no Socket.IO
   Polls /api/story every second for state updates.
═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

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

  // ── Selected image ─────────────────────────────────────
  var selectedImageFile = null;

  // ── DOM refs ───────────────────────────────────────────
  var titleEl           = document.getElementById('story-title');
  var segmentsList      = document.getElementById('segments-list');
  var emptyHint         = document.getElementById('empty-hint');
  var statusBar         = document.getElementById('status-bar');
  var statusText        = document.getElementById('status-text');
  var headerStatus      = document.getElementById('header-status');
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
        handleStoryUpdate(data);
        pollTimer = setTimeout(poll, POLL_INTERVAL);
      })
      .catch(function () {
        setStatus('服务器连接失败，正在重试...', 'error');
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

    // Status bar — refresh every poll so elapsed seconds update
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

  // ── Segment pair (user bubble + system card) ───────────
  function appendSegmentPair(seg) {
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
      numEl.textContent = '第 ' + (seg.id + 1) + ' 段 · 生成中…';
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
    numEl.textContent = '第 ' + (seg.id + 1) + ' 段';

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
    headerStatus.textContent = message.length > 22 ? message.slice(0, 22) + '…' : message;
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
    // Also disable upload button while busy
    if (uploadBtnLabel) {
      uploadBtnLabel.style.opacity = val ? '0.4' : '';
      uploadBtnLabel.style.pointerEvents = val ? 'none' : '';
    }
  }

  // ── Creative mode toggle ───────────────────────────────
  function updateCreativeToggle() {
    if (!creativeCb) return;
    creativeCb.checked = creativeModeOn;

    // Disable toggle when busy or LLM not ready (for enabling)
    var canToggle = !isGenerating && !isBatchImaging;
    var canEnable = canToggle && (llmReady || storyState.mode === 'mock');
    creativeCb.disabled = creativeModeOn ? !canToggle : !canEnable;

    if (creativeText) {
      creativeText.textContent = creativeModeOn ? '创作模式 开' : '创作模式';
    }
    if (creativeLabel) {
      creativeLabel.title = creativeModeOn
        ? '创作模式已开启：连续写文本，关闭后统一生图'
        : (canEnable ? '开启创作模式（需要 Qwen 就绪）' : '等待 Qwen 就绪后可开启');
    }
  }

  window.onCreativeModeChange = function (checkbox) {
    var enable = checkbox.checked;
    // Revert immediately; will be updated by next poll
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
          setStatus(data.message || '切换失败', 'error');
        } else {
          creativeModeOn = !!data.creative_mode;
          checkbox.checked = creativeModeOn;
          if (!enable) {
            setStatus('创作模式已关闭，正在启动批量生图...', 'info');
          }
        }
        updateCreativeToggle();
      })
      .catch(function (err) {
        setStatus('网络错误: ' + err.message, 'error');
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

    var direction = directionInput.value.trim() || '继续故事';
    setGenerating(true);
    setStatus('正在发送请求...', 'info');

    var promise;

    if (selectedImageFile) {
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
          setStatus(data.message || '请求失败', 'error');
          setGenerating(false);
        } else {
          directionInput.value = '';
          removeSelectedImage();
          setStatus('生成中，请稍候...', 'info');
        }
      })
      .catch(function (err) {
        setStatus('网络错误: ' + err.message, 'error');
        setGenerating(false);
      });
  };

  // Enter (without Shift) submits
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
  setStatus('正在连接服务器...', 'info');
  updateCreativeToggle();
  startPolling();

})();
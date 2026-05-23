/* ═══════════════════════════════════════════════════════════
   Mobile Story Client — pure fetch polling, no Socket.IO
   Polls /api/story every second for state updates.
═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────
  var storyState = { title: 'oh-my-comic', segments: [], mode: 'generate', current_index: 0 };
  var isGenerating = false;
  var lastSegmentCount = 0;
  var lastEmittedIndex = -1;
  var lastStatusMsg = '';
  var scrollDebounceTimer = null;
  var pollTimer = null;
  var POLL_INTERVAL = 1000;
  var SCROLL_DEBOUNCE_MS = 150;

  // ── DOM refs ───────────────────────────────────────────
  var titleEl        = document.getElementById('story-title');
  var segmentsList   = document.getElementById('segments-list');
  var emptyHint      = document.getElementById('empty-hint');
  var statusBar      = document.getElementById('status-bar');
  var statusText     = document.getElementById('status-text');
  var headerStatus   = document.getElementById('header-status');
  var generateBtn    = document.getElementById('generate-btn');
  var btnText        = document.getElementById('btn-text');
  var btnSpinner     = document.getElementById('btn-spinner');
  var directionInput = document.getElementById('direction-input');
  var serveOnlyHint  = document.getElementById('serve-only-hint');
  var storyContainer = document.getElementById('story-container');

  // ── Polling ────────────────────────────────────────────
  function startPolling() {
    poll();
  }

  function poll() {
    fetch('/api/story')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        handleStoryUpdate(data);
        pollTimer = setTimeout(poll, POLL_INTERVAL);
      })
      .catch(function (err) {
        setStatus('服务器连接失败，正在重试...', 'error');
        pollTimer = setTimeout(poll, 3000);
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
    var generating = !!data.is_generating;
    var status = data.latest_status || {};

    // Update status bar every poll (so elapsed seconds refresh)
    if (status.message) {
      lastStatusMsg = status.message;
      setStatus(formatStatusWithElapsed(status, generating), status.type || 'info');
    }

    // Update generating state
    if (generating !== isGenerating) {
      isGenerating = generating;
      setGenerating(generating);
    }

    // Update title
    if (data.title) {
      titleEl.textContent = data.title;
      document.title = data.title;
    }

    // Mode
    if (data.mode === 'serve-only') {
      serveOnlyHint.classList.remove('hidden');
      generateBtn.disabled = true;
    }

    // Streaming segment (live text while LLM is generating)
    updateStreamingCard(data.streaming_segment || null);

    // Render new segments
    var segments = data.segments || [];
    if (segments.length === 0 && !data.streaming_segment) {
      emptyHint.style.display = 'flex';
      segmentsList.innerHTML = '';
      lastSegmentCount = 0;
      return;
    }
    emptyHint.style.display = 'none';

    // Only add newly arrived segments
    if (segments.length > lastSegmentCount) {
      var existingIds = new Set(
        Array.from(segmentsList.querySelectorAll('.segment-card:not(.streaming)'))
          .map(function (el) { return parseInt(el.dataset.segId, 10); })
      );
      segments.forEach(function (seg) {
        if (!existingIds.has(seg.id)) {
          segmentsList.appendChild(createSegmentCard(seg));
        }
      });

      // Scroll to newest segment if it just arrived
      var cards = segmentsList.querySelectorAll('.segment-card:not(.streaming)');
      if (cards.length > 0) {
        setTimeout(function () {
          cards[cards.length - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 80);
      }
      lastSegmentCount = segments.length;
    }

    // Update active highlight
    updateActiveCard(data.current_index || 0);
  }

  // ── Streaming card ─────────────────────────────────────
  var streamingCard = null;

  function updateStreamingCard(seg) {
    if (!seg) {
      // Remove streaming card if it exists
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

    // Update text content
    var textEl = streamingCard.querySelector('.segment-text');
    if (textEl) {
      textEl.textContent = seg.text || '';
    }
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
    isGenerating = val;
    generateBtn.disabled = val;
    if (val) {
      btnText.classList.add('hidden');
      btnSpinner.classList.remove('hidden');
    } else {
      btnText.classList.remove('hidden');
      btnSpinner.classList.add('hidden');
    }
  }

  // ── Generate ───────────────────────────────────────────
  window.sendGenerate = function () {
    if (isGenerating) return;
    if (storyState.mode === 'serve-only') return;

    var direction = directionInput.value.trim() || '继续故事';
    setGenerating(true);
    setStatus('正在发送请求...', 'info');

    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction: direction }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) {
          setStatus(data.message || '请求失败', 'error');
          setGenerating(false);
        } else {
          directionInput.value = '';
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
    var cards = segmentsList.querySelectorAll('.segment-card');
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
      // Notify backend (fire-and-forget)
      fetch('/api/current-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ current_index: bestIdx }),
      }).catch(function () {});
    }
  }

  // ── Init ───────────────────────────────────────────────
  setStatus('正在连接服务器...', 'info');
  startPolling();

})();
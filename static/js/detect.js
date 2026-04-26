/**
 * detect.js — CropGuard AI
 * Handles: drag-and-drop, file preview, form submit,
 *          scan animation, confidence bar animation, clear.
 */

(function () {
  'use strict';

  // ── Element refs ──────────────────────────────────────────────
  const input     = document.getElementById('image-input');
  const preview   = document.getElementById('preview-img');
  const dzInner   = document.getElementById('dz-inner');
  const dropZone  = document.getElementById('drop-zone');
  const scanOverlay = document.getElementById('scan-overlay');
  const form      = document.getElementById('detect-form');
  const submitBtn = document.getElementById('submit-btn');
  const btnText   = submitBtn ? submitBtn.querySelector('.btn-text')   : null;
  const btnLoader = submitBtn ? submitBtn.querySelector('.btn-loading') : null;
  const clearBtn  = document.getElementById('clear-btn');

  // ── Preview helper ───────────────────────────────────────────
  function showPreview(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
      preview.style.display = 'block';
      dzInner.style.display = 'none';
      dropZone.style.borderStyle  = 'solid';
      dropZone.style.borderColor  = 'var(--c-green)';
      dropZone.style.boxShadow    = '0 0 24px rgba(62,207,106,.2)';
      dropZone.classList.add('has-image');
    };
    reader.readAsDataURL(file);
  }

  // ── File input change ────────────────────────────────────────
  if (input) {
    input.addEventListener('change', function () {
      if (input.files[0]) showPreview(input.files[0]);
    });
  }

  // ── Drag & drop ──────────────────────────────────────────────
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(function (ev) {
      dropZone.addEventListener(ev, function (e) {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'dragend'].forEach(function (ev) {
      dropZone.addEventListener(ev, function () {
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      var file = e.dataTransfer.files[0];
      if (!file) return;
      // Inject into file input so form picks it up on POST
      try {
        var dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
      } catch (_) {
        // DataTransfer not supported (old Safari) — preview still works
      }
      showPreview(file);
    });
  }

  // ── Form submit — show loading, scan animation ────────────────
  if (form) {
    form.addEventListener('submit', function (e) {
      if (!input || !input.files || !input.files[0]) {
        e.preventDefault();
        // Shake the drop-zone
        if (dropZone) {
          dropZone.style.animation = 'none';
          void dropZone.offsetHeight; // reflow
          dropZone.style.animation = 'shake .4s ease';
        }
        return;
      }

      // Show loading state
      if (btnText)   btnText.style.display   = 'none';
      if (btnLoader) btnLoader.style.display  = 'flex';
      if (submitBtn) submitBtn.disabled       = true;
      if (clearBtn)  clearBtn.disabled        = true;

      // Scan line animation
      if (dropZone)    dropZone.classList.add('scanning');
      if (scanOverlay) scanOverlay.classList.add('active');

      // Let the form POST proceed
    });
  }

  // ── Clear / reset ────────────────────────────────────────────
  function clearForm() {
    if (input) input.value = '';
    if (preview) {
      preview.src = '';
      preview.style.display = 'none';
    }
    if (dzInner) dzInner.style.display = 'block';
    if (dropZone) {
      dropZone.style.borderStyle = 'dashed';
      dropZone.style.borderColor = 'var(--c-border-2)';
      dropZone.style.boxShadow   = 'none';
      dropZone.classList.remove('has-image', 'drag-over', 'scanning');
    }
    if (scanOverlay) scanOverlay.classList.remove('active');
    if (btnText)     btnText.style.display   = 'inline-flex';
    if (btnLoader)   btnLoader.style.display = 'none';
    if (submitBtn)   submitBtn.disabled      = false;
    if (clearBtn)    clearBtn.disabled       = false;
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', clearForm);
  }

  // ── Confidence bar animations ────────────────────────────────
  // Runs after page load (results are server-rendered)
  function animateBars() {
    var bars = document.querySelectorAll('[data-target]');
    bars.forEach(function (bar) {
      var target = parseFloat(bar.dataset.target) || 0;
      // Clamp to 0-100 range
      target = Math.min(Math.max(target, 0), 100);
      // Start at 0, animate to target after a short delay
      bar.style.width = '0%';
      bar.style.transition = 'none';
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          setTimeout(function () {
            bar.style.transition = 'width 1s cubic-bezier(.22,.68,0,1.1)';
            bar.style.width = target + '%';
          }, 150);
        });
      });
    });
  }

  // ── Inject keyframes ────────────────────────────────────────
  var style = document.createElement('style');
style.textContent = [
  '@keyframes shake {',
  '  0%,100%{transform:translateX(0)}',
  '  20%{transform:translateX(-8px)}',
  '  40%{transform:translateX(8px)}',
  '  60%{transform:translateX(-5px)}',
  '  80%{transform:translateX(5px)}',
  '}',
  '',
  '/* Scan line */',
  '.scan-overlay {',
  '  position:absolute;inset:0;pointer-events:none;overflow:hidden;',
  '  border-radius:inherit;z-index:3;',
  '}',
    '.scan-overlay.active::after {',
    '  content:"";',
    '  display:block;width:100%;height:3px;',
    '  background:linear-gradient(90deg,transparent,var(--c-green),transparent);',
    '  box-shadow:0 0 12px var(--c-green);',
    '  position:absolute;top:-3px;left:0;',
    '  animation:scan-line 1.2s linear infinite;',
    '}',
    '@keyframes scan-line {',
    '  0%  {top:-3px}',
    '  100%{top:100%}',
    '}',
  ].join('\n');
  document.head.appendChild(style);

  // ── Init on DOMContentLoaded ─────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', animateBars);
  } else {
    animateBars();
  }

})();

/**
 * Chrome Extension Popup Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const classifier = new ASLClassifier(window.ASL_MODEL_WEIGHTS);

    // UI elements
    const video = document.getElementById('popup-video');
    const canvas = document.getElementById('popup-canvas');
    const ctx = canvas.getContext('2d');
    const camPlaceholder = document.getElementById('popup-cam-placeholder');
    const btnToggleCam = document.getElementById('btn-toggle-cam');
    const headerStatus = document.getElementById('header-status');
    const btnOpenSidepanel = document.getElementById('btn-open-sidepanel');

    const predCharEl = document.getElementById('popup-pred-char');
    const predLabelEl = document.getElementById('popup-pred-label');
    const confidenceEl = document.getElementById('popup-confidence');
    const holdFillEl = document.getElementById('popup-hold-fill');
    const textBufferEl = document.getElementById('popup-text-buffer');

    const btnSpace = document.getElementById('btn-popup-space');
    const btnBack = document.getElementById('btn-popup-backspace');
    const btnClear = document.getElementById('btn-popup-clear');
    const btnSpeak = document.getElementById('btn-popup-speak');
    const btnCopy = document.getElementById('btn-popup-copy');
    const btnInject = document.getElementById('btn-inject-text');

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const dictList = document.getElementById('popup-dict-list');
    const searchInput = document.getElementById('popup-search');

    // UI Elements - Mode Select
    const btnPopupSpelling = document.getElementById('popup-mode-spelling');
    const btnPopupPhrases = document.getElementById('popup-mode-phrases');
    let isSpellingMode = true;

    if (btnPopupSpelling && btnPopupPhrases) {
        btnPopupSpelling.addEventListener('click', () => {
            isSpellingMode = true;
            btnPopupSpelling.style.background = '#6366f1';
            btnPopupSpelling.style.color = 'white';
            btnPopupSpelling.classList.add('active');
            btnPopupPhrases.style.background = 'transparent';
            btnPopupPhrases.style.color = '#94a3b8';
            btnPopupPhrases.classList.remove('active');
        });

        btnPopupPhrases.addEventListener('click', () => {
            isSpellingMode = false;
            btnPopupPhrases.style.background = '#6366f1';
            btnPopupPhrases.style.color = 'white';
            btnPopupPhrases.classList.add('active');
            btnPopupSpelling.style.background = 'transparent';
            btnPopupSpelling.style.color = '#94a3b8';
            btnPopupSpelling.classList.remove('active');
        });
    }

    let isCameraRunning = false;
    let hands = null;
    let stream = null;
    let currentPred = null;
    let holdStart = null;
    const HOLD_MS = 750;
    let lastTyped = null;
    let textBuffer = '';

    // Load saved text buffer from storage if any
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
        chrome.storage.local.get(['asl_saved_text'], (res) => {
            if (res.asl_saved_text) {
                textBuffer = res.asl_saved_text;
                textBufferEl.value = textBuffer;
            }
        });
    }

    // Tabs
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // Side Panel Launcher (if in popup mode)
    if (btnOpenSidepanel) {
        btnOpenSidepanel.addEventListener('click', () => {
            if (typeof chrome !== 'undefined' && chrome.runtime) {
                chrome.runtime.sendMessage({ action: 'open_side_panel' });
            }
        });
    }

    // Initialize MediaPipe Hands
    function initHands() {
        if (typeof Hands === 'undefined') return;
        hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.6
        });
        hands.onResults(onHandsResults);
    }

    initHands();

    // Camera Toggle
    btnToggleCam.addEventListener('click', async () => {
        if (!isCameraRunning) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 480, height: 360, facingMode: 'user' },
                    audio: false
                });
                video.srcObject = stream;
                await video.play();

                canvas.width = video.videoWidth || 480;
                canvas.height = video.videoHeight || 360;

                camPlaceholder.style.display = 'none';
                isCameraRunning = true;
                headerStatus.textContent = 'Tracking Active';
                headerStatus.style.color = '#34d399';
                requestAnimationFrame(processVideoFrame);
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        }
    });

    async function processVideoFrame() {
        if (!isCameraRunning) return;
        if (hands && video.readyState >= 2) {
            await hands.send({ image: video });
        }
        if (isCameraRunning) {
            requestAnimationFrame(processVideoFrame);
        }
    }

    function onHandsResults(results) {
        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);

        if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
            predCharEl.textContent = '—';
            predLabelEl.textContent = 'No hand detected';
            confidenceEl.textContent = '0%';
            holdFillEl.style.width = '0%';
            currentPred = null;
            holdStart = null;
            lastTyped = null;
            ctx.restore();
            return;
        }

        const lms = results.multiHandLandmarks[0];

        // Draw connections
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#6366f1';
        for (let i = 0; i < lms.length; i++) {
            const x = lms[i].x * canvas.width;
            const y = lms[i].y * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fillStyle = '#10b981';
            ctx.fill();
        }
        ctx.restore();

        // 1. Check special gestures (only if Phrases Mode is enabled)
        const special = !isSpellingMode ? CommonGestureDetector.detect(lms) : { detected: false };
        let label = '';
        let conf = 0;
        let isSpecial = false;

        if (special.detected && special.confidence > 0.85) {
            label = special.gesture;
            conf = special.confidence;
            isSpecial = true;
        } else {
            const features = classifier.preProcessLandmarks(lms, canvas.width, canvas.height);
            if (features) {
                const p = classifier.predict(features);
                label = p.label;
                conf = p.confidence;
            }
        }

        if (!label || conf < 0.45) {
            predCharEl.textContent = '—';
            predLabelEl.textContent = 'Unclear sign';
            confidenceEl.textContent = '0%';
            holdFillEl.style.width = '0%';
            return;
        }

        predCharEl.textContent = label;
        predLabelEl.textContent = isSpecial ? special.label : `Letter ${label}`;
        confidenceEl.textContent = `${Math.round(conf * 100)}%`;

        // Hold-to-type debounce
        const now = performance.now();
        if (currentPred === label) {
            if (!holdStart) holdStart = now;
            const elapsed = now - holdStart;
            holdFillEl.style.width = `${Math.min(100, Math.round((elapsed / HOLD_MS) * 100))}%`;

            if (elapsed >= HOLD_MS && lastTyped !== label) {
                if (isSpecial) {
                    appendBuffer(` ${label} `);
                } else {
                    appendBuffer(label);
                }
                lastTyped = label;
            }
        } else {
            currentPred = label;
            holdStart = now;
            lastTyped = null;
            holdFillEl.style.width = '0%';
        }
    }

    function appendBuffer(val) {
        textBuffer += val;
        textBufferEl.value = textBuffer;
        if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
            chrome.storage.local.set({ asl_saved_text: textBuffer });
        }
    }

    btnSpace.addEventListener('click', () => appendBuffer(' '));
    btnBack.addEventListener('click', () => {
        textBuffer = textBuffer.slice(0, -1);
        textBufferEl.value = textBuffer;
    });
    btnClear.addEventListener('click', () => {
        textBuffer = '';
        textBufferEl.value = '';
    });
    btnSpeak.addEventListener('click', () => {
        if (!textBuffer.trim()) return;
        const u = new SpeechSynthesisUtterance(textBuffer);
        window.speechSynthesis.speak(u);
    });
    btnCopy.addEventListener('click', async () => {
        if (!textBuffer) return;
        await navigator.clipboard.writeText(textBuffer);
        btnCopy.textContent = '✓ Copied';
        setTimeout(() => { btnCopy.textContent = '📋 Copy'; }, 1500);
    });

    // Insert text directly into active web page
    btnInjectText();
    function btnInjectText() {
        btnInject.addEventListener('click', () => {
            if (!textBuffer.trim()) return;
            if (typeof chrome !== 'undefined' && chrome.tabs) {
                chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
                    if (tabs && tabs[0]) {
                        chrome.tabs.sendMessage(
                            tabs[0].id,
                            { action: 'insert_text', text: textBuffer },
                            (response) => {
                                if (response && response.success) {
                                    btnInject.innerHTML = '<span>✓ Inserted into Page!</span>';
                                    setTimeout(() => {
                                        btnInject.innerHTML = '<span>✍️ Insert into Active Webpage</span>';
                                    }, 2000);
                                } else {
                                    alert('Could not insert: Please click inside a text input or editor on the active webpage first.');
                                }
                            }
                        );
                    }
                });
            }
        });
    }

    // Populate Dictionary list
    function renderDictList(filter = '') {
        dictList.innerHTML = '';
        const allItems = [
            ...ASL_DICTIONARY.alphabet,
            ...ASL_DICTIONARY.numbers,
            ...ASL_DICTIONARY.commonPhrases
        ];

        const q = filter.toLowerCase();
        const filtered = allItems.filter(i => 
            i.letter.toLowerCase().includes(q) ||
            i.name.toLowerCase().includes(q) ||
            i.description.toLowerCase().includes(q)
        );

        filtered.forEach(item => {
            const card = document.createElement('div');
            card.className = 'dict-item-card';
            card.innerHTML = `
                <div class="dict-item-letter">${item.letter}</div>
                <div class="dict-item-info">
                    <div class="dict-item-name">${item.name}</div>
                    <div class="dict-item-desc">${item.description}</div>
                </div>
            `;
            dictList.appendChild(card);
        });
    }

    searchInput.addEventListener('input', (e) => {
        renderDictList(e.target.value);
    });

    renderDictList();
});

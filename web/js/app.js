/**
 * ASL Web Application Controller
 * Handles UI interactions, Sign-to-Text buffer, hold-to-type stabilization,
 * Text-to-Speech (TTS), Autocomplete, ASL Showcase Dictionary, and Practice Mode.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Instantiate Core Modules
    const classifier = new ASLClassifier(window.ASL_MODEL_WEIGHTS);
    const stabilizer = new PredictionStabilizer({
        alpha: 0.45,
        switchThreshold: 0.42,
        holdThreshold: 0.22,
        debounceFrames: 2,
        gracePeriodMs: 250
    });
    
    // UI Elements - Mode Select
    const btnModeSpelling = document.getElementById('mode-spelling');
    const btnModePhrases = document.getElementById('mode-phrases');
    let isSpellingMode = true; // default to Spelling Mode

    btnModeSpelling.addEventListener('click', () => {
        isSpellingMode = true;
        btnModeSpelling.classList.add('active');
        btnModePhrases.classList.remove('active');
    });

    btnModePhrases.addEventListener('click', () => {
        isSpellingMode = false;
        btnModePhrases.classList.add('active');
        btnModeSpelling.classList.remove('active');
    });

    // UI Elements - Video & Canvas
    const videoElement = document.getElementById('webcam-video');
    const canvasElement = document.getElementById('output-canvas');
    const startCamBtn = document.getElementById('btn-start-camera');
    const stopCamBtn = document.getElementById('btn-stop-camera');
    const cameraOverlay = document.getElementById('camera-placeholder');
    const statusBadge = document.getElementById('status-badge');
    const fpsBadge = document.getElementById('fps-badge');

    // UI Elements - Prediction & Detection
    const predictedCharEl = document.getElementById('predicted-character');
    const confidenceValEl = document.getElementById('confidence-value');
    const confidenceBarEl = document.getElementById('confidence-bar-fill');
    const holdProgressEl = document.getElementById('hold-progress-fill');
    const gestureTypeBadge = document.getElementById('gesture-type-badge');

    // UI Elements - Text Buffer
    const textBufferEl = document.getElementById('sign-text-buffer');
    const wordCountEl = document.getElementById('word-count');
    const charCountEl = document.getElementById('char-count');
    const btnSpace = document.getElementById('btn-space');
    const btnBackspace = document.getElementById('btn-backspace');
    const btnClear = document.getElementById('btn-clear');
    const btnCopy = document.getElementById('btn-copy');
    const btnSpeak = document.getElementById('btn-speak');
    const suggestionsContainer = document.getElementById('word-suggestions');

    // UI Elements - Dictionary & Showcase
    const dictionaryGrid = document.getElementById('dictionary-grid');
    const searchInput = document.getElementById('dict-search-input');
    const categoryTabs = document.querySelectorAll('.dict-tab');
    const signModal = document.getElementById('sign-detail-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // UI Elements - Practice Mode
    const practiceCard = document.getElementById('practice-card');
    const practiceTargetEl = document.getElementById('practice-target-letter');
    const practiceScoreEl = document.getElementById('practice-score');
    const practiceStreakEl = document.getElementById('practice-streak');
    const btnNextPractice = document.getElementById('btn-next-practice');

    // Hold-to-type state
    let currentPrediction = null;
    let holdStartTime = null;
    const HOLD_THRESHOLD_MS = 750; // Hold sign for 0.75s to type
    let lastTypedChar = null;
    let textBuffer = '';

    // Autocomplete dictionary
    const COMMON_WORDS = [
        'HELLO', 'HELP', 'THANKS', 'THANK YOU', 'PLEASE', 'YES', 'NO', 'SORRY', 'GOOD', 'BAD',
        'NAME', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW', 'LOVE', 'YOU', 'ME', 'MY', 'FRIEND',
        'FAMILY', 'HOME', 'WATER', 'FOOD', 'EAT', 'DRINK', 'MORE', 'AGAIN', 'FINE', 'HAPPY'
    ];

    // Practice Mode State
    let practiceTarget = 'A';
    let practiceScore = 0;
    let practiceStreak = 0;
    let practiceMatchStartTime = null;

    // Head Gesture Detector (Sideways Head Nod -> Backspace)
    const headGestureDetector = typeof HeadGestureDetector !== 'undefined' ? new HeadGestureDetector({
        cooldownMs: 650,
        timeWindowMs: 850
    }) : null;

    function performBackspace(reason = 'Head Nod') {
        if (!textBuffer) return;
        textBuffer = textBuffer.slice(0, -1);
        updateTextBufferUI();

        gestureTypeBadge.textContent = `⌫ Backspace (${reason})`;
        gestureTypeBadge.className = 'badge badge-danger';

        const container = document.querySelector('.text-studio-card') || textBufferEl;
        if (container) {
            container.style.transition = 'border-color 0.15s ease';
            container.style.borderColor = '#f43f5e';
            setTimeout(() => { container.style.borderColor = ''; }, 300);
        }
    }

    // 1. Initialize Camera
    const cameraManager = new CameraManager(videoElement, canvasElement, (results, fps, faceResults) => {
        handleFrameResults(results, fps, faceResults);
    });

    startCamBtn.addEventListener('click', async () => {
        try {
            startCamBtn.disabled = true;
            statusBadge.textContent = 'Starting Camera...';
            await cameraManager.start();
            cameraOverlay.style.display = 'none';
            startCamBtn.style.display = 'none';
            stopCamBtn.style.display = 'inline-flex';
            statusBadge.textContent = 'Live Tracking';
            statusBadge.className = 'badge badge-success';
        } catch (err) {
            statusBadge.textContent = 'Camera Denied / Not Found';
            statusBadge.className = 'badge badge-danger';
            startCamBtn.disabled = false;
            alert('Unable to open camera. Please ensure camera permissions are allowed in your browser.');
        }
    });

    stopCamBtn.addEventListener('click', () => {
        cameraManager.stop();
        cameraOverlay.style.display = 'flex';
        startCamBtn.style.display = 'inline-flex';
        startCamBtn.disabled = false;
        stopCamBtn.style.display = 'none';
        statusBadge.textContent = 'Camera Off';
        statusBadge.className = 'badge badge-neutral';
        fpsBadge.textContent = '0 FPS';
        resetPredictionDisplay();
        if (headGestureDetector) headGestureDetector.reset();
    });

    // 2. Real-Time Inference & Hold-to-Type Loop
    function handleFrameResults(results, fps, faceResults) {
        fpsBadge.textContent = `${fps} FPS`;
        const now = performance.now();

        // 0. Check Sideways Head Nod (Head Shake "NO") -> Trigger Backspace
        if (headGestureDetector && faceResults && faceResults.multiFaceLandmarks && faceResults.multiFaceLandmarks.length > 0) {
            const faceLandmarks = faceResults.multiFaceLandmarks[0];
            const headRes = headGestureDetector.update(faceLandmarks, now);
            if (headRes.detected && headRes.action === 'BACKSPACE') {
                performBackspace('Head Nod');
            }
        }

        const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
        let predictedLabel = '';
        let confidence = 0;
        let isCommonPhrase = false;

        if (hasHands) {
            const landmarks = results.multiHandLandmarks[0];

            // 1. Check for common special gestures first (only if Phrases Mode is enabled)
            const commonGesture = !isSpellingMode ? CommonGestureDetector.detect(landmarks) : { detected: false };

            if (commonGesture.detected && commonGesture.confidence > 0.85) {
                predictedLabel = commonGesture.gesture;
                confidence = commonGesture.confidence;
                isCommonPhrase = true;
                gestureTypeBadge.textContent = 'Common Sign';
                gestureTypeBadge.className = 'badge badge-accent';
                stabilizer.reset();
            } else {
                // 2. Classify alphabet keypoint (A-Z) using deep network & stabilizer
                const features = classifier.preProcessLandmarks(
                    landmarks,
                    canvasElement.width,
                    canvasElement.height
                );

                if (features) {
                    const rawPred = classifier.predict(features);
                    const stabilized = stabilizer.update(
                        rawPred.probabilities,
                        rawPred.label,
                        rawPred.confidence,
                        now
                    );

                    if (stabilized.detected) {
                        predictedLabel = stabilized.label;
                        confidence = stabilized.confidence;
                        isCommonPhrase = false;
                        gestureTypeBadge.textContent = 'Alphabet (A-Z)';
                        gestureTypeBadge.className = 'badge badge-primary';
                    }
                }
            }
        } else {
            // Hand not detected in this frame: check grace period to prevent UI flicker
            const stabilized = stabilizer.getState(now);
            if (stabilized.detected) {
                predictedLabel = stabilized.label;
                confidence = stabilized.confidence;
                isCommonPhrase = false;
            }
        }

        if (!predictedLabel || confidence < 0.35) {
            if (!hasHands) {
                resetPredictionDisplay();
            }
            return;
        }

        // Update UI Prediction Display
        predictedCharEl.textContent = predictedLabel;
        const confidencePct = Math.round(confidence * 100);
        confidenceValEl.textContent = `${confidencePct}%`;
        confidenceBarEl.style.width = `${confidencePct}%`;

        // Check Practice Mode match
        checkPracticeMatch(predictedLabel, confidence);

        // Hold-to-type debouncer
        if (currentPrediction === predictedLabel) {
            if (!holdStartTime) holdStartTime = now;
            const elapsed = now - holdStartTime;
            const progressPct = Math.min(100, Math.round((elapsed / HOLD_THRESHOLD_MS) * 100));
            holdProgressEl.style.width = `${progressPct}%`;

            if (elapsed >= HOLD_THRESHOLD_MS && lastTypedChar !== predictedLabel) {
                // Type character/word into buffer
                if (isCommonPhrase) {
                    appendSymbolToBuffer(` ${predictedLabel} `);
                } else {
                    appendSymbolToBuffer(predictedLabel);
                }
                lastTypedChar = predictedLabel;
                // Visual bounce effect
                textBufferEl.classList.add('buffer-flash');
                setTimeout(() => textBufferEl.classList.remove('buffer-flash'), 250);
            }
        } else {
            currentPrediction = predictedLabel;
            holdStartTime = now;
            lastTypedChar = null;
            holdProgressEl.style.width = '0%';
        }
    }

    function resetPredictionDisplay() {
        currentPrediction = null;
        holdStartTime = null;
        lastTypedChar = null;
        stabilizer.reset();
        predictedCharEl.textContent = '—';
        confidenceValEl.textContent = '0%';
        confidenceBarEl.style.width = '0%';
        holdProgressEl.style.width = '0%';
        gestureTypeBadge.textContent = 'Ready';
        gestureTypeBadge.className = 'badge badge-neutral';
    }

    // 3. Text Buffer Actions
    function appendSymbolToBuffer(str) {
        textBuffer += str;
        updateTextBufferUI();
    }

    function updateTextBufferUI() {
        textBufferEl.value = textBuffer;
        charCountEl.textContent = `${textBuffer.length} chars`;
        const words = textBuffer.trim().split(/\s+/).filter(w => w.length > 0);
        wordCountEl.textContent = `${words.length} words`;
        updateWordSuggestions();
    }

    btnSpace.addEventListener('click', () => {
        appendSymbolToBuffer(' ');
    });

    btnBackspace.addEventListener('click', () => {
        textBuffer = textBuffer.slice(0, -1);
        updateTextBufferUI();
    });

    btnClear.addEventListener('click', () => {
        textBuffer = '';
        updateTextBufferUI();
    });

    btnCopy.addEventListener('click', async () => {
        if (!textBuffer) return;
        try {
            await navigator.clipboard.writeText(textBuffer);
            const orig = btnCopy.innerHTML;
            btnCopy.innerHTML = '<span class="icon">✓</span> Copied!';
            setTimeout(() => { btnCopy.innerHTML = orig; }, 2000);
        } catch (err) {
            console.error('Copy failed', err);
        }
    });

    // 4. Text-to-Speech (TTS)
    btnSpeak.addEventListener('click', () => {
        if (!textBuffer.trim()) return;
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel(); // Stop any pending speech
            const utterance = new SpeechSynthesisUtterance(textBuffer);
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        } else {
            alert('Text-to-Speech is not supported in this browser.');
        }
    });

    // 5. Smart Autocomplete & Suggestions
    function updateWordSuggestions() {
        suggestionsContainer.innerHTML = '';
        if (!textBuffer.trim()) return;

        const words = textBuffer.trim().split(/\s+/);
        const currentWord = words[words.length - 1].toUpperCase();
        if (currentWord.length < 2) return;

        const matches = COMMON_WORDS.filter(w => w.startsWith(currentWord) && w !== currentWord).slice(0, 4);
        matches.forEach(match => {
            const chip = document.createElement('button');
            chip.className = 'suggestion-chip';
            chip.textContent = match;
            chip.addEventListener('click', () => {
                // Replace current word with autocomplete
                words[words.length - 1] = match;
                textBuffer = words.join(' ') + ' ';
                updateTextBufferUI();
            });
            suggestionsContainer.appendChild(chip);
        });
    }

    // 6. ASL Symbol Showcase & Interactive Dictionary
    function renderDictionary(category = 'all', filterText = '') {
        dictionaryGrid.innerHTML = '';
        let items = [];

        if (category === 'all' || category === 'alphabet') {
            items = items.concat(ASL_DICTIONARY.alphabet);
        }
        if (category === 'all' || category === 'numbers') {
            items = items.concat(ASL_DICTIONARY.numbers);
        }
        if (category === 'all' || category === 'phrases') {
            items = items.concat(ASL_DICTIONARY.commonPhrases);
        }

        if (filterText) {
            const q = filterText.toLowerCase();
            items = items.filter(item => 
                item.letter.toLowerCase().includes(q) ||
                item.name.toLowerCase().includes(q) ||
                item.description.toLowerCase().includes(q) ||
                (item.commonWords && item.commonWords.some(w => w.toLowerCase().includes(q)))
            );
        }

        items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'symbol-card';
            card.innerHTML = `
                <div class="symbol-card-badge">${item.category.toUpperCase()}</div>
                <div class="symbol-card-letter">${item.letter}</div>
                <div class="symbol-card-name">${item.name}</div>
                <div class="symbol-card-desc">${item.description}</div>
                <button class="symbol-card-btn" data-letter="${item.letter}">View Details & Practice</button>
            `;

            card.querySelector('.symbol-card-btn').addEventListener('click', () => {
                openSignDetailModal(item);
            });

            dictionaryGrid.appendChild(card);
        });

        if (items.length === 0) {
            dictionaryGrid.innerHTML = `<div class="empty-state">No matching ASL signs found for "${filterText}".</div>`;
        }
    }

    // Category Tabs
    categoryTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            categoryTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            renderDictionary(tab.dataset.category, searchInput.value);
        });
    });

    searchInput.addEventListener('input', (e) => {
        const activeTab = document.querySelector('.dict-tab.active');
        renderDictionary(activeTab ? activeTab.dataset.category : 'all', e.target.value);
    });

    function openSignDetailModal(item) {
        document.getElementById('modal-sign-title').textContent = `${item.name} (${item.letter})`;
        document.getElementById('modal-sign-letter').textContent = item.letter;
        document.getElementById('modal-sign-desc').textContent = item.description;
        document.getElementById('modal-sign-tips').textContent = item.tips || 'Keep wrist steady and fingers defined.';
        
        const wordsEl = document.getElementById('modal-common-words');
        wordsEl.innerHTML = (item.commonWords || []).map(w => `<span class="tag-pill">${w}</span>`).join(' ');

        document.getElementById('modal-btn-practice').onclick = () => {
            signModal.style.display = 'none';
            setPracticeTarget(item.letter);
            document.getElementById('practice-section').scrollIntoView({ behavior: 'smooth' });
        };

        signModal.style.display = 'flex';
    }

    modalCloseBtn.addEventListener('click', () => {
        signModal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === signModal) {
            signModal.style.display = 'none';
        }
    });

    // 7. Interactive Practice & Quiz Mode
    function setPracticeTarget(letter) {
        practiceTarget = letter;
        practiceTargetEl.textContent = letter;
        practiceMatchStartTime = null;
    }

    function checkPracticeMatch(predictedLabel, confidence) {
        if (predictedLabel === practiceTarget && confidence >= 0.70) {
            const now = performance.now();
            if (!practiceMatchStartTime) practiceMatchStartTime = now;
            if (now - practiceMatchStartTime >= 600) {
                // Practice match successful!
                practiceScore += 10;
                practiceStreak += 1;
                practiceScoreEl.textContent = practiceScore;
                practiceStreakEl.textContent = practiceStreak;

                practiceCard.classList.add('practice-success');
                setTimeout(() => practiceCard.classList.remove('practice-success'), 800);

                // Auto-advance to next random sign
                setTimeout(pickNextPracticeTarget, 1000);
                practiceMatchStartTime = null;
            }
        } else {
            practiceMatchStartTime = null;
        }
    }

    function pickNextPracticeTarget() {
        const pool = ASL_DICTIONARY.alphabet.map(a => a.letter);
        const randomLetter = pool[Math.floor(Math.random() * pool.length)];
        setPracticeTarget(randomLetter);
    }

    btnNextPractice.addEventListener('click', pickNextPracticeTarget);

    // Initial renders
    renderDictionary('all');
    setPracticeTarget('A');
});

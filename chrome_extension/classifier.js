/**
 * ASL KeyPoint Classifier & Prediction Stabilizer (JavaScript Implementation)
 * Exact neural network reproduction of model/keypoint_classifier/keypoint_classifier.py
 * Architecture:
 *   Input (42) -> BatchNorm -> Dense(256)+Mish -> Dense(256)+Mish -> Dense(128)+Mish -> Dense(64)+Mish -> Dense(26)+Softmax
 */

class ASLClassifier {
    constructor(weights = null) {
        this.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ];

        this.weights = weights || (typeof window !== 'undefined' ? window.ASL_MODEL_WEIGHTS : null);
        if (!this.weights) {
            console.warn('ASLClassifier: Weights not found in window.ASL_MODEL_WEIGHTS, waiting for manual load.');
        }
    }

    setWeights(weights) {
        this.weights = weights;
    }

    /**
     * Softplus: log(1 + exp(x)) computed stably
     */
    _softplus(x) {
        return Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
    }

    /**
     * Mish activation: x * tanh(softplus(x))
     */
    _mish(x) {
        return x * Math.tanh(this._softplus(x));
    }

    /**
     * Dense layer multiplication: y = x @ W + b
     */
    _dense(x, W, b) {
        const outDim = b.length;
        const inDim = x.length;
        const out = new Float32Array(outDim);

        for (let j = 0; j < outDim; j++) {
            let sum = b[j];
            for (let i = 0; i < inDim; i++) {
                sum += x[i] * W[i][j];
            }
            out[j] = sum;
        }
        return out;
    }

    /**
     * Softmax over an array of logits
     */
    _softmax(logits) {
        let maxVal = -Infinity;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) maxVal = logits[i];
        }

        let sum = 0;
        const expVals = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            expVals[i] = Math.exp(logits[i] - maxVal);
            sum += expVals[i];
        }

        const probs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
            probs[i] = expVals[i] / (sum || 1);
        }
        return probs;
    }

    /**
     * Preprocesses 21 MediaPipe hand landmarks into 42 normalized features
     * Exact match to Python's calc_landmark_list and pre_process_landmark
     * @param {Array<{x: number, y: number, z: number}>} landmarks - MediaPipe 21 landmarks
     * @param {number} imageWidth
     * @param {number} imageHeight
     */
    preProcessLandmarks(landmarks, imageWidth = 640, imageHeight = 480) {
        if (!landmarks || landmarks.length !== 21) {
            return null;
        }

        // 1. Convert normalized coordinates to pixel coordinates
        const landmarkPoints = [];
        for (let i = 0; i < 21; i++) {
            const lm = landmarks[i];
            const px = Math.min(Math.floor(lm.x * imageWidth), imageWidth - 1);
            const py = Math.min(Math.floor(lm.y * imageHeight), imageHeight - 1);
            landmarkPoints.push([px, py]);
        }

        // 2. Relative coordinates (origin at wrist / landmark 0)
        const baseX = landmarkPoints[0][0];
        const baseY = landmarkPoints[0][1];

        const relativePoints = [];
        for (let i = 0; i < 21; i++) {
            relativePoints.push([
                landmarkPoints[i][0] - baseX,
                landmarkPoints[i][1] - baseY
            ]);
        }

        // 3. Flatten to 42 elements
        const flattened = [];
        for (let i = 0; i < 21; i++) {
            flattened.push(relativePoints[i][0]);
            flattened.push(relativePoints[i][1]);
        }

        // 4. Max absolute normalization
        let maxVal = 0;
        for (let i = 0; i < flattened.length; i++) {
            const absVal = Math.abs(flattened[i]);
            if (absVal > maxVal) maxVal = absVal;
        }

        if (maxVal === 0) maxVal = 1.0;

        const normalized = new Float32Array(42);
        for (let i = 0; i < flattened.length; i++) {
            normalized[i] = flattened[i] / maxVal;
        }

        return normalized;
    }

    /**
     * Predict ASL alphabet from 42 normalized landmark features
     * @param {Float32Array|Array<number>} features - 42 normalized landmark coordinates
     * @returns {{label: string, index: number, confidence: number, probabilities: Array<number>}}
     */
    predict(features) {
        if (!this.weights) {
            throw new Error('Weights not loaded in ASLClassifier');
        }

        const {
            bn_gamma, bn_beta, bn_mean, bn_var,
            w0, b0,
            w1, b1,
            w2, b2,
            w3, b3,
            w4, b4
        } = this.weights;

        // 1. Batch Normalization: (x - mean) / sqrt(var + 0.001) * gamma + beta
        const xBn = new Float32Array(42);
        for (let i = 0; i < 42; i++) {
            const std = Math.sqrt(bn_var[i] + 0.001);
            xBn[i] = ((features[i] - bn_mean[i]) / std) * bn_gamma[i] + bn_beta[i];
        }

        // 2. Layer 0: Dense(256) + Mish
        const l0 = this._dense(xBn, w0, b0);
        for (let i = 0; i < l0.length; i++) l0[i] = this._mish(l0[i]);

        // 3. Layer 1: Dense(256) + Mish
        const l1 = this._dense(l0, w1, b1);
        for (let i = 0; i < l1.length; i++) l1[i] = this._mish(l1[i]);

        // 4. Layer 2: Dense(128) + Mish
        const l2 = this._dense(l1, w2, b2);
        for (let i = 0; i < l2.length; i++) l2[i] = this._mish(l2[i]);

        // 5. Layer 3: Dense(64) + Mish
        const l3 = this._dense(l2, w3, b3);
        for (let i = 0; i < l3.length; i++) l3[i] = this._mish(l3[i]);

        // 6. Layer 4: Dense(26) + Softmax
        const logits = this._dense(l3, w4, b4);
        const probs = this._softmax(logits);

        // Find argmax
        let maxIndex = 0;
        let maxProb = -1;
        for (let i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }

        return {
            label: this.labels[maxIndex],
            index: maxIndex,
            confidence: maxProb,
            probabilities: Array.from(probs)
        };
    }
}

/**
 * Prediction Stabilizer
 * Eliminates alphabet prediction flickering and rapid jitter using:
 * 1. Exponential Moving Average (EMA) of softmax probability distributions
 * 2. Confidence hysteresis (high activation threshold to switch, lower to maintain)
 * 3. Consecutive-frame debouncing
 * 4. Tracking dropout grace period (holds state through 1-2 dropped frames instead of flashing blank)
 */
class PredictionStabilizer {
    constructor(options = {}) {
        this.alpha = options.alpha !== undefined ? options.alpha : 0.45;
        this.switchThreshold = options.switchThreshold || 0.42;
        this.holdThreshold = options.holdThreshold || 0.22;
        this.debounceFrames = options.debounceFrames || 2;
        this.gracePeriodMs = options.gracePeriodMs || 220;

        this.labels = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ];

        this.smoothedProbabilities = new Float32Array(26);
        this.currentLabel = null;
        this.currentIndex = -1;
        this.currentConfidence = 0;
        this.candidateLabel = null;
        this.candidateCount = 0;
        this.lastDetectionTime = 0;
        this.isInitialized = false;
    }

    /**
     * Update stabilizer with new raw frame prediction
     */
    update(rawProbabilities, rawLabel, rawConfidence, timestamp = performance.now()) {
        this.lastDetectionTime = timestamp;

        if (!this.isInitialized) {
            for (let i = 0; i < 26; i++) {
                this.smoothedProbabilities[i] = rawProbabilities[i];
            }
            this.isInitialized = true;
        } else {
            // Apply Exponential Moving Average: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            for (let i = 0; i < 26; i++) {
                this.smoothedProbabilities[i] = this.alpha * rawProbabilities[i] + (1 - this.alpha) * this.smoothedProbabilities[i];
            }
        }

        // Find top smoothed class
        let topIndex = 0;
        let topProb = -1;
        for (let i = 0; i < 26; i++) {
            if (this.smoothedProbabilities[i] > topProb) {
                topProb = this.smoothedProbabilities[i];
                topIndex = i;
            }
        }
        const topLabel = this.labels[topIndex];

        // Hysteresis & Debounce decision logic
        if (!this.currentLabel) {
            if (topProb >= this.switchThreshold) {
                this.currentLabel = topLabel;
                this.currentIndex = topIndex;
                this.currentConfidence = topProb;
                this.candidateLabel = null;
                this.candidateCount = 0;
            }
        } else if (this.currentLabel === topLabel) {
            // Reinforce current active label
            this.currentConfidence = topProb;
            this.candidateLabel = null;
            this.candidateCount = 0;
        } else {
            // Different candidate trying to take over
            const currentProb = this.smoothedProbabilities[this.currentIndex] || 0;
            
            if (this.candidateLabel === topLabel) {
                this.candidateCount++;
            } else {
                this.candidateLabel = topLabel;
                this.candidateCount = 1;
            }

            // Switch conditions:
            // 1. Candidate exceeds switch threshold AND leads for debounceFrames, OR
            // 2. Current label confidence collapsed below holdThreshold and candidate is significantly stronger
            const shouldSwitch = (topProb >= this.switchThreshold && this.candidateCount >= this.debounceFrames) ||
                                 (currentProb < this.holdThreshold && topProb > currentProb + 0.15);

            if (shouldSwitch) {
                this.currentLabel = topLabel;
                this.currentIndex = topIndex;
                this.currentConfidence = topProb;
                this.candidateLabel = null;
                this.candidateCount = 0;
            } else {
                // Maintain current label with its current smoothed confidence
                this.currentConfidence = Math.max(currentProb, this.holdThreshold);
            }
        }

        return this.getState();
    }

    /**
     * Get current stabilized state with grace period handling
     */
    getState(timestamp = performance.now()) {
        if (!this.currentLabel) {
            return {
                detected: false,
                label: null,
                index: -1,
                confidence: 0,
                probabilities: Array.from(this.smoothedProbabilities)
            };
        }

        const elapsedSinceDetection = timestamp - this.lastDetectionTime;
        if (elapsedSinceDetection > this.gracePeriodMs) {
            this.reset();
            return {
                detected: false,
                label: null,
                index: -1,
                confidence: 0,
                probabilities: Array.from(this.smoothedProbabilities)
            };
        }

        return {
            detected: true,
            label: this.currentLabel,
            index: this.currentIndex,
            confidence: this.currentConfidence,
            probabilities: Array.from(this.smoothedProbabilities),
            isGracePeriod: elapsedSinceDetection > 40
        };
    }

    reset() {
        this.currentLabel = null;
        this.currentIndex = -1;
        this.currentConfidence = 0;
        this.candidateLabel = null;
        this.candidateCount = 0;
        this.isInitialized = false;
        this.smoothedProbabilities.fill(0);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ASLClassifier, PredictionStabilizer };
} else if (typeof window !== 'undefined') {
    window.ASLClassifier = ASLClassifier;
    window.PredictionStabilizer = PredictionStabilizer;
}


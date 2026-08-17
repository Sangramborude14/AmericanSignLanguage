/**
 * ASL Sideways Head Nod / Head Shake Detector (Backspace Action)
 * Detects side-to-side head shaking ("no" / sideways head nod motion)
 * Triggers backspace to erase the last typed alphabet character from the sign text buffer.
 */

class HeadGestureDetector {
    constructor(options = {}) {
        this.cooldownMs = options.cooldownMs || 650; // Minimum time between backspace triggers
        this.timeWindowMs = options.timeWindowMs || 850; // Time window for sideways nod sequence

        // Yaw thresholds (horizontal head rotation)
        this.yawLeftThresh = -0.12;   // Nose shifted left
        this.yawRightThresh = 0.12;    // Nose shifted right
        
        // Roll thresholds (lateral head tilt)
        this.rollLeftThresh = -12.0;   // Head tilted left (degrees)
        this.rollRightThresh = 12.0;   // Head tilted right (degrees)

        this.lastTriggerTime = -10000;
        this.history = [];
        this.currentState = 'CENTER';
        this.onBackspaceCallback = options.onBackspaceCallback || null;
    }

    /**
     * Process 468 MediaPipe FaceMesh landmarks
     * @param {Array<{x: number, y: number, z: number}>} faceLandmarks
     * @param {number} timestamp
     * @returns {{detected: boolean, action: string|null, reason: string|null}}
     */
    update(faceLandmarks, timestamp = performance.now()) {
        if (!faceLandmarks || faceLandmarks.length < 264) {
            return { detected: false, action: null, reason: null };
        }

        // Check cooldown
        if (timestamp - this.lastTriggerTime < this.cooldownMs) {
            return { detected: false, action: null, reason: 'cooldown' };
        }

        // Key Landmarks in MediaPipe FaceMesh:
        // 1: Nose tip
        // 33: Left eye outer corner
        // 263: Right eye outer corner
        const nose = faceLandmarks[1];
        const leftEye = faceLandmarks[33];
        const rightEye = faceLandmarks[263];

        if (!nose || !leftEye || !rightEye) {
            return { detected: false, action: null, reason: null };
        }

        const eyeDx = rightEye.x - leftEye.x;
        const eyeDy = rightEye.y - leftEye.y;
        const eyeDistance = Math.hypot(eyeDx, eyeDy);

        if (eyeDistance < 0.01) return { detected: false, action: null, reason: null };

        // 1. Compute Yaw (Horizontal Offset)
        const eyeMidX = (leftEye.x + rightEye.x) / 2;
        const yawOffset = (nose.x - eyeMidX) / eyeDistance;

        // 2. Compute Roll (Lateral Angle in degrees)
        const rollAngle = Math.atan2(eyeDy, eyeDx) * (180 / Math.PI);

        // Determine current frame state
        let newState = 'CENTER';
        if (yawOffset < this.yawLeftThresh || rollAngle < this.rollLeftThresh) {
            newState = 'LEFT';
        } else if (yawOffset > this.yawRightThresh || rollAngle > this.rollRightThresh) {
            newState = 'RIGHT';
        }

        // State transition tracking
        if (newState !== this.currentState) {
            if (newState === 'LEFT' || newState === 'RIGHT') {
                this.history.push({ state: newState, time: timestamp });
                // Clean old states outside time window
                this.history = this.history.filter(item => timestamp - item.time <= this.timeWindowMs);
            }
            this.currentState = newState;
        }

        // Trigger condition: History contains both 'LEFT' and 'RIGHT' within time window
        if (this.history.length >= 2) {
            const states = this.history.map(h => h.state);
            const hasLeft = states.includes('LEFT');
            const hasRight = states.includes('RIGHT');

            if (hasLeft && hasRight) {
                this.lastTriggerTime = timestamp;
                this.history = [];
                this.currentState = 'CENTER';

                if (this.onBackspaceCallback) {
                    this.onBackspaceCallback();
                }

                return {
                    detected: true,
                    action: 'BACKSPACE',
                    reason: 'Head nod sidewise detected'
                };
            }
        }

        return { detected: false, action: null, reason: null };
    }

    reset() {
        this.history = [];
        this.currentState = 'CENTER';
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HeadGestureDetector };
} else if (typeof window !== 'undefined') {
    window.HeadGestureDetector = HeadGestureDetector;
}

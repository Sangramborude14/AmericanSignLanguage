/**
 * ASL Camera & MediaPipe Hand Tracking Manager
 * Handles camera capture, MediaPipe Hands pipeline, and 60 FPS HTML5 Canvas visual overlay
 */

class CameraManager {
    constructor(videoElement, canvasElement, onResultsCallback) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.onResults = onResultsCallback;

        this.hands = null;
        this.camera = null;
        this.isRunning = false;

        // FPS tracking
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();

        // Finger connections pairs for hand skeleton visualization
        this.connections = [
            // Thumb
            [0, 1], [1, 2], [2, 3], [3, 4],
            // Index finger
            [0, 5], [5, 6], [6, 7], [7, 8],
            // Middle finger
            [0, 9], [9, 10], [10, 11], [11, 12],
            // Ring finger
            [0, 13], [13, 14], [14, 15], [15, 16],
            // Pinky
            [0, 17], [17, 18], [18, 19], [19, 20],
            // Palm base
            [5, 9], [9, 13], [13, 17]
        ];

        this.initMediaPipe();
    }

    initMediaPipe() {
        if (typeof Hands === 'undefined') {
            console.error('MediaPipe Hands library not loaded');
            return;
        }

        this.hands = new Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });

        this.hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.6
        });

        this.hands.onResults((results) => {
            this.handleResults(results);
        });
    }

    async start() {
        if (this.isRunning) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.srcObject = stream;
            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });

            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;

            this.isRunning = true;
            this.processLoop();
            return true;
        } catch (err) {
            console.error('Error opening camera:', err);
            throw err;
        }
    }

    stop() {
        this.isRunning = false;
        if (this.video && this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
        }
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    async processLoop() {
        if (!this.isRunning) return;

        // Calculate FPS
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsUpdate >= 500) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
            this.frameCount = 0;
            this.lastFpsUpdate = now;
        }

        if (this.hands && this.video.readyState >= 2) {
            await this.hands.send({ image: this.video });
        }

        if (this.isRunning) {
            requestAnimationFrame(() => this.processLoop());
        }
    }

    handleResults(results) {
        const { width, height } = this.canvas;
        this.ctx.save();
        this.ctx.clearRect(0, 0, width, height);

        // Mirror display
        this.ctx.translate(width, 0);
        this.ctx.scale(-1, 1);

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            for (let h = 0; h < results.multiHandLandmarks.length; h++) {
                const landmarks = results.multiHandLandmarks[h];
                const handedness = results.multiHandedness && results.multiHandedness[h]
                    ? results.multiHandedness[h].label
                    : 'Hand';

                // Draw bones/connections
                this.drawSkeleton(landmarks, width, height);
                // Draw joint landmarks
                this.drawLandmarks(landmarks, width, height);
                // Draw bounding box
                const bbox = this.calculateBoundingBox(landmarks, width, height);
                this.drawBoundingBox(bbox);
            }
        }

        this.ctx.restore();

        // Pass results up to application controller
        if (this.onResults) {
            this.onResults(results, this.fps);
        }
    }

    drawSkeleton(landmarks, width, height) {
        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.75)'; // Indigo
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';

        for (const [startIdx, endIdx] of this.connections) {
            const start = landmarks[startIdx];
            const end = landmarks[endIdx];

            this.ctx.beginPath();
            this.ctx.moveTo(start.x * width, start.y * height);
            this.ctx.lineTo(end.x * width, end.y * height);
            this.ctx.stroke();
        }
    }

    drawLandmarks(landmarks, width, height) {
        for (let i = 0; i < landmarks.length; i++) {
            const lm = landmarks[i];
            const x = lm.x * width;
            const y = lm.y * height;

            // Fingertips (4, 8, 12, 16, 20) are accented emerald/cyan
            const isFingertip = [4, 8, 12, 16, 20].includes(i);
            const isWrist = i === 0;

            this.ctx.beginPath();
            this.ctx.arc(x, y, isFingertip ? 6 : (isWrist ? 7 : 4), 0, 2 * Math.PI);
            this.ctx.fillStyle = isFingertip ? '#10b981' : (isWrist ? '#f59e0b' : '#ffffff');
            this.ctx.fill();

            this.ctx.lineWidth = 2;
            this.ctx.strokeStyle = '#0f172a';
            this.ctx.stroke();
        }
    }

    calculateBoundingBox(landmarks, width, height) {
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        for (const lm of landmarks) {
            const x = lm.x * width;
            const y = lm.y * height;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        const padding = 24;
        return {
            x: Math.max(0, minX - padding),
            y: Math.max(0, minY - padding),
            width: Math.min(width, (maxX - minX) + padding * 2),
            height: Math.min(height, (maxY - minY) + padding * 2)
        };
    }

    drawBoundingBox(bbox) {
        this.ctx.strokeStyle = 'rgba(16, 185, 129, 0.6)';
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([8, 6]);
        this.ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
        this.ctx.setLineDash([]);
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CameraManager };
} else if (typeof window !== 'undefined') {
    window.CameraManager = CameraManager;
}

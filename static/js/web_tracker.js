/**
 * AttentionAI — Browser-Native Client Tracker
 * Runs 100% inside the browser via HTML5 getUserMedia and MediaPipe FaceMesh.
 * Enables zero-server deployment on Vercel, Netlify, GitHub Pages, etc.
 */

"use strict";

(function (window) {
  // Key Landmark Indices (MediaPipe 468+10 iris mesh)
  const LEFT_EAR_IDS  = [362, 385, 387, 263, 373, 380];
  const RIGHT_EAR_IDS = [33, 160, 158, 133, 153, 144];
  const LEFT_IRIS_IDS = [474, 475, 476, 477];
  const RIGHT_IRIS_IDS= [469, 470, 471, 472];

  // Head Pose Anchor Indices
  const NOSE_TIP   = 1;
  const CHIN       = 152;
  const LEFT_EYE   = 263;
  const RIGHT_EYE  = 33;
  const LEFT_MOUTH = 287;
  const RIGHT_MOUTH= 57;

  function dist(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
  }

  function computeEAR(landmarks, indices, width, height) {
    const pts = indices.map(i => ({ x: landmarks[i].x * width, y: landmarks[i].y * height }));
    const v1 = dist(pts[1], pts[5]);
    const v2 = dist(pts[2], pts[4]);
    const h  = dist(pts[0], pts[3]);
    return h === 0 ? 0.0 : (v1 + v2) / (2.0 * h);
  }

  function getIrisCenter(landmarks, indices, width, height) {
    let sx = 0, sy = 0;
    for (const i of indices) {
      sx += landmarks[i].x * width;
      sy += landmarks[i].y * height;
    }
    return { x: sx / indices.length, y: sy / indices.length };
  }

  class WebTracker {
    constructor(canvasEl) {
      this.canvas = canvasEl;
      this.ctx = canvasEl.getContext("2d");
      this.video = document.createElement("video");
      this.video.setAttribute("playsinline", "true");
      this.video.setAttribute("autoplay", "true");
      this.video.setAttribute("muted", "true");
      this.video.style.display = "none";
      document.body.appendChild(this.video);

      // Tracking variables
      this.isRunning = false;
      this.faceMesh = null;
      this.camera = null;

      // History & metrics
      this.consecClosedFrames = 0;
      this.blinkCount = 0;
      this.blinkTimes = [];
      this.earHistory = [];

      // Attention session tracking
      this.sessionStartTime = Date.now();
      this.lastTickTime = Date.now();
      this.durations = { Focused: 0, Distracted: 0, Drowsy: 0 };
      this.currentState = "No Face";
      this.currentConfidence = 1.0;
      this.timeline = []; // { ts: seconds, state: string }

      // Streaks
      this.streakState = null;
      this.streakStart = Date.now();
      this.bestFocusStreak = 0;
      this.longestDistract = 0;

      // FPS tracking
      this.lastFrameTime = performance.now();
      this.fps = 0;
      this.fpsSamples = [];
    }

    async start(callbacks = {}) {
      this.callbacks = callbacks;
      this.isRunning = true;

      // Draw initial loading indicator on canvas
      this.ctx.fillStyle = "#070b12";
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.fillStyle = "#6ee7f7";
      this.ctx.font = "14px Inter, sans-serif";
      this.ctx.textAlign = "center";
      this.ctx.fillText("Requesting camera access…", this.canvas.width / 2, this.canvas.height / 2);

      // Clean up previous active stream if any
      if (this.stream) {
        try { this.stream.getTracks().forEach(t => t.stop()); } catch (_) {}
        this.stream = null;
      }
      if (this.video && this.video.srcObject) {
        try { this.video.srcObject.getTracks().forEach(t => t.stop()); } catch (_) {}
        this.video.srcObject = null;
      }

      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("getUserMedia is not supported by your browser");
        }

        let stream = null;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false,
          });
        } catch (firstErr) {
          console.warn("Retrying with simple video constraint...", firstErr);
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        }

        this.stream = stream;
        this.video.srcObject = stream;
        this.video.muted = true;
        await this.video.play();
      } catch (err) {
        console.error("Camera access failed:", err);
        if (callbacks.onError) callbacks.onError(err);
        return false;
      }

      this.ctx.fillText("Loading MediaPipe FaceMesh AI…", this.canvas.width / 2, this.canvas.height / 2);

      // Wait for MediaPipe FaceMesh script if still loading from CDN
      let waitCount = 0;
      while (typeof window.FaceMesh === "undefined" && waitCount < 50) {
        await new Promise(r => setTimeout(r, 100));
        waitCount++;
      }

      if (typeof window.FaceMesh === "undefined") {
        const err = new Error("MediaPipe FaceMesh script not loaded from CDN");
        console.error(err);
        if (callbacks.onError) callbacks.onError(err);
        return false;
      }

      this.faceMesh = new window.FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      this.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      this.faceMesh.onResults(this.onResults.bind(this));

      // Continuous loop
      let isProcessing = false;
      const processFrame = async () => {
        if (!this.isRunning) return;
        if (this.video.readyState >= 2 && !isProcessing) {
          isProcessing = true;
          try {
            await this.faceMesh.send({ image: this.video });
          } catch (e) {
            console.warn("FaceMesh send error:", e);
          } finally {
            isProcessing = false;
          }
        }
        requestAnimationFrame(processFrame);
      };

      requestAnimationFrame(processFrame);
      return true;
    }

    onResults(results) {
      const nowMs = performance.now();
      const dt = (nowMs - this.lastFrameTime) / 1000;
      this.lastFrameTime = nowMs;
      if (dt > 0) {
        this.fpsSamples.push(1 / dt);
        if (this.fpsSamples.length > 10) this.fpsSamples.shift();
        this.fps = Math.round(this.fpsSamples.reduce((a, b) => a + b, 0) / this.fpsSamples.length);
      }

      const cw = this.canvas.width;
      const ch = this.canvas.height;
      this.ctx.save();
      this.ctx.clearRect(0, 0, cw, ch);

      // Mirror video feed
      this.ctx.translate(cw, 0);
      this.ctx.scale(-1, 1);
      this.ctx.drawImage(this.video, 0, 0, cw, ch);
      this.ctx.restore();

      const now = Date.now();
      const sessionDt = (now - this.lastTickTime) / 1000;
      this.lastTickTime = now;

      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        this.currentState = "No Face";
        this.currentConfidence = 0.0;
        this.dispatchUpdates({
          state: "No Face",
          confidence: 0,
          ear: 0,
          blink_rate: this.getBlinkRate(),
          gaze: "unknown",
          yaw: 0,
          pitch: 0,
          roll: 0,
          fps: this.fps,
          face_detected: false,
        });
        return;
      }

      const lms = results.multiFaceLandmarks[0];

      // ── 1. Calculate EAR & Blinks ──────────────────────────
      const leftEar  = computeEAR(lms, LEFT_EAR_IDS, cw, ch);
      const rightEar = computeEAR(lms, RIGHT_EAR_IDS, cw, ch);
      const ear = (leftEar + rightEar) / 2.0;

      this.earHistory.push(ear);
      if (this.earHistory.length > 90) this.earHistory.shift();

      if (ear < 0.21) {
        this.consecClosedFrames++;
      } else {
        if (this.consecClosedFrames >= 2) {
          this.blinkCount++;
          this.blinkTimes.push(now);
        }
        this.consecClosedFrames = 0;
      }
      this.blinkTimes = this.blinkTimes.filter(t => t > now - 60000);
      const blinkRate = this.blinkTimes.length;

      // ── 2. Gaze Estimation ─────────────────────────────────
      let gaze = "center";
      try {
        const leftIrisCenter = getIrisCenter(lms, LEFT_IRIS_IDS, cw, ch);
        const rightIrisCenter= getIrisCenter(lms, RIGHT_IRIS_IDS, cw, ch);

        const lOuter = { x: lms[263].x * cw, y: lms[263].y * ch };
        const lInner = { x: lms[362].x * cw, y: lms[362].y * ch };
        const rOuter = { x: lms[33].x * cw,  y: lms[33].y * ch };
        const rInner = { x: lms[133].x * cw, y: lms[133].y * ch };

        const lEyeW = Math.abs(lOuter.x - lInner.x) || 1;
        const rEyeW = Math.abs(rOuter.x - rInner.x) || 1;

        // User gaze direction
        const lRatioX = (leftIrisCenter.x - Math.min(lOuter.x, lInner.x)) / lEyeW;
        const rRatioX = (rightIrisCenter.x - Math.min(rOuter.x, rInner.x)) / rEyeW;
        const avgX = (lRatioX + rRatioX) / 2;

        if (avgX < 0.38) gaze = "left";
        else if (avgX > 0.62) gaze = "right";
        else gaze = "center";
      } catch (_) {
        gaze = "center";
      }

      // ── 3. Head Pose Estimation ────────────────────────────
      const nose   = { x: lms[NOSE_TIP].x * cw, y: lms[NOSE_TIP].y * ch };
      const chin   = { x: lms[CHIN].x * cw, y: lms[CHIN].y * ch };
      const lEyePt = { x: lms[LEFT_EYE].x * cw, y: lms[LEFT_EYE].y * ch };
      const rEyePt = { x: lms[RIGHT_EYE].x * cw, y: lms[RIGHT_EYE].y * ch };

      const eyeMidX = (lEyePt.x + rEyePt.x) / 2;
      const eyeSpan = Math.abs(rEyePt.x - lEyePt.x) || 1;
      const eyeMidY = (lEyePt.y + rEyePt.y) / 2;
      const faceHeight = Math.abs(chin.y - eyeMidY) || 1;

      // Inverted for mirrored preview
      const yaw = Math.round(-((nose.x - eyeMidX) / (eyeSpan * 0.45)) * 40);
      const pitch = Math.round(((nose.y - eyeMidY) / faceHeight - 0.42) * 80);
      const roll = Math.round(Math.atan2(rEyePt.y - lEyePt.y, rEyePt.x - lEyePt.x) * (180 / Math.PI));

      // ── 4. Classification Logic ────────────────────────────
      let state = "Focused";
      let confidence = 0.95;

      const isDrowsy = this.consecClosedFrames >= 18 || (blinkRate > 30 && ear < 0.23);
      const isDistracted = gaze !== "center" || Math.abs(yaw) > 22 || Math.abs(pitch) > 20 || Math.abs(roll) > 25;

      if (isDrowsy) {
        state = "Drowsy";
        confidence = 0.92;
      } else if (isDistracted) {
        state = "Distracted";
        confidence = 0.88;
      } else {
        state = "Focused";
        confidence = 0.96;
      }

      // Update durations & streaks
      if (this.durations[state] !== undefined) {
        this.durations[state] += sessionDt;
      }

      if (this.streakState !== state) {
        this.streakState = state;
        this.streakStart = now;
      }
      const curStreakSec = (now - this.streakStart) / 1000;
      if (state === "Focused" && curStreakSec > this.bestFocusStreak) {
        this.bestFocusStreak = Math.round(curStreakSec);
      }
      if (state === "Distracted" && curStreakSec > this.longestDistract) {
        this.longestDistract = Math.round(curStreakSec);
      }

      // Timeline point (every second)
      const elapsedSec = Math.floor((now - this.sessionStartTime) / 1000);
      if (this.timeline.length === 0 || this.timeline[this.timeline.length - 1].ts !== elapsedSec) {
        this.timeline.push({ ts: elapsedSec, state: state });
        if (this.timeline.length > 3600) this.timeline.shift();
      }

      // ── 5. Draw Canvas Overlays ────────────────────────────
      this.drawVisualizations(lms, cw, ch, state, confidence, ear, gaze, yaw, pitch, roll);

      // Dispatch UI updates
      this.currentState = state;
      this.currentConfidence = confidence;

      this.dispatchUpdates({
        state,
        confidence,
        ear: Number(ear.toFixed(3)),
        blink_rate: blinkRate,
        gaze,
        yaw,
        pitch,
        roll,
        fps: this.fps,
        face_detected: true,
      });
    }

    drawVisualizations(lms, cw, ch, state, conf, ear, gaze, yaw, pitch, roll) {
      const stateColors = {
        Focused:    "#22d67a",
        Distracted: "#f59e0b",
        Drowsy:     "#ef4444",
      };
      const color = stateColors[state] || "#22d67a";

      // Compute mirrored bounding box
      let minX = 1, maxX = 0, minY = 1, maxY = 0;
      for (const lm of lms) {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
      }

      const pad = 24;
      const bx = Math.max(0, (1 - maxX) * cw - pad);
      const by = Math.max(0, minY * ch - pad);
      const bw = Math.min(cw - bx, (maxX - minX) * cw + pad * 2);
      const bh = Math.min(ch - by, (maxY - minY) * ch + pad * 2);

      // Bounding box corners
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 2.5;
      const cornerLen = Math.min(24, bw * 0.2);

      // Top-Left
      this.ctx.beginPath();
      this.ctx.moveTo(bx, by + cornerLen);
      this.ctx.lineTo(bx, by);
      this.ctx.lineTo(bx + cornerLen, by);
      this.ctx.stroke();

      // Top-Right
      this.ctx.beginPath();
      this.ctx.moveTo(bx + bw - cornerLen, by);
      this.ctx.lineTo(bx + bw, by);
      this.ctx.lineTo(bx + bw, by + cornerLen);
      this.ctx.stroke();

      // Bottom-Left
      this.ctx.beginPath();
      this.ctx.moveTo(bx, by + bh - cornerLen);
      this.ctx.lineTo(bx, by + bh);
      this.ctx.lineTo(bx + cornerLen, by + bh);
      this.ctx.stroke();

      // Bottom-Right
      this.ctx.beginPath();
      this.ctx.moveTo(bx + bw - cornerLen, by + bh);
      this.ctx.lineTo(bx + bw, by + bh);
      this.ctx.lineTo(bx + bw, by + bh - cornerLen);
      this.ctx.stroke();

      // Draw iris highlights (mirrored)
      for (const idx of [...LEFT_IRIS_IDS, ...RIGHT_IRIS_IDS]) {
        const mx = (1 - lms[idx].x) * cw;
        const my = lms[idx].y * ch;
        this.ctx.beginPath();
        this.ctx.arc(mx, my, 2, 0, 2 * Math.PI);
        this.ctx.fillStyle = "#6ee7f7";
        this.ctx.fill();
      }

      // Top state badge
      const badgeText = `${state.toUpperCase()} • ${Math.round(conf * 100)}%`;
      this.ctx.font = "bold 13px Inter, sans-serif";
      const textWidth = this.ctx.measureText(badgeText).width;

      this.ctx.fillStyle = "rgba(10, 15, 29, 0.85)";
      this.ctx.fillRect(bx, Math.max(8, by - 26), textWidth + 18, 22);

      this.ctx.fillStyle = color;
      this.ctx.fillRect(bx, Math.max(8, by - 26), 4, 22);
      this.ctx.fillText(badgeText, bx + 10, Math.max(24, by - 10));
    }

    getBlinkRate() {
      const now = Date.now();
      return this.blinkTimes.filter(t => t > now - 60000).length;
    }

    dispatchUpdates(statusData) {
      if (this.callbacks.onStatus) this.callbacks.onStatus(statusData);

      // Session Summary
      const totalSec = this.durations.Focused + this.durations.Distracted + this.durations.Drowsy || 1;
      const fPct  = Math.round((this.durations.Focused / totalSec) * 100);
      const dPct  = Math.round((this.durations.Distracted / totalSec) * 100);
      const drPct = Math.round((this.durations.Drowsy / totalSec) * 100);

      // Attention Score formula
      const score = Math.max(0, Math.min(100, Math.round(fPct - drPct * 0.5)));

      const sessionData = {
        total_seconds: Math.round(totalSec),
        focused_pct: fPct,
        distracted_pct: dPct,
        drowsy_pct: drPct,
        best_focus_streak_sec: this.bestFocusStreak,
        longest_distract_sec: this.longestDistract,
        attention_score: score,
      };

      if (this.callbacks.onSession) this.callbacks.onSession(sessionData);
      if (this.callbacks.onTimeline) this.callbacks.onTimeline(this.timeline);
    }

    reset() {
      this.sessionStartTime = Date.now();
      this.lastTickTime = Date.now();
      this.durations = { Focused: 0, Distracted: 0, Drowsy: 0 };
      this.timeline = [];
      this.bestFocusStreak = 0;
      this.longestDistract = 0;
      this.streakStart = Date.now();
      this.blinkCount = 0;
      this.blinkTimes = [];
    }

    exportCSV() {
      const totalSec = this.durations.Focused + this.durations.Distracted + this.durations.Drowsy || 1;
      const fPct  = Math.round((this.durations.Focused / totalSec) * 100);
      const dPct  = Math.round((this.durations.Distracted / totalSec) * 100);
      const drPct = Math.round((this.durations.Drowsy / totalSec) * 100);
      const score = Math.max(0, Math.min(100, Math.round(fPct - drPct * 0.5)));

      let csv = "AttentionAI Session Report\n";
      csv += `Generated,${new Date().toISOString()}\n\n`;
      csv += "=== SESSION SUMMARY ===\n";
      csv += "Metric,Value\n";
      csv += `Total Duration (s),${Math.round(totalSec)}\n`;
      csv += `Focused (%),${fPct}\n`;
      csv += `Distracted (%),${dPct}\n`;
      csv += `Drowsy (%),${drPct}\n`;
      csv += `Best Focus Streak (s),${this.bestFocusStreak}\n`;
      csv += `Longest Distraction (s),${this.longestDistract}\n`;
      csv += `Attention Score (0-100),${score}\n\n`;
      csv += "=== ATTENTION TIMELINE ===\n";
      csv += "Elapsed (s),State\n";
      for (const pt of this.timeline) {
        csv += `${pt.ts},${pt.state}\n`;
      }
      return csv;
    }
  }

  window.WebTracker = WebTracker;
})(window);

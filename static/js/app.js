/**
 * AttentionAI — Dashboard JavaScript
 * Polls /api/status every 300ms and /api/session_summary every 2s.
 * Draws donut chart + timeline sparkline on <canvas>.
 */

"use strict";

// ── DOM refs ────────────────────────────────────────────────────
const liveDot        = document.getElementById("liveDot");
const liveLabel      = document.getElementById("liveLabel");
const fpsBadge       = document.getElementById("fpsBadge");
const stateCard      = document.getElementById("stateCard");
const stateIcon      = document.getElementById("stateIcon");
const stateLabel     = document.getElementById("stateLabel");
const confidenceBar  = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");
const earValue       = document.getElementById("earValue");
const earBar         = document.getElementById("earBar");
const blinkValue     = document.getElementById("blinkValue");
const gazeIcon       = document.getElementById("gazeIcon");
const gazeValue      = document.getElementById("gazeValue");
const yawValue       = document.getElementById("yawValue");
const pitchValue     = document.getElementById("pitchValue");
const rollValue      = document.getElementById("rollValue");
const gaugeYawFill   = document.getElementById("gaugeYawFill");
const gaugePitchFill = document.getElementById("gaugePitchFill");
const gaugeRollFill  = document.getElementById("gaugeRollFill");
const sessionTimer   = document.getElementById("sessionTimer");
const focusedPct     = document.getElementById("focusedPct");
const lgFocused      = document.getElementById("lgFocused");
const lgDistracted   = document.getElementById("lgDistracted");
const lgDrowsy       = document.getElementById("lgDrowsy");
const donutCanvas    = document.getElementById("donutChart");
const timelineCanvas = document.getElementById("timelineChart");
const videoCanvas    = document.getElementById("videoCanvas");   // ← canvas feed
const btnReset       = document.getElementById("btnReset");
const toast          = document.getElementById("toast");
const noFaceOverlay  = document.getElementById("noFaceOverlay");
const videoWrapper   = document.getElementById("videoWrapper");

const donutCtx    = donutCanvas.getContext("2d");
const timelineCtx = timelineCanvas.getContext("2d");
const videoCtx    = videoCanvas.getContext("2d");

// ── State ───────────────────────────────────────────────────────
const CONFIG = {
  POLL_STATUS_MS:  300,
  POLL_SESSION_MS: 2000,
  POLL_TIMELINE_MS: 2500,
};

const STATE_META = {
  "Focused":    { icon: "🎯", cls: "focused",    color: "#22d67a" },
  "Distracted": { icon: "👀", cls: "distracted", color: "#f59e0b" },
  "Drowsy":     { icon: "😴", cls: "drowsy",     color: "#ef4444" },
  "No Face":    { icon: "❓", cls: "",            color: "#8899bb" },
};

const GAZE_ICONS = {
  center:  "👁️", left: "👈", right: "👉",
  up: "⬆️", down: "⬇️", unknown: "❓",
};

let sessionStartTime = Date.now();
let timerInterval    = null;
let isConnected      = false;
let donutData        = { focused: 0, distracted: 0, drowsy: 0 };
let timelineData     = [];

// ── Helpers ─────────────────────────────────────────────────────
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function angleToBarPercent(deg, maxDeg = 45) {
  // Maps -maxDeg..+maxDeg → 0..100 centered at 50
  const ratio = clamp(deg / maxDeg, -1, 1);
  return 50 + ratio * 50;
}

function showToast(msg, duration = 2500) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), duration);
}

// ── Session Timer ────────────────────────────────────────────────
function startTimer() {
  sessionStartTime = Date.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const hh = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const mm = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const ss = String(elapsed % 60).padStart(2, "0");
    sessionTimer.textContent = `${hh}:${mm}:${ss}`;
  }, 1000);
}

// ── Update UI from /api/status ───────────────────────────────────
function applyStatus(data) {
  const meta = STATE_META[data.state] || STATE_META["No Face"];

  // Connection indicator
  if (!isConnected) {
    isConnected = true;
    liveDot.classList.add("active");
    liveLabel.textContent = "Live";
  }

  // FPS
  fpsBadge.textContent = `${data.fps} FPS`;

  // No-face overlay
  if (!data.face_detected) {
    noFaceOverlay.classList.remove("hidden");
  } else {
    noFaceOverlay.classList.add("hidden");
  }

  // State card
  stateCard.className = `state-card ${meta.cls}`;
  stateIcon.textContent = meta.icon;
  stateLabel.textContent = data.state;

  const pct = Math.round((data.confidence || 0) * 100);
  confidenceBar.style.width = `${pct}%`;
  confidenceText.textContent = `Confidence: ${pct}%`;

  // EAR
  earValue.textContent = data.ear.toFixed(3);
  const earPct = clamp((data.ear / 0.35) * 100, 0, 100);
  earBar.style.width = `${earPct}%`;

  // Blink rate
  blinkValue.textContent = data.blink_rate;

  // Gaze
  const gDir = data.gaze || "unknown";
  gazeIcon.textContent = GAZE_ICONS[gDir] || "👁";
  gazeValue.textContent = gDir.charAt(0).toUpperCase() + gDir.slice(1);

  // Head pose
  yawValue.textContent   = `${data.yaw}°`;
  pitchValue.textContent = `${data.pitch}°`;
  rollValue.textContent  = `${data.roll}°`;

  const yPct   = angleToBarPercent(data.yaw);
  const piPct  = angleToBarPercent(data.pitch);
  const roPct  = angleToBarPercent(data.roll);

  setGauge(gaugeYawFill,   yPct);
  setGauge(gaugePitchFill, piPct);
  setGauge(gaugeRollFill,  roPct);
}

function setGauge(el, pct) {
  // Center at 50% → display as bar extending left or right from center
  const half = Math.abs(pct - 50);
  if (pct >= 50) {
    el.style.left  = "50%";
    el.style.width = `${half}%`;
  } else {
    el.style.left  = `${pct}%`;
    el.style.width = `${half}%`;
  }
}

// ── Donut Chart ──────────────────────────────────────────────────
function drawDonut(f, d, dr) {
  const cx = donutCanvas.width  / 2;
  const cy = donutCanvas.height / 2;
  const R  = 68;
  const r  = 44;
  const total = f + d + dr || 1;

  donutCtx.clearRect(0, 0, donutCanvas.width, donutCanvas.height);

  const slices = [
    { value: f,  color: "#22d67a" },
    { value: d,  color: "#f59e0b" },
    { value: dr, color: "#ef4444" },
  ];

  let startAngle = -Math.PI / 2;
  for (const { value, color } of slices) {
    const sweep = (value / total) * 2 * Math.PI;
    donutCtx.beginPath();
    donutCtx.moveTo(cx, cy);
    donutCtx.arc(cx, cy, R, startAngle, startAngle + sweep);
    donutCtx.closePath();
    donutCtx.fillStyle = color;
    donutCtx.fill();
    startAngle += sweep;
  }

  // Inner cutout
  donutCtx.beginPath();
  donutCtx.arc(cx, cy, r, 0, 2 * Math.PI);
  donutCtx.fillStyle = "#111827";
  donutCtx.fill();

  // Thin border ring
  donutCtx.beginPath();
  donutCtx.arc(cx, cy, R, 0, 2 * Math.PI);
  donutCtx.strokeStyle = "rgba(255,255,255,0.04)";
  donutCtx.lineWidth = 2;
  donutCtx.stroke();
}

function applySession(data) {
  const f  = data.focused_pct    || 0;
  const d  = data.distracted_pct || 0;
  const dr = data.drowsy_pct     || 0;

  donutData = { focused: f, distracted: d, drowsy: dr };
  drawDonut(f, d, dr);

  focusedPct.textContent   = `${f}%`;
  lgFocused.textContent    = `${f}%`;
  lgDistracted.textContent = `${d}%`;
  lgDrowsy.textContent     = `${dr}%`;

  // ── Performance Highlights ─────────────────────────────────────
  function fmtSec(s) {
    if (!s || s < 1) return "—";
    const m   = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  }

  const streak  = data.best_focus_streak_sec || 0;
  const distLen = data.longest_distract_sec  || 0;
  const score   = (data.attention_score !== undefined) ? data.attention_score : null;

  document.getElementById("hlFocusStreakVal").textContent = fmtSec(streak);
  document.getElementById("hlDistractVal").textContent    = fmtSec(distLen);

  if (score !== null) {
    document.getElementById("hlScoreVal").textContent = `${score} / 100`;
    document.getElementById("hlScoreBar").style.width = `${score}%`;
  }
}

// ── Timeline Sparkline ────────────────────────────────────────────
const STATE_Y = { "Focused": 0.1, "Distracted": 0.5, "Drowsy": 0.9, "No Face": 0.5 };
const STATE_C = { "Focused": "#22d67a", "Distracted": "#f59e0b", "Drowsy": "#ef4444", "No Face": "#8899bb" };

function drawTimeline(records) {
  const cw = timelineCanvas.offsetWidth  || timelineCanvas.width;
  const ch = timelineCanvas.height;
  timelineCanvas.width = cw;

  timelineCtx.clearRect(0, 0, cw, ch);

  if (!records || records.length < 2) {
    timelineCtx.fillStyle = "rgba(255,255,255,0.06)";
    const msg = "Collecting data…";
    timelineCtx.font = "12px Inter, sans-serif";
    timelineCtx.textAlign = "center";
    timelineCtx.fillText(msg, cw / 2, ch / 2 + 4);
    return;
  }

  const maxTs = records[records.length - 1].ts;
  const minTs = records[0].ts;
  const tsRange = maxTs - minTs || 1;

  // Draw grid lines
  timelineCtx.strokeStyle = "rgba(255,255,255,0.04)";
  timelineCtx.lineWidth = 1;
  [0.1, 0.5, 0.9].forEach(yf => {
    const y = yf * ch;
    timelineCtx.beginPath();
    timelineCtx.moveTo(0, y); timelineCtx.lineTo(cw, y);
    timelineCtx.stroke();
  });

  // Draw each segment with its color
  for (let i = 1; i < records.length; i++) {
    const prev = records[i - 1];
    const curr = records[i];
    const x0 = ((prev.ts - minTs) / tsRange) * cw;
    const x1 = ((curr.ts - minTs) / tsRange) * cw;
    const y0 = STATE_Y[prev.state] * ch;
    const y1 = STATE_Y[curr.state] * ch;

    timelineCtx.beginPath();
    timelineCtx.moveTo(x0, y0);
    timelineCtx.lineTo(x1, y1);
    timelineCtx.strokeStyle = STATE_C[curr.state] || "#8899bb";
    timelineCtx.lineWidth = 2.5;
    timelineCtx.lineJoin = "round";
    timelineCtx.stroke();
  }

  // Dot at last point
  const last = records[records.length - 1];
  const lx = cw - 4;
  const ly = STATE_Y[last.state] * ch;
  timelineCtx.beginPath();
  timelineCtx.arc(lx, ly, 4, 0, 2 * Math.PI);
  timelineCtx.fillStyle = STATE_C[last.state] || "#8899bb";
  timelineCtx.fill();

  // Labels
  timelineCtx.font = "9px JetBrains Mono, monospace";
  timelineCtx.fillStyle = "rgba(255,255,255,0.3)";
  timelineCtx.textAlign = "left";
  timelineCtx.fillText("Focused   ─", 6, 0.1 * ch + 3);
  timelineCtx.fillText("Distracted─", 6, 0.5 * ch + 3);
  timelineCtx.fillText("Drowsy    ─", 6, 0.9 * ch + 3);
}

// ── Mode Handling (Server vs Client-Side WebTracker) ────────────────
let clientTracker = null;
let isClientMode  = false;

async function checkModeAndStart() {
  const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

  if (isLocal) {
    try {
      const ctrl = new AbortController();
      const timeout = setTimeout(() => ctrl.abort(), 1200);
      const r = await fetch("/api/status", { signal: ctrl.signal });
      clearTimeout(timeout);
      const contentType = r.headers.get("content-type") || "";
      if (r.ok && contentType.includes("application/json")) {
        const data = await r.json();
        if (data && (data.state !== undefined || data.face_detected !== undefined)) {
          console.log("[AttentionAI] Server mode detected. Connecting to Python backend...");
          setInterval(pollStatus,   CONFIG.POLL_STATUS_MS);
          setInterval(pollSession,  CONFIG.POLL_SESSION_MS);
          setInterval(pollTimeline, CONFIG.POLL_TIMELINE_MS);
          pollStatus();
          pollSession();
          pollTimeline();
          framePollLoop();
          return;
        }
      }
    } catch (_) {}
  }

  // Running on Vercel or cloud static host -> Browser-Native Client Mode
  startClientMode();
}

async function startClientMode() {
  if (isClientMode && clientTracker && clientTracker.isRunning) return;
  isClientMode = true;
  console.log("[AttentionAI] Activating Browser-Native Client-Side WebTracker...");

  const startBtn = document.getElementById("btnStartCam");
  if (startBtn) startBtn.style.display = "none";

  liveLabel.textContent = "Requesting Cam…";

  clientTracker = new window.WebTracker(videoCanvas);
  const started = await clientTracker.start({
    onStatus: (data) => {
      applyStatus(data);
      if (typeof window.handleAlerts === "function") {
        window.handleAlerts(data.state, 100);
      }
    },
    onSession: (summary) => {
      applySession(summary);
    },
    onTimeline: (timeline) => {
      timelineData = timeline;
      drawTimeline(timelineData);
    },
    onError: (err) => {
      console.warn("Client tracker error:", err);
      liveLabel.textContent = "Camera Blocked";
      liveDot.classList.remove("active");
      if (startBtn) {
        startBtn.style.display = "inline-flex";
        startBtn.textContent = "📹 Click to Enable Camera";
      }
      showToast("❌ Camera permission required. Please allow camera access.", 5000);
    }
  });

  if (started) {
    showToast("🎥 Camera active — AI tracking live!", 3000);
    liveDot.classList.add("active");
    liveLabel.textContent = "Live (Web AI)";
    if (startBtn) startBtn.style.display = "none";
  }
}

// Bind Start Camera button
const btnStartCam = document.getElementById("btnStartCam");
if (btnStartCam) {
  btnStartCam.addEventListener("click", () => {
    startClientMode();
  });
}

// ── Polling ──────────────────────────────────────────────────────
async function pollStatus() {
  if (isClientMode) return;
  try {
    const r = await fetch("/api/status");
    if (!r.ok) throw new Error();
    const data = await r.json();
    applyStatus(data);
    if (typeof window.handleAlerts === "function") {
      window.handleAlerts(data.state, CONFIG.POLL_STATUS_MS);
    }
  } catch (e) {
    if (isConnected) {
      liveLabel.textContent = "Reconnecting…";
      liveDot.classList.remove("active");
      isConnected = false;
    }
  }
}

async function pollSession() {
  if (isClientMode) return;
  try {
    const r = await fetch("/api/session_summary");
    if (r.ok) applySession(await r.json());
  } catch (_) {}
}

async function pollTimeline() {
  if (isClientMode) return;
  try {
    const r = await fetch("/api/timeline");
    if (r.ok) {
      timelineData = await r.json();
      drawTimeline(timelineData);
    }
  } catch (_) {}
}

// ── Reset Session ─────────────────────────────────────────────────
btnReset.addEventListener("click", async () => {
  if (isClientMode && clientTracker) {
    clientTracker.reset();
    startTimer();
    timelineData = [];
    drawDonut(0, 0, 0);
    drawTimeline([]);
    showToast("✅ Session reset successfully");
    return;
  }

  try {
    await fetch("/api/reset_session", { method: "POST" });
    startTimer();
    timelineData = [];
    drawDonut(0, 0, 0);
    showToast("✅ Session reset successfully");
  } catch (e) {
    showToast("❌ Failed to reset session");
  }
});

// ── Export CSV ─────────────────────────────────────────────────
const btnExportCSV = document.getElementById("btnExportCSV");
if (btnExportCSV) {
  btnExportCSV.addEventListener("click", async () => {
    btnExportCSV.classList.add("downloading");
    btnExportCSV.textContent = "⏳ Downloading...";
    try {
      if (isClientMode && clientTracker) {
        const csvContent = clientTracker.exportCSV();
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `attentionai_session_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("📅 Session exported successfully");
        return;
      }

      const resp = await fetch("/api/export_csv");
      if (!resp.ok) throw new Error("export failed");
      const blob     = await resp.blob();
      const url      = URL.createObjectURL(blob);
      const filename = (resp.headers.get("Content-Disposition") || "attentionai.csv")
        .split("filename=")[1] || "attentionai_session.csv";
      const a = document.createElement("a");
      a.href     = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      showToast("📅 Session exported successfully");
    } catch (e) {
      showToast("❌ Export failed");
    } finally {
      btnExportCSV.classList.remove("downloading");
      btnExportCSV.textContent = "📅 Export CSV";
    }
  });
}

// ── Canvas Frame Polling ─────────────────────────────────────────────
// Fetches /api/frame as JPEG blob and draws to canvas — works in ALL browsers.
let framePollActive = true;

async function framePollLoop() {
  while (framePollActive && !isClientMode) {
    try {
      const r = await fetch("/api/frame", { cache: "no-store" });
      if (!r.ok) throw new Error("frame fetch failed");
      const blob = await r.blob();
      const bmp  = await createImageBitmap(blob);
      // Resize canvas to match if needed
      if (videoCanvas.width !== bmp.width || videoCanvas.height !== bmp.height) {
        videoCanvas.width  = bmp.width;
        videoCanvas.height = bmp.height;
      }
      videoCtx.drawImage(bmp, 0, 0);
      bmp.close();
    } catch (e) {
      if (isClientMode) break;
      // Draw a subtle "reconnecting" overlay on the canvas
      videoCtx.fillStyle = "#080c14";
      videoCtx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
      videoCtx.fillStyle = "rgba(255,255,255,0.15)";
      videoCtx.font = "14px Inter, sans-serif";
      videoCtx.textAlign = "center";
      videoCtx.fillText("Reconnecting…", videoCanvas.width / 2, videoCanvas.height / 2);
    }
    // Throttle to ~20 fps (50ms) to avoid hammering the server
    await new Promise(r => setTimeout(r, 50));
  }
}

// ── Init ──────────────────────────────────────────────────────────
startTimer();
drawDonut(0, 0, 0);
drawTimeline([]);

checkModeAndStart();

// Resize timeline on window resize
window.addEventListener("resize", () => drawTimeline(timelineData));

// ── Generate Background Particles ────────────────────────────────
const particlesEl = document.getElementById("bgParticles");
for (let i = 0; i < 18; i++) {
  const dot = document.createElement("div");
  const size = Math.random() * 3 + 1;
  Object.assign(dot.style, {
    position: "absolute",
    width:  `${size}px`,
    height: `${size}px`,
    borderRadius: "50%",
    background: `hsla(${190 + Math.random() * 80}, 80%, 70%, ${0.1 + Math.random() * 0.2})`,
    top:  `${Math.random() * 100}%`,
    left: `${Math.random() * 100}%`,
    animation: `drift${Math.random() > 0.5 ? "1" : "2"} ${10 + Math.random() * 15}s ease-in-out infinite alternate`,
    animationDelay: `${Math.random() * 5}s`,
  });
  particlesEl.appendChild(dot);
}

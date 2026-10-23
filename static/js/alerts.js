/**
 * AttentionAI — Alert Engine (no mute, always on)
 *
 * Behaviour:
 *   • Every 3 s of distracted/drowsy → warning beep (up to 3)
 *   • After 3rd warning              → CONTINUOUS alarm until Focused
 *   • Returns to Focused             → everything stops, counter resets
 *
 * SETUP: User must click "🔔 Enable Audio" once to unlock the browser
 *        AudioContext. After that all beeps play automatically.
 */

"use strict";

/* ── DOM ─────────────────────────────────────────────────────── */
const _banner  = document.getElementById("warningBanner");
const _bText   = document.getElementById("warningText");
const _bIcon   = document.getElementById("warningIcon");
const _btnDis  = document.getElementById("btnDismiss");

/* ── Config ──────────────────────────────────────────────────── */
const CFG = {
  WARN_INTERVAL_SEC : 3,    // seconds of distraction per warning step
  MAX_WARNINGS      : 3,    // total warnings before continuous alarm
  BEEP_FREQ         : 880,  // Hz
  BEEP_DURATION     : 0.35, // seconds
  BEEP_GAP_MS       : 900,  // ms between continuous beeps
  VOLUME            : 1.0,  // always full volume
};

/* ── AudioContext ────────────────────────────────────────────── */
let _ctx   = null;
let _ready = false;

function _initAudio() {
  if (!_ctx) _ctx = new (window.AudioContext || window.webkitAudioContext)();
}
function _ensureRunning() {
  _initAudio();
  if (_ctx.state !== "running") _ctx.resume();
}

/** Play one sharp beep — always at full volume. */
function _beep() {
  _ensureRunning();
  try {
    const osc  = _ctx.createOscillator();
    const gain = _ctx.createGain();
    osc.connect(gain);
    gain.connect(_ctx.destination);

    osc.type = "square";
    osc.frequency.value = CFG.BEEP_FREQ;

    const t = _ctx.currentTime;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(CFG.VOLUME, t + 0.01);
    gain.gain.setValueAtTime(CFG.VOLUME, t + CFG.BEEP_DURATION - 0.04);
    gain.gain.linearRampToValueAtTime(0, t + CFG.BEEP_DURATION);

    osc.start(t);
    osc.stop(t + CFG.BEEP_DURATION + 0.01);
  } catch (e) {
    console.warn("[Alert] beep error:", e);
  }
}

// Keep AudioContext alive (prevents Chrome/Safari auto-suspend)
setInterval(() => {
  if (_ctx && _ctx.state === "suspended") _ctx.resume();
}, 20_000);

/* ── State ───────────────────────────────────────────────────── */
let _dismissed  = false;
let _accumSec   = 0;
let _warnings   = 0;
let _alarming   = false;
let _alarmTimer = null;

function _startAlarm() {
  if (_alarming) return;
  _alarming = true;
  _beep();
  _alarmTimer = setInterval(_beep, CFG.BEEP_GAP_MS);
}

function _stopAlarm() {
  clearInterval(_alarmTimer);
  _alarmTimer = null;
  _alarming   = false;
}

function _reset() {
  _stopAlarm();
  _accumSec  = 0;
  _warnings  = 0;
  _dismissed = false;
  _hideBanner();
}

/* ── Banner ──────────────────────────────────────────────────── */
function _showBanner(text, icon, type, urgent) {
  if (_dismissed && !urgent) return;
  _banner.className =
    "warning-banner warn-" + type + (urgent ? " warn-urgent" : "");
  _bIcon.textContent = icon;
  _bText.textContent = text;
  _banner.classList.remove("hidden");
}
function _hideBanner() { _banner.classList.add("hidden"); }

/* ── Main handler (called every ~300 ms from pollStatus) ─────── */
window.handleAlerts = function handleAlerts(state, pollMs) {
  if (state === "Focused" || state === "No Face") {
    _reset();
    return;
  }

  const isDistracted = state === "Distracted";
  const isDrowsy     = state === "Drowsy";
  if (!isDistracted && !isDrowsy) return;

  _accumSec += (pollMs || 300) / 1000;

  // Continuous alarm is running — keep banner fresh, do nothing else
  if (_alarming) {
    _showBanner(
      "🚨 ALARM — Look at the camera to stop the alarm!",
      "🚨", isDistracted ? "distracted" : "drowsy", true
    );
    return;
  }

  const dueWarnings = Math.floor(_accumSec / CFG.WARN_INTERVAL_SEC);

  // Issue next warning beep
  if (dueWarnings > _warnings && _warnings < CFG.MAX_WARNINGS) {
    _warnings++;
    _dismissed = false;
    const type = isDistracted ? "distracted" : "drowsy";
    const icon = isDistracted ? "⚠️" : "😴";
    _beep();
    _showBanner(
      `Warning ${_warnings}/${CFG.MAX_WARNINGS} — ` +
      (isDistracted ? "You look distracted! Please focus." : "You look drowsy! Stay alert."),
      icon, type, false
    );
  }

  // All 3 warnings done → continuous alarm
  if (_warnings >= CFG.MAX_WARNINGS && dueWarnings > CFG.MAX_WARNINGS) {
    _dismissed = false;
    _showBanner(
      "🚨 ALARM — Look at the camera to stop the alarm!",
      "🚨", isDistracted ? "distracted" : "drowsy", true
    );
    _startAlarm();
  }
};

/* ── Enable Audio button (one-time unlock) ───────────────────── */
const _btnEnable = document.getElementById("btnEnableAudio");
if (_btnEnable) {
  _btnEnable.addEventListener("click", () => {
    _initAudio();
    _ctx.resume().then(() => {
      _ready = true;
      _beep();  // confirmation beep so user knows it worked
      _btnEnable.textContent = "🔔 Audio ON";
      _btnEnable.classList.add("enabled");
      _btnEnable.disabled = true;
    });
  });
}

// Live dot → test beep
document.getElementById("liveDot").addEventListener("click", () => {
  _ensureRunning();
  _beep();
});

// Dismiss — hides banner but alarm keeps beeping until Focused
_btnDis.addEventListener("click", () => {
  _hideBanner();
  _dismissed = true;
  // alarm intentionally NOT stopped here
});

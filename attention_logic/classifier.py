"""
Attention Classification Logic.
Combines EAR, blink rate, gaze direction, and head pose into a single attention state.
States: "Focused", "Distracted", "Drowsy"
"""

import time
from collections import deque


# ── Drowsiness thresholds ──────────────────────────────────────────
# Only flag drowsy on SUSTAINED eye closure — avg EAR alone is too noisy
# because a few blinks pull the rolling average below any reasonable threshold.
DROWSY_BLINK_RATE_HIGH = 30     # blinks/min above this = very high → drowsy
DROWSY_CONSEC_FRAMES   = 20     # sustained eye closure frames → drowsy

# ── Distraction thresholds ─────────────────────────────────────────
DISTRACTED_AWAY_FRAMES = 15     # frames looking away → distracted
DISTRACTED_GAZE_RATIO  = 0.5    # >50% of recent frames gaze off-center

# ── Confidence smoothing ───────────────────────────────────────────
HISTORY_LEN = 30                # smooth over ~1s at 30fps


class AttentionClassifier:
    def __init__(self):
        self.state_history = deque(maxlen=HISTORY_LEN)
        self.current_state = "Focused"
        self.confidence    = 1.0

        # Session timing
        self.session_start   = time.time()
        self.state_durations = {"Focused": 0.0, "Distracted": 0.0, "Drowsy": 0.0}
        self.last_tick_time  = time.time()
        self.timeline_buffer = deque(maxlen=3600)

        # Streak tracking
        self._streak_state   = None   # which state the current streak is for
        self._streak_start   = time.time()
        self.best_focus_streak_sec   = 0.0   # 🏆 best continuous Focused run
        self.longest_distract_sec    = 0.0   # 📉 longest continuous Distracted run

    # ── Core Classification ─────────────────────────────────────────

    def classify(self, ear_stats, gaze_stats, pose_stats):
        """
        Returns (state: str, confidence: float, signals: dict)
        """
        signals = {}

        # ── Check Drowsy first (highest priority) ─────────────────
        is_drowsy = (
            ear_stats.get("is_drowsy_by_closure", False)
            or ear_stats.get("blink_rate_per_min", 0) > DROWSY_BLINK_RATE_HIGH
        )
        signals["drowsy_closure"]    = ear_stats.get("is_drowsy_by_closure", False)
        signals["drowsy_blink_high"] = ear_stats.get("blink_rate_per_min", 0) > DROWSY_BLINK_RATE_HIGH

        # ── Check Distracted ──────────────────────────────────────
        gaze_away    = gaze_stats.get("looking_away", False)
        head_away    = pose_stats.get("looking_away", False)
        is_distracted = gaze_away or head_away
        signals["gaze_away"] = gaze_away
        signals["head_away"] = head_away

        # ── Rule-based state ──────────────────────────────────────
        if is_drowsy:
            raw_state = "Drowsy"
        elif is_distracted:
            raw_state = "Distracted"
        else:
            raw_state = "Focused"

        # Push to history
        self.state_history.append(raw_state)

        # Majority vote for smoothing
        from collections import Counter
        counts = Counter(self.state_history)
        self.current_state = counts.most_common(1)[0][0]
        self.confidence    = counts[self.current_state] / len(self.state_history)

        # Update session durations & streaks
        now  = time.time()
        dt   = now - self.last_tick_time
        self.state_durations[self.current_state] += dt
        self.last_tick_time = now

        # Streak tracking
        if self.current_state != self._streak_state:
            # State changed — close out the previous streak
            if self._streak_state is not None:
                elapsed = now - self._streak_start
                if self._streak_state == "Focused":
                    self.best_focus_streak_sec = max(
                        self.best_focus_streak_sec, elapsed
                    )
                elif self._streak_state == "Distracted":
                    self.longest_distract_sec = max(
                        self.longest_distract_sec, elapsed
                    )
            self._streak_state = self.current_state
            self._streak_start = now

        return self.current_state, round(self.confidence, 2), signals

    # ── Session Analytics ───────────────────────────────────────────

    def session_summary(self):
        total = sum(self.state_durations.values())
        focused_s     = self.state_durations["Focused"]
        distracted_s  = self.state_durations["Distracted"]
        drowsy_s      = self.state_durations["Drowsy"]

        if total == 0:
            return {
                "focused_pct": 0, "distracted_pct": 0, "drowsy_pct": 0,
                "total_seconds": 0,
                "focused_seconds": 0, "distracted_seconds": 0, "drowsy_seconds": 0,
                "best_focus_streak_sec": 0,
                "longest_distract_sec": 0,
                "attention_score": 0,
            }

        f_pct  = float(100 * focused_s    / total)
        d_pct  = float(100 * distracted_s / total)
        dr_pct = float(100 * drowsy_s     / total)

        # Attention score: focused base, penalised by distraction & drowsiness
        score = int(min(100, max(0, f_pct - 0.4 * d_pct - 0.7 * dr_pct)))

        # Current streak might still be running — factor it in
        live_focus_streak  = self.best_focus_streak_sec
        live_dist_streak   = self.longest_distract_sec
        if self._streak_state == "Focused":
            live_focus_streak = max(live_focus_streak,
                                    time.time() - self._streak_start)
        elif self._streak_state == "Distracted":
            live_dist_streak  = max(live_dist_streak,
                                    time.time() - self._streak_start)

        return {
            "focused_pct":        round(f_pct, 1),
            "distracted_pct":     round(d_pct, 1),
            "drowsy_pct":         round(dr_pct, 1),
            "total_seconds":      round(float(total), 1),
            "focused_seconds":    round(float(focused_s),    1),
            "distracted_seconds": round(float(distracted_s), 1),
            "drowsy_seconds":     round(float(drowsy_s),     1),
            "best_focus_streak_sec":  round(float(live_focus_streak), 1),
            "longest_distract_sec":   round(float(live_dist_streak),  1),
            "attention_score":        score,
        }

    def record_timeline_point(self):
        """Call once per second to record state for the timeline chart."""
        self.timeline_buffer.append({
            "ts":    round(float(time.time() - self.session_start), 1),
            "state": self.current_state,
        })

    def get_timeline(self):
        return list(self.timeline_buffer)

    def reset_session(self):
        self.state_durations = {"Focused": 0.0, "Distracted": 0.0, "Drowsy": 0.0}
        self.session_start   = time.time()
        self.last_tick_time  = time.time()
        self.timeline_buffer.clear()
        self.state_history.clear()
        self.best_focus_streak_sec = 0.0
        self.longest_distract_sec  = 0.0
        self._streak_state = None
        self._streak_start = time.time()

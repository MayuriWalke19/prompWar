"""
Session data store — saves attention state per second to a JSON file.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


class SessionLogger:
    def __init__(self):
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath    = DATA_DIR / f"session_{self.session_id}.json"
        self.records     = []
        self.last_record = 0

    def record(self, state, confidence, ear, blink_rate, gaze, yaw, pitch, roll):
        """Call every frame; internally throttles to ~1 record/second."""
        now = time.time()
        if now - self.last_record < 1.0:
            return
        self.last_record = now
        self.records.append({
            "timestamp":   round(now, 2),
            "state":       state,
            "confidence":  confidence,
            "ear":         round(ear, 3),
            "blink_rate":  blink_rate,
            "gaze":        gaze,
            "yaw":         round(yaw, 1),
            "pitch":       round(pitch, 1),
            "roll":        round(roll, 1),
        })

    def save(self):
        with open(self.filepath, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "records":    self.records,
            }, f, indent=2)
        return str(self.filepath)

    def list_sessions(self):
        sessions = []
        for p in sorted(DATA_DIR.glob("session_*.json"), reverse=True)[:20]:
            try:
                with open(p) as f:
                    d = json.load(f)
                sessions.append({
                    "id":      d.get("session_id", p.stem),
                    "file":    p.name,
                    "records": len(d.get("records", [])),
                })
            except Exception:
                pass
        return sessions

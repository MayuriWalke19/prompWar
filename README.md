# 🧠 AttentionAI — Real-Time Attention Tracker

A fully local, real-time attention tracking system that uses your webcam to detect and classify student focus states using computer vision and facial analysis.

---

## 🎯 Features

| Feature | Description |
|---|---|
| Face Detection | MediaPipe Face Mesh — 468 landmarks |
| Eye Aspect Ratio (EAR) | Blink detection & drowsiness scoring |
| Gaze Estimation | Iris tracking → left/right/center/up/down |
| Head Pose | Yaw, Pitch, Roll via solvePnP |
| Attention State | Focused / Distracted / Drowsy |
| Real-Time Dashboard | Live MJPEG stream + metrics |
| Session Analytics | Pie chart + timeline + % focus time |
| Session Logging | JSON records saved to `data/` folder |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
python main.py
```

### 3. Open your browser

```
http://localhost:5050
```

> Make sure your webcam is connected and accessible.

---

## 📁 Project Structure

```
hackitout/
├── main.py                      # Flask server + camera processing loop
├── requirements.txt
│
├── face_detection/
│   └── detector.py              # MediaPipe Face Mesh wrapper
│
├── eye_tracking/
│   ├── ear.py                   # EAR calculation + blink detection
│   └── gaze.py                  # Iris-based gaze direction
│
├── head_pose/
│   └── estimator.py             # solvePnP head pose (Yaw/Pitch/Roll)
│
├── attention_logic/
│   └── classifier.py            # Rule-based attention classifier + session tracking
│
├── utils/
│   ├── drawing.py               # OpenCV overlay & visualization helpers
│   └── session_logger.py        # JSON session data writer
│
├── data/                        # Session JSON files saved here
├── static/
│   ├── css/style.css            # Premium dark dashboard styles
│   └── js/app.js                # Dashboard JS (charts, polling)
└── templates/
    └── index.html               # Main dashboard HTML
```

---

## 🧠 Attention Classification Logic

| Signal | Drowsy Threshold | Distracted Threshold |
|---|---|---|
| EAR (Eye Aspect Ratio) | < 0.20 | — |
| Consecutive eye closure | ≥ 20 frames | — |
| Blinks per minute | > 25 | — |
| Gaze direction | — | Not center |
| Head yaw | — | > ±20° |
| Head pitch | — | > ±20° |

**Priority**: Drowsy > Distracted > Focused

---

## 🎨 Color Coding

- 🟢 **Green** — Focused
- 🟡 **Amber** — Distracted  
- 🔴 **Red** — Drowsy

---

## ⚙️ Configuration

Thresholds can be tuned in each module:
- `eye_tracking/ear.py` — `EAR_THRESHOLD`, `DROWSY_BLINK_RATE_HIGH`
- `eye_tracking/gaze.py` — `GAZE_HORIZONTAL_THRESHOLD`
- `head_pose/estimator.py` — `YAW_THRESHOLD`, `PITCH_THRESHOLD`
- `attention_logic/classifier.py` — all combined thresholds

---

## 🔒 Privacy

All processing happens **100% locally**. No data is sent to any server or cloud. Session JSON files stay in the `data/` directory on your machine.

---

## 📋 Requirements

- Python 3.8+
- Webcam (any standard USB or built-in)
- macOS / Linux / Windows
- ~500MB RAM during runtime

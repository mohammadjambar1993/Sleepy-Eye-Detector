# Drowsiness Detector (EAR) 👀🔔
<!-- PROJECT TAGLINE -->
Detect microsleeps in real time using Eye Aspect Ratio (EAR) with MediaPipe FaceMesh + OpenCV. Triggers an audible alarm when eyes stay closed beyond a threshold.

<!-- BADGES -->
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange.svg)](#)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-green.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing)
<!-- Optional CI badge:
[![CI](https://github.com/<USER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<USER>/<REPO>/actions) -->

---


## ✨ Features
- **Lightweight & real-time**: EAR via MediaPipe FaceMesh (no heavy model).
- **Webcam only**, no Gradio/extra UI.
- **Multi-backend audio** fallback: `simpleaudio → sounddevice (PortAudio) → aplay (ALSA)`.
- **ENV-based config** (camera index, thresholds, FPS).
- **Visual-only mode** for servers/WSL/headless.

---

## 🧩 How It Works
- Webcam → OpenCV frame → MediaPipe FaceMesh → EAR (per eye) → average EAR
- └──────────────────── if EAR < threshold for N seconds ──► Alarm


---

## 🚀 Quick Start
```bash
# 1) Create & activate a virtual env (optional)
python -m venv .venv && source .venv/bin/activate

# 2) Install Python deps
pip install -r requirements.txt

# 3) (Linux recommended extras)
# sudo apt install libportaudio2 alsa-utils

# 4) Place alarm.wav in repo root (this repo already includes it)
# 5) Run
python app.py
```
---
## 🚀 Configuration (ENV)

| Variable             |  Default  | Description                                        |
| -------------------- | :-------: | -------------------------------------------------- |
| `CAM_INDEX`          |     2     | Camera device index (try 0/1/2…)                   |
| `EAR_THRESHOLD`      |    0.20   | Eyes considered “closed” when EAR falls below this |
| `CLOSED_TIME`        |    0.8s   | Duration eyes must stay closed to trigger alarm    |
| `RETRIGGER_INTERVAL` |     6s    | Minimum delay between alarm retriggers             |
| `ENABLE_AUDIO`       |     1     | 1=on, 0=off (visual-only)                          |
| `ALARM_PATH`         | alarm.wav | Path to WAV file                                   |
| `FRAME_WIDTH`        |    1280   | Capture width                                      |
| `FRAME_HEIGHT`       |    720    | Capture height                                     |
| `TARGET_FPS`         |     30    | Requested FPS                                      |

---

## 🛠 Troubleshooting
- **No sound** → install system deps: `sudo apt install libportaudio2 alsa-utils`. Avoid launching from a conda shell that overrides system ALSA.
- **Cannot open camera** → try `CAM_INDEX=0` (or 1/2/3). Ensure no other app is using the webcam.
- **WSL/headless** → set `ENABLE_AUDIO=0` (visual alerts only).
- **Performance** → reduce resolution/FPS; close other camera/GPU apps.

---


## 📜 License
MIT — see [LICENSE](LICENSE).

---

## 🙌 Acknowledgements
- [MediaPipe FaceMesh](https://developers.google.com/mediapipe)
- [OpenCV](https://opencv.org/)




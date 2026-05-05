# Stereo Vision System

A real-time obstacle detection and distance estimation system built for the Raspberry Pi 5, using dual RPi camera modules and YOLOv5n.

## How it works

- Two cameras capture a stereo image pair simultaneously
- SGBM stereo matching produces a disparity map, refined with a WLS filter
- YOLOv5n model detects objects in the left frame
- Distance to each detected object is estimated from the depth map and smoothed with an EMA filter
- Voice alerts via `espeak` when an obstacle is within range

## Hardware

- Raspberry Pi 5
- 2x Raspberry Pi Camera Module (8 cm baseline)

## Requirements

```bash
pip install opencv-contrib-python numpy picamera2 onnxruntime
sudo apt install espeak
```

## Setup

**1. Print and mount the calibration board**

Print a 9�6 checkerboard with 1.9 cm  squares and glue it flat onto a rigid surface.

**2. Capture calibration images**

```bash
python capture_calibration.py
```

Press `s` to save a pair, `q` when done. Aim for ~ 20 - 25 accepted pairs at varied distances and angles.

**3. Run calibration**

```bash
python calibration.py
```

Target: stereo RMS < 1.0 px. If above, run `filter_calibration.py` then re-run.

**4. Run**

```bash
python main.py
```

Press `q` to quit.

## Project structure

```
- main.py                  # Main loop
- camera.py                # Camera init and synchronised capture
- stereo.py                # SGBM + WLS depth map generation
- distance.py              # ROI sampling and EMA smoothing
- detector.py              # YOLOv5 ONNX inference
- alerts.py                # Voice alerts via espeak
- config.py                # Tunable parameters
- capture_calibration.py   # Calibration image capture tool
- calibration.py           # Stereo calibration
- filter_calibration.py    # Remove bad calibration pairs
- models/                  # ONNX model weights (not tracked in git)
- calibration/             # Calibration output (not tracked in git)
- test_camera_index        # Camera index identification test
```
# Real-Time Pose-Based Image Matcher

A real-time computer vision application built with Python, OpenCV, and MediaPipe that tracks upper-body skeletal landmarks via webcam and dynamically displays corresponding visual outputs based on live user postures.

---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [License](#license)

---

## Features
- **Real-Time Pose Estimation:** Tracks 33 human body landmarks at low latency using MediaPipe's Pose Landmarker Tasks API.
- **Dynamic Visual Matching:** Maps live user postures to a pre-defined dataset of reference images and updates the display window instantaneously.
- **Automated Model Provisioning:** Automatically downloads and validates the required `.task` landmarker model file on initial launch.
- **Live Debug Telemetry:** Renders real-time wrist tracking markers, Euclidean distance metrics, and active pose classifications directly onto the webcam feed.

---

## Tech Stack
- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (`opencv-python`)
- **Pose Estimation:** MediaPipe (`mediapipe`)
- **Numerical Computing:** NumPy (`numpy`)

---

## How It Works
1. **Frame Capture:** Captures live video frames from the default webcam and mirrors them horizontally for natural user interaction.
2. **Landmark Extraction:** Converts BGR frames to RGB and passes them through the MediaPipe Pose Landmarker to extract normalized $(x, y)$ coordinates for key upper-body joints (wrists, shoulders, and ears).
3. **Spatial Classification:** Computes Euclidean distances and relative vertical positions between joints to classify the active gesture:
   - **Joined Hands:** Distance between left and right wrists falls below the proximity threshold.
   - **Hand Near Head/Ear:** Distance between either wrist and the corresponding ear falls below the proximity threshold.
   - **Raised Hand:** Either wrist is positioned vertically above its corresponding shoulder.
4. **Output Rendering:** Displays the live webcam stream alongside the matched reference image in dedicated OpenCV windows.

---

## Project Structure
```text
pose-image-matcher/
├── app.py                        # Main application script
├── pose_landmarker_lite.task     # MediaPipe pose model (auto-downloaded on first run)
├── ayubowan.jpeg                 # Reference image for joined-hands pose
├── winpos.jpeg                   # Reference image for raised-hand pose
├── angry.jpg                     # Reference image for hand-near-ear pose
└── README.md                     # Project documentation

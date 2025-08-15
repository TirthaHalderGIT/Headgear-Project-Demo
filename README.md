# Headgear-Project-Demo

**Simulating LIDAR-Style Proximity Detection with YOLO & CNN**

---

##  Overview

Headgear-Project-Demo is a Python-based software designed to detect vehicles and other objects dangerously close to a camera—emulating LIDAR proximity sensing using computer vision. It leverages YOLO (You Only Look Once) for real-time object detection and a convolutional neural network (CNN) for classifying distance or proximity risk.

- Detects cars, trucks, motorcycles, and similar vehicles in live video or prerecorded footage.
- Highlights and optionally logs objects within predefined dangerous proximity thresholds.
- Suitable for driver-assistance systems, robotics, or any application requiring real-time hazard detection.

---

##  Features

- **Real-time object detection** using YOLO (version included/configurable).
- **Proximity risk classification** via CNN—evaluates how "near" an object is to the camera to simulate LIDAR data.
- **Video input support**: process live camera, webcam, or prerecorded video files.
- **Annotated output**: bounding boxes, risk labels, distance alerts, and tagged video outputs.
- **Configurable thresholds**: define what you consider "dangerously near."
- **Visual demo assets**: demo video (`output_with_detection.mp4`) to showcase detection and alerts.

---

##  Demo

Check out the demonstration video: `output_with_detection.mp4` shows how detected objects are annotated with bounding boxes and proximity warnings.

---

##  Directory Structure


NOTE- Change the Video Directory in the .py files where you save the video file at

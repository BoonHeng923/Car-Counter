# Car Counter

## Overview
This **Car Counter** projects processes a video file to detect, track, count, and estimate the speed of vehicles using YOLOv8 and OpenCV. The system detecs each vehicle by drawing a bounding box around in in every frame, assigns a unique tracking ID, and monitors its movement across time.

## Approach & Key Ideas
**1. Vehicle Detection**
- The YOLOv8 model from Ultralytics is used to detect vehicles in each frame.
- Only specifci class ID (default: [2] = "car from COCO dataset) are processed.
- Each detected object is enclosed in a bounding box that includes coordinates (x1, y1, x2, y2).
  
**2. Tracking and ID Assigmnent**
- YOLOv8's built in tracking mode, *model.track()* provides consistent object ID between frames.
- Each vehicle's center position is extracted from its bounding box and stored with the corresponding frame number in deque.
- This historical record helps compute distance moved between consecutive frames.

**3. Speed Estimation**
- Once a bounding box width exceeds 40 pixels, the system performs an automatic calibration assuming an average car length of 4.5m.
- Recent speeds per ID are smoothed with a moving average for stability.

**4. Vehicle COunting**
- A verhicle counting line is drawn at a configurable vertical position.
- When a vehicle's bounding box centr moves across this line:
  - From left to right: increment *count_lr*
  - From right to left: increment *count_rl*
 
**5. Visualization and Output**
- Bounding boxes and labels show class name, confidence, and speed.
- Top-left corner displays:
  - Total vehciles
  - Number of vehicles from left to right
  - Number of vehicle from right to left
  - A speed record panel shows speeds of all tracked cars.
 
## Technologies and Tools Used
- **Programming Language:** Python
- **Deep Learning Framework:** Ultralytics YOLOv8
- **Computer Vision Library:** OpenCV
- **Numerical Library:** NumPy
- **Data Structures:** collections(deque, defaultdict)
- **Typing Library:** typing(Deque, Dict, Optional, Tuple)
- **Time Library:** time

## Steps to Reproduce the Results


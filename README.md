# Car Counter

## Overview
This **Car Counter** projects processes a video file to detect, track, count, and estimate the speed of vehicles using YOLOv8 and OpenCV. The system detecs each vehicle by drawing a bounding box around in in every frame, assigns a unique tracking ID, and monitors its movement across time.

## Approach & Key Ideas
**1. Downloading YouTube Video as MP4**
- The video is downloaded as an MP4 file by using the *yt-dlp* Python package.
- Once the video is downloaded, it is saved as *input_video.mp4*, and placed in the same direcory as *car_counter.py*.

**2. Vehicle Detection**
- The YOLOv8 model from Ultralytics is used to detect vehicles in each frame.
- Only specific class ID (default: [2] = "car from COCO dataset) are processed.
- Each detected object is enclosed in a bounding box that includes coordinates (x1, y1, x2, y2).
  
**3. Tracking and ID Assigmnent**
- YOLOv8's built in tracking mode, *model.track()* provides consistent object ID between frames.
- Each vehicle's center position is extracted from its bounding box and stored with the corresponding frame number in deque.
- This historical record helps compute distance moved between consecutive frames.

**4. Speed Estimation**
- Once a bounding box width exceeds 40 pixels, the system performs an automatic calibration assuming an average car length of 4.5m.
- Recent speeds per ID are smoothed with a moving average for stability.

**5. Vehicle Counting**
- A vehicle counting line is drawn at a configurable vertical position.
- When a vehicle's bounding box center moves across this line:
  - From left to right: increment *count_lr*
  - From right to left: increment *count_rl*
 
**6. Visualization and Output**
- Bounding boxes and labels show class name, confidence, and speed.
- Top-left corner displays:
  - Total number of cars
  - Number of cars from left to right
  - Number of cars from right to left
  - A speed record panel shows speeds of all tracked cars.
 
## Technologies and Tools Used
- **Programming Language:** Python
- **Deep Learning Framework:** Ultralytics YOLOv8
- **Computer Vision Library:** OpenCV
- **Numerical Library:** NumPy
- **Data Structures:** collections(deque, defaultdict)
- **Typing Library:** typing(Deque, Dict, Optional, Tuple)
- **Time Library:** time
- **Downloader:** yt-dlp

## Steps to Reproduce the Results
**Step 1: Clone the Repository**
- Clone the project repository from GitHub to the local computer.
  
**Step 2: Install dependencies**
- Install all required packages for this car counter program. *(pip install ultralytics opencv-python numpy yt-dlp)*.
  
**Step 3: Download YOLOv8 model**
- Download the pretrained YOLOv8 model using the Ultralytics CLI.
  
**Step 4: Prepare input video**
- Use the *yt-dlp* script to download the chosen Youtube video and save it as *input_video.mp4* in the project folder.
  
**Step 5: Write the Code for the Main Program**
- Writing the main Python program that integrates vehicle detection, tracking, counting, and speed estimation into a complete automated system using YOLOv8 and OpenCV.

**Step 6: Run the Program**
- Ensure the main Python file to start detection, tracking, and speed estimation.
- The program will:
  - Detect and track vehicles frame by frame.
  - Draw bounding boxes with live speed labels.
  - Count vehicles crossing the center line in both directions.

**Step 7: Check the Output**
- Once processing is completed, an output video named *output_video.mp4* will be created in the project directory.
- The output will show:
  - Total number of cars
  - Number of cars from left to right
  - Number of cars from right to left
  - A speed record panel shows speeds of all tracked cars.
  - Cars with bounding boxes

## How to Run
1. Put your input video in the project folder and name it *input_video.mp4* (or change the constant in the script).
2. Run the main script: *python car_counter.py*
3. The processed video will be saved as *output_video.mp4* in the same folder.
   
## Output
<img width="1603" height="901" alt="image" src="https://github.com/user-attachments/assets/64080ab7-fe80-4237-90fe-68cd6777abc0" />

## Credit
**Developer:** Ong Boon Heng <br>
**Date:** October 2025



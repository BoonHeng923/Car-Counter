from ultralytics import YOLO
import cv2
import time

INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
FPS_FALLBACK = 25.0

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit("Error! Cannot open the input video!")

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

if fps <= 0:
    print(f"Warning: Input FPS reported as {fps}. Falling back to {FPS_FALLBACK}")
    fps = FPS_FALLBACK

duration = (frame_count / fps) if frame_count and fps else None
print(f"Input FPS = {fps}, frames = {frame_count}, duration(s) = {duration}, size = {w}*{h}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(fps), (w, h))

model = YOLO("yolov8n.pt")

written = 0
start = time.time()
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    results = model(frame)
    annotated = results[0].plot()

    out.write(annotated)
    written += 1

    if frame_no % 200 == 0:
        print(f"Processed {frame_no} frames, wirtten = {written}")

cap.release()
out.release()
print(f"Finished. Frames written: {written}. Expected frames: {frame_count}")
print(f"Estimated output duration(s): {written / fps: .2f}")
print("Saved to", OUTPUT_VIDEO)
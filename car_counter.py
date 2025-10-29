from ultralytics import YOLO
import cv2
import time
import numpy as np

INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
FPS_FALLBACK = 25.0
LINE_X_RATIO = 0.5
CONFIDENCE = 0.45
CLASS_ID = [2]

model = YOLO("yolov8n.pt")

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

print(f"Video: {w} * {h} @ {fps} FPS, {frame_count} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(fps), (w, h))

track_history = {}
counted_id = set()
count_lr = 0
count_rl = 0
line_x = int(w * LINE_X_RATIO)

start = time.time()
frame_no = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    results = model.track(frame, persist = True, conf = CONFIDENCE, classes = CLASS_ID)
    annotated = results[0].plot() if hasattr(results[0], "plot") else frame.copy()

    boxes = results[0].boxes 
    if boxes is not None and len(boxes) > 0 and getattr(boxes, "id", None) is not None:
        ids_np = boxes.id.cpu().numpy()
        if hasattr(boxes, "xywh"):
            xywh = boxes.xywh.cpu().numpy()
            centers_x = xywh[:, 0]
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0

        for idx, obj_id in enumerate(ids_np):
            obj_id = int(obj_id)
            cx = float(centers_x[idx])

            if obj_id not in track_history:
                track_history[obj_id] = [cx]
            else:
                history = track_history[obj_id]
                history.append(cx)
                if len(history) > 2:
                    history.pop(0)

                if len(history) == 2 and obj_id not in counted_id:
                    x_prev, x_curr = history[0], history[1]

                    if x_prev < line_x and x_curr >= line_x:
                        count_lr += 1
                        counted_id.add(obj_id)
                        print(f"[Frame {frame_no}] ID {obj_id} From left to right counted. Total from left to right: {count_lr}")

                    elif x_prev > line_x and x_curr <= line_x:
                        count_rl += 1
                        counted_id.add(obj_id)
                        print(f"[Frame {frame_no}] ID {obj_id} From right to left counted. Total from right to left: {count_rl}")

    line_top = int(h * 0.15)
    line_bottom = int (h * 0.85)
    cv2.line(annotated, (line_x, line_top), (line_x, line_bottom), (0, 255, 255), 3)
    total = count_lr + count_rl 
    cv2.putText(annotated, f"Total number of cars: {total}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    cv2.putText(annotated, f"From left to right: {count_lr}", (20,85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, f"From right to left: {count_rl}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    out.write(annotated)

    if frame_no % 200 == 0:
        print(f"Processed {frame_no} / {frame_count} frames - total: {total} (From left to right {count_lr}, From right to left {count_rl}")

cap.release()
out.release()

print(f"Finished. Frames processed: {frame_no}. From left to right: {count_lr}, From right to left: {count_rl}, Total number of cars: {count_lr + count_rl}")
print("Saved to", OUTPUT_VIDEO)
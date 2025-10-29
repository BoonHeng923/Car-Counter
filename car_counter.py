from ultralytics import YOLO
import cv2
import time
import numpy as np
from collections import deque, defaultdict

INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"
FPS_FALLBACK = 25.0
LINE_X_RATIO = 0.5
CONFIDENCE = 0.45
CLASS_ID = [2]
ASSUMED_CAR_LENGTH = 4.5
SPEED_MOTION = 5
MIN_DT = 1e-3

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise SystemExit("Error! Cannot open the input video!")

fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

if fps <= 0:
    fps = FPS_FALLBACK

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, float(fps), (w, h))

track_history = {}
per_id_speed = defaultdict(lambda: deque(maxlen = SPEED_MOTION))
per_id_last_speed = {}
finalized_speed = {}
car_number_map = {}
numbered_order = []
next_car_number = 1
last_seen = {}
counted_id = set()
count_lr = 0
count_rl = 0
line_x = int(w * LINE_X_RATIO)
meters_per_pixel = None

start = time.time()
frame_no = 0

def to_numpy(x):
    try:
        return x.cpu().numpy()
    except:
        return np.array(x)
    
print("Processing started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1

    results = model.track(frame, persist = True, conf = CONFIDENCE, classes = CLASS_ID)
    r = results[0]
    annotated = frame.copy()
    boxes = getattr(r, "boxes", None)

    if boxes is not None and len(boxes) > 0 and getattr(boxes, "id", None) is not None:
        ids_np = to_numpy(boxes.id).astype(int)

        if hasattr(boxes, "xywh"):
            xywh = to_numpy(boxes.xywh)
            centers_x = xywh[:, 0]
            widths_px = xywh[:, 2]
        else:
            xyxy = to_numpy(boxes.xyxy)
            centers_x = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
            widths_px = (xyxy[:, 2] - xyxy[:, 0])

        for i, obj_id in enumerate(ids_np):
            cx = float(centers_x[i])
            bw = float(widths_px[i])
            last_seen[obj_id] = frame_no

            if obj_id not in track_history:
                track_history[obj_id] = deque(maxlen = 10)
            track_history[obj_id].append((frame_no, cx))

            if meters_per_pixel is None and bw > 40:
                meters_per_pixel = ASSUMED_CAR_LENGTH / bw 
                print(f"[Calibration] meters_per_pixel = {meters_per_pixel:.7f} m/px (bbox {bw:.1f}px)")

            if len(track_history[obj_id]) >= 2 and meters_per_pixel is not None:
                f_old, x_old = track_history[obj_id][-2]
                f_new, x_new = track_history[obj_id][-1]
                dx_pixels = x_new - x_old
                dt_seconds = max((f_new - f_old) / fps, MIN_DT)
                speed_ms = (abs(dx_pixels) * meters_per_pixel) / dt_seconds
                speed_kmh = speed_ms * 3.6
                per_id_speed[obj_id].append(speed_kmh)
                per_id_last_speed[obj_id] = float(np.mean(per_id_speed[obj_id]))

            if hasattr(boxes, "xyxy"):
                coords = to_numpy(boxes.xyxy)[i]
                x1, y1, x2, y2 = map(int, coords[:4])
            else:
                x1, y1, x2, y2 = int(cx -bw/2), int (h * 0.3), int (cx + bw/2), int(h * 0.4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)

            cls_i = int(to_numpy(boxes.cls)[i]) if getattr(boxes, "cls", None) is not None else 2
            conf_i = float(to_numpy(boxes.conf)[i]) if getattr(boxes, "conf", None) is not None else 0.0
            class_name = model.names[cls_i] if hasattr(model, "names") else str(cls_i)

            if obj_id in per_id_speed and len(per_id_speed[obj_id]) > 0:
                avg_car_speed = float(np.mean(per_id_speed[obj_id]))
                label = f"{class_name} {conf_i:.2f} | {avg_car_speed:.1f} km/h"
            else:
                label = f"{class_name} {conf_i:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_label = max(5, y1)
            cv2.rectangle(annotated, (x1, y_label - th -6), (x1 + tw + 4, y_label), (50, 50, 50), -1)
            cv2.putText(annotated, label, (x1 + 2, y_label - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if len(track_history[obj_id]) >= 2 and obj_id not in counted_id:
                x_prev = track_history[obj_id][-2][1]
                x_curr = track_history[obj_id][-1][-1]
                if x_prev < line_x and x_curr >= line_x:
                    count_lr += 1
                    counted_id.add(obj_id)
                elif x_prev > line_x and x_curr <= line_x:
                    count_rl += 1
                    counted_id.add(obj_id)
            
            if obj_id in per_id_last_speed:
                finalized_speed[obj_id] = per_id_last_speed[obj_id]

                if obj_id not in car_number_map:
                    car_number_map[obj_id] = next_car_number
                    numbered_order.append(obj_id)
                    next_car_number += 1

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
    
    if numbered_order:
        lines_count = len(numbered_order)
        box_w = 320
        box_h = 26 * (lines_count + 1) + 20
        box_x1, box_y1 = 15, 150
        box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h
        cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
        cv2.putText(annotated, "Speeds Record (km/h):", (box_x1 + 10, box_y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset = box_y1 + 60
        for display_idx, obj_id in enumerate(numbered_order, start = 1):
            sp = finalized_speed.get(obj_id, None)
            if sp is not None:
                text = f"Car {display_idx}: {sp:.1f}"
            else:
                text = f"Car {display_idx}: -"
            cv2.putText(annotated, text, (box_x1 + 10, y_offset + (display_idx - 1) * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(annotated)

    timeout_frames = int(1.5 * fps)
    to_clean = [tid for tid, lastf in list(last_seen.items()) if frame_no - lastf > timeout_frames]
    for tid in to_clean:
        track_history.pop(tid, None)
        per_id_speed.pop(tid, None)
        per_id_last_speed.pop(tid, None)
        last_seen.pop(tid, None)

    if frame_no % 200 == 0:
        print(f"Processed {frame_no} / {frame_count} frames - total: {total} From left to right {count_lr}, From right to left {count_rl}")

cap.release()
out.release()

print(f"Finished. Frames processed: {frame_no}. From left to right: {count_lr}, From right to left: {count_rl}, Total number of cars: {count_lr + count_rl}")
print("Saved to", OUTPUT_VIDEO)
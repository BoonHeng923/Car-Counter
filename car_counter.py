import cv2
import time
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
from typing import Deque, Dict, Optional, Tuple

INPUT_VIDEO = "input_video.mp4"
OUTPUT_VIDEO = "output_video.mp4"

FPS_FALLBACK = 25.0
LINE_X_RATIO = 0.5
CONFIDENCE = 0.45
CLASS_ID = [2]
ASSUMED_CAR_LENGTH = 4.5
SPEED_SMOOTH_LEN = 5
MIN_DT = 1e-3
CALIB_BBOX_MIN_PX = 40

MODEL_PATH = "yolov8n.pt"
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

def to_numpy(x) -> np.ndarray:
    try:
        return x.cpu().numpy()
    except Exception:
        return np.array(x)
    
class CarCounter:
    def __init__(self, model: YOLO):
        self.model = model 

        self.track_history: Dict[int, Deque[Tuple[int, float]]] = {}
        self.per_id_speed: Dict[int, Deque[float]] = defaultdict(lambda: deque(maxlen=SPEED_SMOOTH_LEN))
        self.per_id_last_speed: Dict[int, float] = {}
        self.finalized_speed: Dict[int, float] = {}
        self.car_number_map: Dict[int, int] = {}
        self.numbered_order: list[int] = []
        self.next_car_number = 1
        self.last_seen: Dict[int, int] = {}
        self.counted_id: set[int] = set()

        self.count_lr = 0
        self.count_rl = 0
        self.meters_per_pixel: Optional[float] = None

    def process_video(self, input_path: str, output_path: str):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise SystemExit("Error! Cannot open the input video!")

        fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        line_x = int(w * LINE_X_RATIO)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

        frame_no = 0
        start_t = time.time()

        print("Processing started...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            self._process_frame(frame, frame_no, fps, line_x, w, h)

            out.write(frame)

            if frame_no % 200 == 0:
                total = self.count_lr + self.count_rl
                print(
                    f"Processed {frame_no} / {frame_count} frames - total: {total}"
                      f"From left to right: {self.count_lr}, From right to left: {self.count_rl}"
                )
                
        cap.release()
        out.release()
        elapsed = time.time() - start_t
        total = self.count_lr + self.count_rl
        print(
            f"Finished. Frames processed: {frame_no}. From left to right: {self.count_lr}, "
            f"From right to left: {self.count_rl}, Total: {total}. Elapsed: {elapsed:.1f}s"
        )    

    def _process_frame(self, frame, frame_no: int, fps: float, line_x: int, w: int, h: int):
        results = self.model.track(frame, persist=True, conf=CONFIDENCE, classes=CLASS_ID)
        r = results[0]
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
                self.last_seen[obj_id] = frame_no

                if obj_id not in self.track_history:
                    self.track_history[obj_id] = deque(maxlen = 10)
                self.track_history[obj_id].append((frame_no, cx))

                if self.meters_per_pixel is None and bw > 40:
                    self.meters_per_pixel = ASSUMED_CAR_LENGTH / bw 
                    print(f"[Calibration] meters_per_pixel = {self.meters_per_pixel:.7f} m/px (bbox {bw:.1f}px)")

                if len(self.track_history[obj_id]) >= 2 and self.meters_per_pixel is not None:
                    f_old, x_old = self.track_history[obj_id][-2]
                    f_new, x_new = self.track_history[obj_id][-1]
                    dx_pixels = x_new - x_old
                    dt_seconds = max((f_new - f_old) / fps, MIN_DT)
                    speed_ms = (abs(dx_pixels) * self.meters_per_pixel) / dt_seconds
                    speed_kmh = speed_ms * 3.6
                    self.per_id_speed[obj_id].append(speed_kmh)
                    self.per_id_last_speed[obj_id] = float(np.mean(self.per_id_speed[obj_id]))

                if hasattr(boxes, "xyxy"):
                    coords = to_numpy(boxes.xyxy)[i]
                    x1, y1, x2, y2 = map(int, coords[:4])
                else:
                    x1, y1, x2, y2 = int(cx -bw/2), int (h * 0.3), int (cx + bw/2), int(h * 0.4)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cls_i = int(to_numpy(boxes.cls)[i]) if getattr(boxes, "cls", None) is not None else 2
                conf_i = float(to_numpy(boxes.conf)[i]) if getattr(boxes, "conf", None) is not None else 0.0
                class_name = self.model.names[cls_i] if hasattr(self.model, "names") else str(cls_i)

                if obj_id in self.per_id_speed and len(self.per_id_speed[obj_id]) > 0:
                    avg_car_speed = float(np.mean(self.per_id_speed[obj_id]))
                    label = f"{class_name} {conf_i:.2f} | {avg_car_speed:.1f} km/h"
                else:
                    label = f"{class_name} {conf_i:.2f}"

                (tw, th), _ = cv2.getTextSize(label, TEXT_FONT, 0.6, 2)
                y_label = max(5, y1)
                cv2.rectangle(frame, (x1, y_label - th -6), (x1 + tw + 4, y_label), (50, 50, 50), -1)
                cv2.putText(frame, label, (x1 + 2, y_label - 3),
                            TEXT_FONT, 0.6, (255, 255, 255), 2)
                
                if len(self.track_history[obj_id]) >= 2 and obj_id not in self.counted_id:
                    x_prev = self.track_history[obj_id][-2][1]
                    x_curr = self.track_history[obj_id][-1][-1]
                    if x_prev < line_x and x_curr >= line_x:
                        self.count_lr += 1
                        self.counted_id.add(obj_id)
                    elif x_prev > line_x and x_curr <= line_x:
                        self.count_rl += 1
                        self.counted_id.add(obj_id)
                
                if obj_id in self.per_id_last_speed:
                    self.finalized_speed[obj_id] = self.per_id_last_speed[obj_id]

                    if obj_id not in self.car_number_map:
                        self.car_number_map[obj_id] = self.next_car_number
                        self.numbered_order.append(obj_id)
                        self.next_car_number += 1

        self._draw_overlays(frame, line_x, w, h)
    
        timeout_frames = int(1.5 * (fps or FPS_FALLBACK))
        to_remove = [tid for tid, last in list(self.last_seen.items()) if frame_no - last > timeout_frames]
        for tid in to_remove:
            self.track_history.pop(tid, None)
            self.per_id_speed.pop(tid, None)
            self.per_id_last_speed.pop(tid, None)
            self.last_seen.pop(tid, None)

    def _draw_overlays(self, frame, line_x: int, w:int, h: int):
        line_top = int(h * 0.15)
        line_bottom = int (h * 0.85)
        cv2.line(frame, (line_x, line_top), (line_x, line_bottom), (0, 255, 255), 3)

        total = self.count_lr + self.count_rl 
        cv2.putText(frame, f"Total number of cars: {total}", (20,40), TEXT_FONT, 1.1, (0, 255, 0), 3)
        cv2.putText(frame, f"From left to right: {self.count_lr}", (20,85), TEXT_FONT, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"From right to left: {self.count_rl}", (20,120), TEXT_FONT, 0.9, (255, 255, 255), 2)
        
        if self.numbered_order:
            lines_count = len(self.numbered_order)
            box_w = 320
            box_h = 26 * (lines_count + 1) + 20
            box_x1, box_y1 = 15, 150
            box_x2, box_y2 = box_x1 + box_w, box_y1 + box_h
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), -1)
            cv2.putText(frame, "Speeds Record (km/h):", (box_x1 + 10, box_y1 + 30), TEXT_FONT, 0.7, (255, 255, 0), 2)
            
            y_offset = box_y1 + 60
            for display_idx, obj_id in enumerate(self.numbered_order, start = 1):
                sp = self.finalized_speed.get(obj_id)
                if sp is not None:
                    text = f"Car {display_idx}: {sp:.1f}"
                else:
                    text = f"Car {display_idx}: -"
                cv2.putText(frame, text, (box_x1 + 10, y_offset + (display_idx - 1) * 26), TEXT_FONT, 0.7, (255, 255, 255), 2)

def main():
    model = YOLO(MODEL_PATH)
    counter = CarCounter(model)
    counter.process_video(INPUT_VIDEO, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
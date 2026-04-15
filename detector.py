"""
Player detection using the provided fine-tuned YOLOv11 model.
Wraps Ultralytics inference and returns clean detection results per frame.
"""
import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

import config


@dataclass
class Detection:
    """Single player detection in a frame."""
    bbox: np.ndarray   # [x1, y1, x2, y2]
    conf: float
    class_id: int
    crop: np.ndarray   # BGR image crop of the player


class Detector:
    """YOLOv11 player detector."""

    def __init__(self, model_path: str = None):
        path = model_path or str(config.MODEL_PATH)
        self.model = YOLO(path)
        print(f"[Detector] Loaded model from {path}")
        print(f"[Detector] Class names: {self.model.names}")

    def detect(self, frame: np.ndarray, conf: float = None) -> list[Detection]:
        """Run detection on a single BGR frame. Returns list of Detections."""
        results = self.model(
            frame,
            conf=conf or config.DETECTION_CONF,
            iou=config.DETECTION_IOU,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in (config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID):
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = frame[y1:y2, x1:x2].copy()
            
            # Filter out tiny partial detections (ghost players)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 1500:
                continue
 
            if crop.size == 0:
                continue

            detections.append(Detection(
                bbox=xyxy,
                conf=float(box.conf[0]),
                class_id=cls_id,
                crop=crop,
            ))

        return detections


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else str(config.VIDEO_PATH)
    detector = Detector()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {video_path} | {total} frames @ {fps} FPS")

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        sys.exit(1)

    dets = detector.detect(frame)
    print(f"[Frame 0] Found {len(dets)} players")
    for i, d in enumerate(dets):
        print(f"  Player {i}: bbox={d.bbox}, conf={d.conf:.2f}, crop={d.crop.shape}")

    # Save annotated first frame
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.OUTPUT_DIR / "step1_detection_test.jpg")
    cv2.imwrite(out_path, frame)
    print(f"[Saved] {out_path}")

    cap.release()
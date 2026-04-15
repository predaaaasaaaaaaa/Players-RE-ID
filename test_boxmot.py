"""Quick test: BoxMOT DeepOCSORT + OSNet re-ID."""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOcSort

import config

# Load detector
model = YOLO(str(config.MODEL_PATH))
print(f"[Detector] Loaded {config.MODEL_PATH}")

# Load tracker with OSNet re-ID (auto-downloads weights)
tracker = DeepOcSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device="cpu",
    half=False,
)
print("[Tracker] DeepOcSort + OSNet loaded")

# Run on first frame
cap = cv2.VideoCapture(str(config.VIDEO_PATH))
ret, frame = cap.read()
cap.release()

# Detect
results = model(frame, conf=config.DETECTION_CONF, verbose=False)[0]
dets = []
for box in results.boxes:
    cls_id = int(box.cls[0])
    if cls_id not in (config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID):
        continue
    xyxy = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0])
    dets.append([*xyxy, conf, cls_id])

dets = np.array(dets) if dets else np.empty((0, 6))
print(f"[Detections] {len(dets)} players")

# Track
tracks = tracker.update(dets, frame)
print(f"[Tracks] {len(tracks)} tracked")
for t in tracks:
    x1, y1, x2, y2, tid, conf, cls, idx = t
    print(f"  Track {int(tid)}: [{int(x1)},{int(y1)},{int(x2)},{int(y2)}] conf={conf:.2f}")

print("\n[OK] BoxMOT works!")
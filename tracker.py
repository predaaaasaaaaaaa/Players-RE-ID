"""
Frame-to-frame player tracking using BoxMOT's DeepOcSort.
Uses OSNet (trained on MSMT17 person re-ID dataset) for appearance-based
re-identification — far stronger than generic models for same-team players.
"""
import kalman_patch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from boxmot import DeepOcSort
from dataclasses import dataclass, field

import config


@dataclass
class TrackedPlayer:
    """Single tracked player in a frame."""
    track_id: int
    bbox: np.ndarray       # [x1, y1, x2, y2]
    conf: float
    class_id: int
    crop: np.ndarray       # BGR image crop


@dataclass
class FrameResult:
    """All tracked players in a single frame."""
    frame_idx: int
    players: list[TrackedPlayer] = field(default_factory=list)


class Tracker:
    """DeepOcSort player tracker with OSNet re-ID."""

    def __init__(self, model_path: str = None):
        path = model_path or str(config.MODEL_PATH)
        self.model = YOLO(path)
        self.player_classes = [config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID]
        self.tracker = DeepOcSort(
            reid_weights=Path("osnet_x0_25_msmt17.pt"),
            device="cpu",
            half=False,
        )
        print(f"[Tracker] Loaded detector from {path}")
        print(f"[Tracker] DeepOcSort + OSNet re-ID ready")
        print(f"[Tracker] Tracking classes: {self.player_classes}")

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection, return Nx6 array [x1,y1,x2,y2,conf,cls]."""
        results = self.model(
            frame,
            conf=config.DETECTION_CONF,
            iou=config.DETECTION_IOU,
            verbose=False,
        )[0]

        dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.player_classes:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            # Min area filter
            if (x2 - x1) * (y2 - y1) < 1000:
                continue
            conf = float(box.conf[0])
            dets.append([x1, y1, x2, y2, conf, cls_id])

        return np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

    def track_video(self, video_path: str = None) -> list[FrameResult]:
        """Run tracking on full video. Returns list of FrameResults."""
        path = video_path or str(config.VIDEO_PATH)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] {path} | {total} frames @ {fps} FPS")

        all_frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            dets = self._detect(frame)
            tracks = self.tracker.update(dets, frame)

            frame_result = FrameResult(frame_idx=frame_idx)

            for t in tracks:
                x1, y1, x2, y2, tid, conf, cls, idx = t
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))

                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                frame_result.players.append(TrackedPlayer(
                    track_id=int(tid),
                    bbox=np.array([x1, y1, x2, y2]),
                    conf=float(conf),
                    class_id=int(cls),
                    crop=crop,
                ))

            all_frames.append(frame_result)

            if frame_idx % 50 == 0:
                print(f"  [Frame {frame_idx}] {len(frame_result.players)} players tracked")

            frame_idx += 1

        cap.release()
        print(f"[Tracker] Done. {len(all_frames)} frames processed.")
        return all_frames


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    tracker = Tracker()
    frames = tracker.track_video()

    # Print summary
    all_ids = set()
    for fr in frames:
        for p in fr.players:
            all_ids.add(p.track_id)

    print(f"\n=== Tracking Summary ===")
    print(f"Total frames: {len(frames)}")
    print(f"Unique track IDs: {len(all_ids)}")
    print(f"Track IDs: {sorted(all_ids)}")

    # Per-track lifespan
    track_spans = {}
    for fr in frames:
        for p in fr.players:
            if p.track_id not in track_spans:
                track_spans[p.track_id] = [fr.frame_idx, fr.frame_idx]
            track_spans[p.track_id][1] = fr.frame_idx

    print(f"\n=== Track Lifespans ===")
    for tid in sorted(track_spans):
        start, end = track_spans[tid]
        print(f"  Track {tid}: frames {start}-{end} ({end - start + 1} frames)")

    # Save summary
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "step7_deepocsort_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"Total frames: {len(frames)}\n")
        f.write(f"Unique track IDs: {len(all_ids)}\n")
        f.write(f"Track IDs: {sorted(all_ids)}\n\n")
        for tid in sorted(track_spans):
            start, end = track_spans[tid]
            f.write(f"Track {tid}: frames {start}-{end} ({end - start + 1} frames)\n")
    print(f"[Saved] {out_path}")
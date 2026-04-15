"""
Frame-to-frame player tracking using BoT-SORT via Ultralytics.
Uses native re-ID features from the YOLO model itself.
"""
import cv2
import numpy as np
from ultralytics import YOLO
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
    """BoT-SORT player tracker with native re-ID."""

    def __init__(self, model_path: str = None):
        path = model_path or str(config.MODEL_PATH)
        self.model = YOLO(path)
        self.player_classes = [config.PLAYER_CLASS_ID, config.GOALKEEPER_CLASS_ID]
        print(f"[Tracker] Loaded model from {path}")
        print(f"[Tracker] Tracking classes: {self.player_classes}")

    def track_video(self, video_path: str = None) -> list[FrameResult]:
        """Run tracking on full video. Returns list of FrameResults."""
        path = video_path or str(config.VIDEO_PATH)
        results = self.model.track(
            source=path,
            tracker="botsort_custom.yaml",
            conf=config.DETECTION_CONF,
            iou=config.DETECTION_IOU,
            persist=True,
            stream=True,
            classes=self.player_classes,
            verbose=False,
        )

        all_frames = []
        for frame_idx, result in enumerate(results):
            frame = result.orig_img
            h, w = frame.shape[:2]
            frame_result = FrameResult(frame_idx=frame_idx)

            if result.boxes is not None and result.boxes.id is not None:
                for box in result.boxes:
                    if box.id is None:
                        continue

                    track_id = int(box.id[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    x1, y1, x2, y2 = xyxy
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Min area filter
                    if (x2 - x1) * (y2 - y1) < 1000:
                        continue

                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size == 0:
                        continue

                    frame_result.players.append(TrackedPlayer(
                        track_id=track_id,
                        bbox=xyxy,
                        conf=conf,
                        class_id=cls_id,
                        crop=crop,
                    ))

            all_frames.append(frame_result)

            if frame_idx % 50 == 0:
                print(f"  [Frame {frame_idx}] {len(frame_result.players)} players tracked")

        print(f"[Tracker] Done. {len(all_frames)} frames processed.")
        return all_frames


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    tracker = Tracker()
    frames = tracker.track_video()

    all_ids = set()
    for fr in frames:
        for p in fr.players:
            all_ids.add(p.track_id)

    print(f"\n=== Tracking Summary ===")
    print(f"Total frames: {len(frames)}")
    print(f"Unique track IDs: {len(all_ids)}")
    print(f"Track IDs: {sorted(all_ids)}")

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

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "tracking_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"Total frames: {len(frames)}\n")
        f.write(f"Unique track IDs: {len(all_ids)}\n\n")
        for tid in sorted(track_spans):
            start, end = track_spans[tid]
            f.write(f"Track {tid}: frames {start}-{end} ({end - start + 1} frames)\n")
    print(f"[Saved] {out_path}")
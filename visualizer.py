"""
Visualize DeepOcSort tracking results as an annotated output video.
No separate re-ID layer needed — DeepOcSort handles it internally via OSNet.
"""
import cv2
import numpy as np

import config
from tracker import Tracker


# Fixed color palette (BGR)
COLORS = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (0, 130, 200),
    (245, 130, 48),  (145, 30, 180),  (70, 240, 240),  (240, 50, 230),
    (210, 245, 60),  (250, 190, 212), (0, 128, 128),   (220, 190, 255),
    (170, 110, 40),  (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 215, 180), (0, 0, 128),     (128, 128, 128),
    (255, 255, 255), (0, 0, 0),       (100, 100, 100),  (200, 200, 0),
]


def get_color(cid: int) -> tuple:
    return COLORS[cid % len(COLORS)]


def draw_player(frame, bbox, cid):
    x1, y1, x2, y2 = bbox
    color = get_color(cid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)
    label = f"ID:{cid}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (255, 255, 255), 2)


def run_visualizer():
    # 1. Track (DeepOcSort handles re-ID internally)
    tracker = Tracker()
    frames = tracker.track_video()

    # 2. Annotate and write video
    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = str(config.OUTPUT_DIR / "result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame_idx, fr in enumerate(frames):
        ret, frame = cap.read()
        if not ret:
            break

        for player in fr.players:
            draw_player(frame, player.bbox, player.track_id)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"  [Frame {frame_idx}/{len(frames)}] written")

    cap.release()
    writer.release()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    run_visualizer()
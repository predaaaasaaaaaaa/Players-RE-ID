"""
Configuration for the Player Re-ID pipeline.
All tunable parameters and paths live here.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
VIDEO_PATH = PROJECT_ROOT / "data" / "15sec_input_720p.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Detection ──────────────────────────────────────────
DETECTION_CONF = 0.5          # confidence threshold
DETECTION_IOU = 0.5           # NMS IoU threshold
PLAYER_CLASS_ID = 2           # 'player' (confirmed from model)
BALL_CLASS_ID = 0             # 'ball' (confirmed from model)
GOALKEEPER_CLASS_ID = 1       # 'goalkeeper'
REFEREE_CLASS_ID = 3          # 'referee'

# ── Tracking ───────────────────────────────────────────
TRACK_HIGH_THRESH = 0.3       # lower = keep more detections in first pass
TRACK_LOW_THRESH = 0.05       # lower = rescue more low-conf detections
TRACK_BUFFER = 90             # 3x longer buffer to hold lost tracks (was 30)

# ── Re-ID ──────────────────────────────────────────────
REID_SIMILARITY_THRESH = 0.80  # weighted cosine (0.4*hsv + 0.6*deep) — raised from 0.6 after fine-tuned OSNet shifted distribution upward
GALLERY_EMA_ALPHA = 0.9       # exponential moving average for gallery updates
REID_MODEL_NAME = "osnet_x0_25"  # lightweight OSNet variant

# ── Visualization ──────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
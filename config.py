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
DETECTION_CONF = 0.3          # confidence threshold
DETECTION_IOU = 0.5           # NMS IoU threshold
PLAYER_CLASS_ID = 0           # class index for 'player' (verify after first run)
BALL_CLASS_ID = 1             # class index for 'ball' (verify after first run)

# ── Tracking ───────────────────────────────────────────
TRACK_HIGH_THRESH = 0.5       # ByteTrack high detection threshold
TRACK_LOW_THRESH = 0.1        # ByteTrack low detection threshold
TRACK_BUFFER = 30             # frames to keep lost tracks alive

# ── Re-ID ──────────────────────────────────────────────
REID_SIMILARITY_THRESH = 0.6  # cosine similarity threshold for re-id match
GALLERY_EMA_ALPHA = 0.9       # exponential moving average for gallery updates
REID_MODEL_NAME = "osnet_x0_25"  # lightweight OSNet variant

# ── Visualization ──────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
# Players Re-ID — Single-Feed Football Broadcast

Re-identify the same player across a 15-second football broadcast clip, even after they leave frame or get occluded. Built as a hiring test for Liat.ai.

**Match:** Manchester City vs Manchester United (FA Community Shield)
**Input:** `data/15sec_input_720p.mp4` (375 frames @ 25 FPS, 1280×720)

---

## What it does

Takes a single-feed football clip in → writes an annotated MP4 out, where each player keeps the same `ID:n` label throughout the clip (including after disappearing and reappearing).

Pipeline:

```
YOLOv11 detection  →  BoT-SORT tracking  →  team classification
                                                 ↓
annotated video  ←  ReID matcher (gallery + EMA + team constraint)
```

---

## Requirements

- **Python 3.12** (tested on 3.12.10)
- A GPU is optional — CPU works, just slower
- Windows / Linux / macOS
- YOLOv11 model provided by Liat.ai
---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/predaaaasaaaaaaa/Players-RE-ID.git
cd Players-RE-ID

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 3. Install PyTorch
# → GPU (CUDA 12.4):
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# → CPU only:
pip install torch==2.6.0 torchvision==0.21.0

# 4. Install the rest
pip install -r requirements.txt

# 5. Add model + input.mp4
input.mp4 --> data/
model.pt (YOLOv11) --> models/
```

First run will auto-download the OSNet ReID weights (~10MB) from torchreid.

---

## Run it

```bash
python visualizer.py
```

Output: `output/result.mp4` — the annotated video with consistent player IDs.

That's it. No flags, no args. The input clip and YOLO model are already in the repo.

---

## Project structure

```
Players-RE-ID/
├── config.py              # all paths + thresholds
├── detector.py            # YOLOv8 wrapper (loads best.pt)
├── tracker.py             # BoT-SORT with native re-ID
├── team_classifier.py     # 2-cluster KMeans on HSV (team assignment)
├── feature_extractor.py   # HSV histogram + OSNet embedding
├── reid_matcher.py        # gallery + EMA + team constraint + confident storage
├── visualizer.py          # entry point — runs the full pipeline
├── botsort_custom.yaml    # BoT-SORT tracker config
├── models/best.pt         # YOLOv11 trained on football
├── data/15sec_input_720p.mp4
└── output/                # result.mp4 gets written here
```

---

## How it works (in 30 seconds)

1. **Detect** — YOLOv8 finds players (class 2) and goalkeepers (class 1) each frame. Ghost detections filtered by min bbox area (1500px²).
2. **Track** — BoT-SORT assigns short-term track IDs and handles frame-to-frame matching with its own re-ID head.
3. **Classify teams** — KMeans on HSV color histograms from the first 5 frames splits everyone into two clusters (City vs United).
4. **Match re-IDs** — when a player disappears and a new ByteTrack ID appears later, we compare the new crop's feature vector (HSV + OSNet) against lost players' gallery. Match requires:
   - Same team (hard constraint)
   - Cosine similarity ≥ 0.6
   - Dirty-loss IDs (lost during collision) need +0.15 similarity to re-match — prevents ID theft after occlusions
5. **Gallery hygiene** — only update a player's stored embedding when their crop is clean (high confidence, no overlap with others, not too close to another player). Prevents gallery contamination during scrums.

---

## Config

All tunables live in `config.py`. Most important:

| Parameter | Default | What it does |
|---|---|---|
| `DETECTION_CONF` | 0.5 | YOLO confidence floor |
| `REID_SIMILARITY_THRESH` | 0.6 | cosine threshold for re-matching |
| `GALLERY_EMA_ALPHA` | 0.9 | how much to weight old vs new embeddings |
| `TRACK_BUFFER` | 90 | frames to hold a lost track before giving up |

---

## Known limits

- **Same-team discrimination is hard** — players wear identical jerseys, so embedding similarity between two City players is often as high as between two crops of the same player across a camera cut. No single threshold perfectly separates them.
- **Camera cuts** — abrupt shot changes drop re-ID accuracy. The 4 correct re-IDs in the final result all happen within continuous camera shots or gentle pans.
- **Appearance-only ceiling** — this is a fundamental constraint of team sports. The obvious next step to break it is jersey-number OCR, which gives a unique per-player signal.

See `REPORT.md` for the full technical deep-dive — including what was tried, what failed (SoccerNet fine-tuning rabbit hole), and what the diagnostic data revealed about why.

---

## Credits

Built by [Samy Metref](https://github.com/predaaaasaaaaaaa).
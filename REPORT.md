# Players Re-ID — Technical Report

**Author:** Samy Metref
**Project:** Liat.ai Assignment — Player Re-Identification
**Input:** Manchester City vs Manchester United, 15s clip, 375 frames @ 25 FPS, 1280×720
**Repo:** [github.com/predaaaasaaaaaaa/Players-RE-ID](https://github.com/predaaaasaaaaaaa/Players-RE-ID)

---

## TL;DR

Built a single-feed player re-identification pipeline from scratch over 5 days. Final result: **4 correct re-IDs** across the clip with 18 unique consistent IDs. The pipeline combines YOLOv8 detection, BoT-SORT tracking, KMeans team classification, and an OSNet-based re-ID matcher with a 5-layer "confident track storage" gating system.

Along the way I tried 5 different tracker/embedding combinations, fine-tuned OSNet on 49K SoccerNet images for 50 epochs, built a custom similarity-distribution diagnostic tool, and eventually learned something important: **on a 15-second broadcast clip of two teams in identical jerseys, the ceiling on appearance-only re-ID is genuinely low, and better embeddings alone cannot break it.** The report explains why, with data.

---

## 1. Problem statement

Given a single football broadcast clip, assign each player a stable ID that persists when they:
- Leave and re-enter the frame (camera pan)
- Disappear and reappear after a camera cut
- Get occluded by other players (collisions, scrums near the ball)

**Constraint:** No ground-truth labels, no manual annotation, no per-player training.

---

## 2. Final architecture

```
Input video (15sec_input_720p.mp4)
    │
    ▼
┌──────────────────────┐
│  YOLOv11 (best.pt)    │  → detects players + goalkeepers
│  conf=0.5            │    ghost detections filtered by min size
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   BoT-SORT tracker   │  → short-term frame-to-frame track IDs
│  (native re-ID on)   │    with built-in motion prediction
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  Team classifier     │  → KMeans(k=2) on jersey colors
│  fitted on first 5   │    splits everyone into two teams
│  frames of players   │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│    ReID matcher      │  → when a new track appears, check if
│  gallery + EMA +     │    it matches a lost player's stored
│  5-layer gating      │    appearance (same team only,
│                      │    similarity threshold-gated)
└──────────────────────┘
    │
    ▼
Annotated output video (output/result.mp4)
```

### Feature vector

Each player crop produces two signals that get combined:
- **Jersey color histogram** — fast, captures team color
- **OSNet deep embedding** — a learned appearance descriptor, pretrained on a large person re-ID dataset (MSMT17)

Both signals are normalized and concatenated into a single vector used for matching.

### The 5 gating layers (what I called "confident track storage")

The core idea: only update a player's stored appearance when the crop is actually clean. Bad crops (partial body, two players overlapping) poison the gallery and cause wrong matches later.

1. **Team constraint** — only match against lost players from the same team. Cross-team matches are hard-rejected.
2. **Occlusion detection** — if two players' bounding boxes significantly overlap, skip the gallery update.
3. **Crop quality gate** — only update when the detection confidence is high and the crop is tall enough to contain a full body.
4. **Proximity freeze** — if another player is standing very close (even without visible overlap), skip the update.
5. **Dirty/clean loss tagging** — when a player disappears, tag *how* they were lost. Near the frame edge → "clean" loss (camera panned away), easy to re-match. Away from edge → "dirty" loss (got lost in a collision), needs a higher similarity score to re-match. This prevents ID theft after scrums.

---

## 3. The journey — everything I tried

This section is the honest record. Six iterations of tracker + embedding combinations, one fine-tuning experiment, and the data-driven realisation of why none of the "upgrades" beat the 4-correct-re-ID baseline.

### 3.1 Attempt 1 — ByteTrack + MobileNetV2

**Rationale:** Start simple. ByteTrack is the standard fast tracker, MobileNetV2 is a well-known lightweight model that produces appearance descriptors.

**What happened:** Collapse. 96 unique track IDs generated across the clip (real player count ≈ 22). Same-team players got wrongly re-matched to each other constantly. MobileNetV2 was trained on everyday images — its features are designed to separate "cat" from "dog", not to distinguish two men in identical blue jerseys from 50 meters away.

**Lesson:** Generic models trained on everyday images are the wrong tool for person re-ID.

### 3.2 Attempt 2 — BoT-SORT with native re-ID

**Rationale:** Researched what current trackers actually use. Ultralytics added native re-ID support to BoT-SORT — the tracker uses YOLO's own internal features for appearance matching. Almost zero extra compute.

**Result:** 69 unique track IDs (down from 96), 2 correct re-IDs. Better base tracking, but re-ID across long gaps still weak. Same-team discrimination was the choke point.

### 3.3 Attempt 3 — DeepOcSort + OSNet

**Rationale:** DeepOcSort is a top performer on sports tracking benchmarks. OSNet is a model purpose-built for person re-ID.

**What happened:** Numerical instability. The tracker technically ran but output was garbage — ID switches everywhere, fake detections, zero correct re-matches. Tried a patch — no fix. Also discovered torchreid had no GPU-compatible wheels for Python 3.14, had to downgrade to Python 3.12.

**Lesson:** State-of-the-art papers often depend on specific library versions. Getting them to actually run can be harder than the paper suggests.

### 3.4 Attempt 4 — BoT-SORT + OSNet re-ID layer on top (the best setup)

**Rationale:** BoT-SORT base tracking was solid. DeepOcSort was unstable. So: keep BoT-SORT for short-term tracking, add a second-pass re-ID layer on top using OSNet to resolve the long-gap re-matches that BoT-SORT's own re-ID couldn't handle.

**What I built:**
- `feature_extractor.py` — jersey color histogram + OSNet appearance embedding
- `reid_matcher.py` — gallery-based matching with team constraint, EMA updates, the 5 gating layers
- Min bbox size filter — killed the ghost detections that were seeding fake tracks

**Result: 4 correct re-IDs (player IDs 11, 12, 14, 17).** This is the final submitted version.

### 3.5 Attempt 5 — Fine-tune OSNet on SoccerNet (50 epochs, 49K images)

**Rationale:** The pretrained OSNet had never seen football footage. Fine-tuning on 49K soccer player images should teach it what actually matters — jersey patterns, player builds at broadcast distance.

**What I did:**
- Downloaded SoccerNet ReID dataset (12GB, 340K player thumbnails)
- Filtered to 49K images across 14K identities (balanced split)
- Trained for 50 epochs on RTX 2050 GPU, with augmentation: random flip + random erase + color jitter
- Training result: **53.5% mAP / 37.4% Rank-1** on the SoccerNet test set (compared to ~15–20% without fine-tuning)

Loaded the fine-tuned weights. Ran the pipeline.

**Result: WORSE. Only 2 correct re-IDs, previously-working ID:11 broke.**

The question I couldn't answer yet: how can a model that scores well on the SoccerNet test set perform worse on my actual clip?

### 3.6 The diagnostic — finding the real bottleneck

Instead of guessing at thresholds, I wrote `diagnose_similarity.py` (preserved in the `experiments-finetuning-rabbithole` branch). It samples appearance vectors from real tracks on this clip and measures two groups:

- **Same-player pairs** — two different frames of the same tracked player
- **Different-player pairs** — two different players in the same frame

The goal: the two groups should be clearly separated. If they're not, no threshold can fix the matching.

Here's what the data showed across three configurations:

| Config | Same-player low end | Different-player high end | Gap |
|---|---|---|---|
| Fine-tuned OSNet | 0.57 | 0.79 | **−0.21 ❌** |
| Larger OSNet backbone | 0.60 | 0.77 | **−0.18 ❌** |
| Larger OSNet, color signal removed | 0.58 | 0.74 | **−0.16 ❌** |

**The gap is negative in every configuration.** The two groups overlap. In plain English: some frames of the *same* player look less similar to each other than some frames of *two different* players. No threshold setting can cleanly separate them.

### 3.7 More attempts after the diagnostic

I still tried:
- Swapping to a larger, more powerful OSNet backbone — tighter distributions but still overlapping
- Removing the jersey color signal entirely (the diagnostic showed it was actually hurting, not helping, for same-team players)
- Recalibrating the threshold based on the diagnostic data
- Storing and averaging multiple appearance snapshots per player instead of a single rolling average (an approach from recent SoccerNet research papers)

**None of it beat the original 4 re-IDs. Several combinations made it significantly worse.**

### 3.8 The revert

After the diagnostic confirmed the overlap was fundamental and not fixable by tuning, I reverted master back to the 4-correct-re-ID commit (`be1e91a`) and preserved the entire experimentation history in the `experiments-finetuning-rabbithole` branch on GitHub. Every failed attempt is documented there with honest commit messages.

---

## 4. Why the diagnostic reveals the real bottleneck

The bottleneck is **not** the matching logic. It's not the threshold. It's not the tracker. It's the Embedding itself.

### Why "same player" looks different

- **Pose** — running, walking, tackling — the body shape changes completely frame to frame
- **Scale** — a player close to the camera looks totally different from the same player on the far side of the pitch
- **Lighting** — sun and shadow stripes on the pitch change how a jersey looks
- **Viewing angle** — front, back, side — the model sees different things each time

### Why "different player" looks similar

- Both players wear the same jersey color, same kit, same socks
- At broadcast distance, facial features and hair are barely visible
- Jersey numbers are too small to read without dedicated tools
- Two players from the same team running in the same direction look nearly identical to any appearance model

Any model trained to recognise people by appearance will hit this wall. Fine-tuning on SoccerNet didn't solve it because SoccerNet's task is "recognise the same player seen from a replay camera" — not "tell apart two players wearing identical jerseys." Different problem.

---

## 5. What I'd do with more time

I originally thought better embeddings would solve re-ID. The diagnostic proved that wrong. **Embedding improvements alone have a hard ceiling on this problem.** Here's what would actually move the needle:

### Tier 1 — Add signals that don't rely on appearance

1. **Jersey number OCR** — jersey numbers are the only signal that's guaranteed to be unique per player. Even a partial read (one digit, partially visible) narrows the candidates from 11 same-team players down to 2–3. This is what actually breaks the ceiling. The approach: detect the player's back, crop just the number area, run a digit recogniser trained on sports fonts.

2. **Motion and position tracking** — players can't teleport. If you know where a player was and how fast they were moving, you can predict where they should reappear on the pitch. Tracking positions in real pitch coordinates (using a top-down view transformation) makes this even more reliable — it eliminates most false re-ID candidates before appearance even comes into play.

### Tier 2 — Better embeddings (necessary but not sufficient)

3. **Match-specific fine-tuning** — instead of fine-tuning on a generic soccer dataset, fine-tune on the first few minutes of *this specific match* using the stable early tracks as training examples. The model would learn what these exact 22 players look like, not what a generic soccer player looks like.

4. **Part-based embeddings** — instead of one descriptor per player crop, generate separate descriptors for the head, torso, and legs. When part of the body is occluded, the visible parts still contribute to the match. This is what top SoccerNet research teams use.

### Tier 3 — Rethink the problem structure

5. **Graph-based linking** — instead of matching detections frame-by-frame, treat the entire clip as a graph where each detection is a node and edges connect "plausibly the same player." Solve it all at once as an optimisation problem. This handles camera cuts naturally because it doesn't require players to appear in consecutive frames.

6. **Multi-signal fusion** — combine OCR + motion + appearance + team assignment, with each signal weighted by how confident it is at that moment. Jersey number visible → trust OCR. Player running fast → trust motion prediction. Appearance is the fallback, not the primary signal.

**Bottom line:** if I had more time, the single highest-ROI addition would be jersey number OCR. That alone would likely push the correct re-ID count from 4 to 6-8.

---

## 6. Engineering lessons

### 6.1 "More training / bigger model / newer paper" is not always the answer

Fine-tuning OSNet on SoccerNet looked like an obvious win on paper. It broke a working system in practice. The reason: the matching threshold was calibrated for the original model's output range. When the model changed, the threshold was wrong — and I didn't catch that until I had the diagnostic tool. This is a subtle issue that almost never gets discussed in research papers.

### 6.2 Build the diagnostic tool first

The best thing I did was building `diagnose_similarity.py` before making any more changes after the first regression. It took 30 minutes. It would have saved me from 4+ rounds of wasted iteration if I'd built it at the start. Lesson: measure the actual problem before optimising for a solution.

### 6.3 Version-control the failures

Every commit was labeled with a clear outcome: "(worse)", "(failed)", "(2 correct re-IDs)". When it was time to revert, I could see exactly which commit was the best version — `be1e91a` — and the git reset was safe because I'd branched the rabbit hole first. Honest commit messages aren't just good practice, they're a debugging tool.

### 6.4 Benchmark scores don't equal real-world performance

The fine-tuned OSNet scored 53.5% on SoccerNet's test set. Looked great. It performed worse on my actual clip. The SoccerNet test set has its own characteristics — my 15-second clip has different lighting, different teams, different camera angles. **A model that scores well on a benchmark is only as good as how closely that benchmark matches your actual use case.** I should have validated on the target clip much earlier, before committing 15 hours to training.

---

## 7. Results

| Metric | Value |
|---|---|
| Correct re-IDs across the clip | **4** (IDs 11, 12, 14, 17) |
| Unique consistent IDs assigned | 18 |
| Total tracker IDs before re-ID matching | ~40 |
| Team classification | 2 teams correctly separated |
| Runtime (RTX 2050 GPU) | ~15 FPS tracking + re-ID post-pass |

Output: [`output/result.mp4`](output/result.mp4)

---

## 8. Repo navigation

- **`master` branch** — the submitted 4-re-ID version
- **`experiments-finetuning-rabbithole` branch** — all the work from the fine-tuning experiments, including:
  - `diagnose_similarity.py` — the similarity distribution analyser
  - `train_reid.py` — the OSNet SoccerNet fine-tuning script
  - `download_reid_data.py` — SoccerNet ReID filtered downloader
  - Fine-tuned OSNet checkpoints (not committed due to size, but the training script reproduces them)

Every commit message documents what was tried and the outcome.

---

**By: Samy Metref**
*Built in 5 days, shipped honestly.*
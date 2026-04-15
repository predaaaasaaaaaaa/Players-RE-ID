"""
Player re-identification via appearance matching.
Maintains a gallery of player feature embeddings and resolves
new ByteTrack IDs against lost tracks using cosine similarity.

Includes four layers of gallery protection (Confident Track Storage):
  1. Team constraint     — never match across teams
  2. Occlusion detection — skip gallery update when player overlaps another
  3. Crop quality gate   — only update gallery with high-conf, clean crops
  4. Gallery freeze      — freeze embeddings when players are too close
  5. Dirty/clean loss    — collision-lost IDs need higher similarity to re-match

Flow per frame:
  1. Detect occlusions + proximity between all players in frame
  2. Known track IDs → update gallery ONLY if crop is clean (not occluded)
  3. New track IDs → compare against lost tracks' gallery (same team only)
     - Clean-lost IDs (camera pan) → normal threshold
     - Dirty-lost IDs (collision)  → higher threshold to prevent stolen IDs
  4. Missing IDs → mark as lost, tag HOW they were lost
"""
import numpy as np
from scipy.spatial.distance import cosine

import config
from feature_extractor import FeatureExtractor
from team_classifier import TeamClassifier


def _bbox_iou(a, b):
    """Compute IoU between two bboxes [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def _bbox_center_dist(a, b):
    """Euclidean distance between bbox centers."""
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    return np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)


class ReIDMatcher:
    """Manages player identity with confident track storage."""

    # Quality thresholds
    MIN_CONF_FOR_GALLERY = 0.65
    MAX_IOU_FOR_GALLERY = 0.15
    MIN_PROXIMITY = 60
    MIN_CROP_HEIGHT = 40
    MIN_ASPECT_RATIO = 1.0
    DIRTY_LOSS_PENALTY = 0.15      # extra similarity needed for collision-lost IDs
    EDGE_MARGIN = 80               # pixels from frame edge to count as camera-pan loss

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.team_classifier = TeamClassifier()
        self.gallery = {}          # {consistent_id: feature_vector}
        self.team_labels = {}      # {consistent_id: team_label (0 or 1)}
        self.id_map = {}           # {bytetrack_id: consistent_id}
        self.active_ids = set()    # consistent IDs seen in current frame
        self.lost_ids = set()      # consistent IDs not seen recently
        self.next_id = 1           # counter for new consistent IDs
        self.lost_frame_count = {} # {consistent_id: frames_since_lost}
        self.max_lost_frames = 150
        self.lost_dirty = {}       # {consistent_id: True if lost during collision}
        self.last_bbox = {}        # {consistent_id: last known [x1,y1,x2,y2]}

    def fit_teams(self, all_crops: list[np.ndarray]):
        """Fit team classifier on a batch of crops (from first N frames)."""
        self.team_classifier.fit(all_crops)

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        return 1.0 - cosine(a, b)

    def _was_near_edge(self, cid) -> bool:
        """Was player near frame edge when lost? (camera pan = clean loss)."""
        if cid not in self.last_bbox:
            return False
        bbox = self.last_bbox[cid]
        # Near left or right edge → camera pan
        return bbox[0] < self.EDGE_MARGIN or bbox[2] > 1280 - self.EDGE_MARGIN

    def _find_best_match(self, feat: np.ndarray, team: int) -> tuple[int | None, float]:
        """Compare feature against lost gallery entries of the SAME team.
        Dirty-lost IDs (collision) need higher similarity to match.
        """
        best_id = None
        best_sim = 0.0

        for cid in self.lost_ids:
            if cid not in self.gallery:
                continue
            # Hard constraint: skip if different team
            if team != -1 and self.team_labels.get(cid, -1) != -1:
                if self.team_labels[cid] != team:
                    continue

            # Dirty-lost = collision, need higher bar to prevent stolen IDs
            if self.lost_dirty.get(cid, False):
                thresh = config.REID_SIMILARITY_THRESH + self.DIRTY_LOSS_PENALTY
            else:
                thresh = config.REID_SIMILARITY_THRESH

            sim = self._cosine_sim(feat, self.gallery[cid])
            if sim > best_sim and sim >= thresh:
                best_sim = sim
                best_id = cid

        if best_id is not None:
            return best_id, best_sim
        return None, best_sim

    def _update_gallery(self, cid: int, feat: np.ndarray):
        """Update gallery entry with exponential moving average."""
        if cid in self.gallery:
            alpha = config.GALLERY_EMA_ALPHA
            self.gallery[cid] = alpha * self.gallery[cid] + (1 - alpha) * feat
            norm = np.linalg.norm(self.gallery[cid])
            if norm > 0:
                self.gallery[cid] /= norm
        else:
            self.gallery[cid] = feat.copy()

    def _is_crop_clean(self, player, all_players) -> bool:
        """Check if this player's crop is clean enough to update gallery."""
        bbox = player.bbox
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]

        if h < self.MIN_CROP_HEIGHT:
            return False
        if w > 0 and (h / w) < self.MIN_ASPECT_RATIO:
            return False
        if player.conf < self.MIN_CONF_FOR_GALLERY:
            return False

        for other in all_players:
            if other.track_id == player.track_id:
                continue
            iou = _bbox_iou(bbox, other.bbox)
            if iou > self.MAX_IOU_FOR_GALLERY:
                return False

        for other in all_players:
            if other.track_id == player.track_id:
                continue
            dist = _bbox_center_dist(bbox, other.bbox)
            if dist < self.MIN_PROXIMITY:
                return False

        return True

    def process_frame(self, players: list, frame_idx: int) -> dict:
        """Process one frame of tracked players."""
        current_bt_ids = set()
        frame_mapping = {}

        # Pre-compute which players have clean crops
        clean_flags = {}
        for player in players:
            clean_flags[player.track_id] = self._is_crop_clean(player, players)

        for player in players:
            bt_id = player.track_id
            current_bt_ids.add(bt_id)
            team = self.team_classifier.predict(player.crop)
            is_clean = clean_flags[bt_id]

            if bt_id in self.id_map:
                # Known track
                cid = self.id_map[bt_id]

                # ONLY update gallery if crop is clean
                if is_clean:
                    feat = self.extractor.extract(player.crop)["combined"]
                    self._update_gallery(cid, feat)

                if team != -1:
                    self.team_labels[cid] = team
                frame_mapping[bt_id] = cid

                # Always update last known position
                self.last_bbox[cid] = player.bbox.copy()

                if cid in self.lost_ids:
                    self.lost_ids.discard(cid)
                    if cid in self.lost_frame_count:
                        del self.lost_frame_count[cid]
                    if cid in self.lost_dirty:
                        del self.lost_dirty[cid]
            else:
                # New ByteTrack ID → try to match against lost (same team only)
                feat = self.extractor.extract(player.crop)["combined"]
                match_id, sim = self._find_best_match(feat, team)

                if match_id is not None:
                    self.id_map[bt_id] = match_id
                    if is_clean:
                        self._update_gallery(match_id, feat)
                    self.lost_ids.discard(match_id)
                    if match_id in self.lost_frame_count:
                        del self.lost_frame_count[match_id]
                    if match_id in self.lost_dirty:
                        del self.lost_dirty[match_id]
                    self.last_bbox[match_id] = player.bbox.copy()
                    frame_mapping[bt_id] = match_id
                else:
                    # Genuinely new player
                    cid = self.next_id
                    self.next_id += 1
                    self.id_map[bt_id] = cid
                    self.gallery[cid] = feat.copy()
                    if team != -1:
                        self.team_labels[cid] = team
                    self.last_bbox[cid] = player.bbox.copy()
                    frame_mapping[bt_id] = cid

        # Update lost tracks — tag HOW they were lost
        prev_active = self.active_ids.copy()
        self.active_ids = {self.id_map[bt] for bt in current_bt_ids if bt in self.id_map}

        newly_lost = prev_active - self.active_ids
        for cid in newly_lost:
            self.lost_ids.add(cid)
            self.lost_frame_count[cid] = 0
            # Tag: near edge = clean loss (camera pan), otherwise = dirty (collision)
            self.lost_dirty[cid] = not self._was_near_edge(cid)

        stale = []
        for cid in self.lost_ids:
            self.lost_frame_count[cid] = self.lost_frame_count.get(cid, 0) + 1
            if self.lost_frame_count[cid] > self.max_lost_frames:
                stale.append(cid)
        for cid in stale:
            self.lost_ids.discard(cid)
            del self.lost_frame_count[cid]
            if cid in self.lost_dirty:
                del self.lost_dirty[cid]

        return frame_mapping


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    from tracker import Tracker

    tracker = Tracker()
    matcher = ReIDMatcher()

    print("[Test] Running tracker...")
    frames = tracker.track_video()

    print("[Test] Fitting team classifier on first 5 frames...")
    init_crops = []
    for fr in frames[:5]:
        for p in fr.players:
            init_crops.append(p.crop)
    matcher.fit_teams(init_crops)

    print("[Test] Running re-ID matching...")
    all_mappings = []
    for fr in frames:
        mapping = matcher.process_frame(fr.players, fr.frame_idx)
        all_mappings.append(mapping)

        if fr.frame_idx % 50 == 0:
            active = len(matcher.active_ids)
            lost = len(matcher.lost_ids)
            print(f"  [Frame {fr.frame_idx}] active={active}, lost={lost}, "
                  f"total_consistent_ids={matcher.next_id - 1}")

    all_consistent = set()
    for m in all_mappings:
        all_consistent.update(m.values())

    print(f"\n=== Re-ID Summary ===")
    print(f"Consistent IDs after re-ID: {len(all_consistent)}")
    print(f"Consistent IDs: {sorted(all_consistent)}")

    t0 = [cid for cid in all_consistent if matcher.team_labels.get(cid) == 0]
    t1 = [cid for cid in all_consistent if matcher.team_labels.get(cid) == 1]
    tn = [cid for cid in all_consistent if matcher.team_labels.get(cid, -1) == -1]
    print(f"Team 0: {len(t0)} players {sorted(t0)}")
    print(f"Team 1: {len(t1)} players {sorted(t1)}")
    print(f"Unclassified: {len(tn)} players {sorted(tn)}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "reid_confident_storage.txt"
    with open(out_path, "w") as f:
        f.write(f"Consistent IDs after re-ID: {len(all_consistent)}\n")
        f.write(f"Team 0: {len(t0)} players {sorted(t0)}\n")
        f.write(f"Team 1: {len(t1)} players {sorted(t1)}\n")
        f.write(f"Unclassified: {len(tn)} players {sorted(tn)}\n")
    print(f"[Saved] {out_path}")
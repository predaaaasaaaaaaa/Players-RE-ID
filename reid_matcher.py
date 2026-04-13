"""
Player re-identification via appearance matching.
Maintains a gallery of player feature embeddings and resolves
new ByteTrack IDs against lost tracks using cosine similarity.

Flow per frame:
  1. Known track IDs → update gallery with EMA
  2. New track IDs → compare against lost tracks' gallery
     - Match found → remap to old consistent ID
     - No match    → assign fresh consistent ID
  3. Missing IDs   → mark as lost (available for future matching)
"""
import numpy as np
from scipy.spatial.distance import cosine

import config
from feature_extractor import FeatureExtractor


class ReIDMatcher:
    """Manages player identity across track fragmentations."""

    def __init__(self):
        self.extractor = FeatureExtractor()
        self.gallery = {}          # {consistent_id: feature_vector}
        self.id_map = {}           # {bytetrack_id: consistent_id}
        self.active_ids = set()    # consistent IDs seen in current frame
        self.lost_ids = set()      # consistent IDs not seen recently
        self.next_id = 1           # counter for new consistent IDs
        self.lost_frame_count = {} # {consistent_id: frames_since_lost}
        self.max_lost_frames = 150 # remove from lost pool after this many frames

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        return 1.0 - cosine(a, b)

    def _find_best_match(self, feat: np.ndarray) -> tuple[int | None, float]:
        """Compare feature against all lost gallery entries.
        Returns (matched_consistent_id, similarity) or (None, 0.0).
        """
        best_id = None
        best_sim = 0.0

        for cid in self.lost_ids:
            if cid not in self.gallery:
                continue
            sim = self._cosine_sim(feat, self.gallery[cid])
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        if best_sim >= config.REID_SIMILARITY_THRESH:
            return best_id, best_sim
        return None, best_sim

    def _update_gallery(self, cid: int, feat: np.ndarray):
        """Update gallery entry with exponential moving average."""
        if cid in self.gallery:
            alpha = config.GALLERY_EMA_ALPHA
            self.gallery[cid] = alpha * self.gallery[cid] + (1 - alpha) * feat
            # Re-normalize
            norm = np.linalg.norm(self.gallery[cid])
            if norm > 0:
                self.gallery[cid] /= norm
        else:
            self.gallery[cid] = feat.copy()

    def process_frame(self, players: list, frame_idx: int) -> dict:
        """Process one frame of tracked players.

        Args:
            players: list of TrackedPlayer from tracker.py
            frame_idx: current frame index

        Returns:
            dict mapping {bytetrack_id: consistent_id} for this frame
        """
        current_bt_ids = set()
        frame_mapping = {}

        for player in players:
            bt_id = player.track_id
            current_bt_ids.add(bt_id)

            if bt_id in self.id_map:
                # Known track → update gallery
                cid = self.id_map[bt_id]
                feat = self.extractor.extract(player.crop)["combined"]
                self._update_gallery(cid, feat)
                frame_mapping[bt_id] = cid

                # If this was lost, recover it
                if cid in self.lost_ids:
                    self.lost_ids.discard(cid)
                    if cid in self.lost_frame_count:
                        del self.lost_frame_count[cid]
            else:
                # New ByteTrack ID → try to match against lost
                feat = self.extractor.extract(player.crop)["combined"]
                match_id, sim = self._find_best_match(feat)

                if match_id is not None:
                    # Re-identified! Map new BT ID to old consistent ID
                    self.id_map[bt_id] = match_id
                    self._update_gallery(match_id, feat)
                    self.lost_ids.discard(match_id)
                    if match_id in self.lost_frame_count:
                        del self.lost_frame_count[match_id]
                    frame_mapping[bt_id] = match_id
                else:
                    # Genuinely new player
                    cid = self.next_id
                    self.next_id += 1
                    self.id_map[bt_id] = cid
                    self.gallery[cid] = feat.copy()
                    frame_mapping[bt_id] = cid

        # Update lost tracks
        prev_active = self.active_ids.copy()
        self.active_ids = {self.id_map[bt] for bt in current_bt_ids if bt in self.id_map}

        newly_lost = prev_active - self.active_ids
        for cid in newly_lost:
            self.lost_ids.add(cid)
            self.lost_frame_count[cid] = 0

        # Age lost tracks and remove stale ones
        stale = []
        for cid in self.lost_ids:
            self.lost_frame_count[cid] = self.lost_frame_count.get(cid, 0) + 1
            if self.lost_frame_count[cid] > self.max_lost_frames:
                stale.append(cid)
        for cid in stale:
            self.lost_ids.discard(cid)
            del self.lost_frame_count[cid]

        return frame_mapping


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    from tracker import Tracker

    tracker = Tracker()
    matcher = ReIDMatcher()

    print("[Test] Running tracker...")
    frames = tracker.track_video()

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

    # Summary
    all_consistent = set()
    for m in all_mappings:
        all_consistent.update(m.values())

    print(f"\n=== Re-ID Summary ===")
    print(f"ByteTrack IDs: 96")
    print(f"Consistent IDs after re-ID: {len(all_consistent)}")
    print(f"Consistent IDs: {sorted(all_consistent)}")

    # Save summary
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "step4_reid_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"ByteTrack IDs: 96\n")
        f.write(f"Consistent IDs after re-ID: {len(all_consistent)}\n")
        f.write(f"Consistent IDs: {sorted(all_consistent)}\n")
    print(f"[Saved] {out_path}")
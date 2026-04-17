"""
Diagnostic: measure the actual similarity distribution of the pipeline
on THIS video. Samples same-player pairs (same ByteTrack ID, different
frames) vs different-player pairs (different IDs, same frame) and prints
the separation gap.

Use this to pick REID_SIMILARITY_THRESH empirically instead of guessing.
"""
import random
import numpy as np

from tracker import Tracker
from feature_extractor import FeatureExtractor


def main():
    random.seed(42)
    np.random.seed(42)

    print("[Diag] Running tracker...")
    tracker = Tracker()
    frames = tracker.track_video()

    extractor = FeatureExtractor()

    # Build: track_id -> list of (frame_idx, crop)
    track_crops = {}
    for fr in frames:
        for p in fr.players:
            track_crops.setdefault(p.track_id, []).append((fr.frame_idx, p.crop))

    # Keep only tracks that appear in >=5 frames (stable ones)
    stable_tracks = {tid: crops for tid, crops in track_crops.items() if len(crops) >= 5}
    print(f"[Diag] Stable tracks (>=5 frames): {len(stable_tracks)}")

    # Extract features once per (track_id, frame) — cache
    print("[Diag] Extracting features...")
    feats = {}  # (tid, fidx) -> feat_dict
    for tid, crops in stable_tracks.items():
        # Subsample to max 20 frames per track to save time
        sample = random.sample(crops, min(20, len(crops)))
        for fidx, crop in sample:
            feats[(tid, fidx)] = extractor.extract(crop)

    print(f"[Diag] Extracted {len(feats)} feature vectors")

    # Same-player pairs: same tid, different frames
    same_sims = []
    for tid in stable_tracks:
        tid_keys = [k for k in feats if k[0] == tid]
        if len(tid_keys) < 2:
            continue
        # All unique pairs within this track
        for i in range(len(tid_keys)):
            for j in range(i + 1, len(tid_keys)):
                sim = extractor.similarity(feats[tid_keys[i]], feats[tid_keys[j]])
                same_sims.append(sim)

    # Different-player pairs: different tids
    diff_sims = []
    all_keys = list(feats.keys())
    # Sample up to 2000 random different-pair comparisons
    n_samples = min(2000, len(all_keys) * (len(all_keys) - 1) // 2)
    for _ in range(n_samples):
        a, b = random.sample(all_keys, 2)
        if a[0] == b[0]:
            continue
        sim = extractor.similarity(feats[a], feats[b])
        diff_sims.append(sim)

    same_sims = np.array(same_sims)
    diff_sims = np.array(diff_sims)

    print(f"\n=== SAME-player pairs (n={len(same_sims)}) ===")
    print(f"  mean={same_sims.mean():.4f}  std={same_sims.std():.4f}")
    print(f"  min ={same_sims.min():.4f}   max ={same_sims.max():.4f}")
    print(f"  percentiles: 5%={np.percentile(same_sims, 5):.4f}  "
          f"25%={np.percentile(same_sims, 25):.4f}  "
          f"50%={np.percentile(same_sims, 50):.4f}  "
          f"75%={np.percentile(same_sims, 75):.4f}")

    print(f"\n=== DIFFERENT-player pairs (n={len(diff_sims)}) ===")
    print(f"  mean={diff_sims.mean():.4f}  std={diff_sims.std():.4f}")
    print(f"  min ={diff_sims.min():.4f}   max ={diff_sims.max():.4f}")
    print(f"  percentiles: 75%={np.percentile(diff_sims, 75):.4f}  "
          f"90%={np.percentile(diff_sims, 90):.4f}  "
          f"95%={np.percentile(diff_sims, 95):.4f}  "
          f"99%={np.percentile(diff_sims, 99):.4f}")

    # Recommended threshold = midpoint between same-5% and diff-95%
    lo = np.percentile(same_sims, 5)
    hi = np.percentile(diff_sims, 95)
    gap = lo - hi
    recommended = (lo + hi) / 2

    print(f"\n=== SEPARATION ===")
    print(f"  same-5%     = {lo:.4f}  (95% of true matches above this)")
    print(f"  diff-95%    = {hi:.4f}  (95% of non-matches below this)")
    print(f"  gap         = {gap:+.4f}  ({'GOOD' if gap > 0 else 'BAD — distributions overlap'})")
    print(f"  recommended REID_SIMILARITY_THRESH = {recommended:.2f}")


if __name__ == "__main__":
    main()
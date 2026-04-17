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

    # Compute three similarity variants per pair: hsv-only, deep-only, combined
    def sim_hsv(a, b):
        return float(np.dot(a["hsv"], b["hsv"]))
    def sim_deep(a, b):
        return float(np.dot(a["deep"], b["deep"]))
    def sim_comb(a, b):
        return extractor.similarity(a, b)

    same_hsv, same_deep, same_comb = [], [], []
    for tid in stable_tracks:
        tid_keys = [k for k in feats if k[0] == tid]
        if len(tid_keys) < 2:
            continue
        for i in range(len(tid_keys)):
            for j in range(i + 1, len(tid_keys)):
                a, b = feats[tid_keys[i]], feats[tid_keys[j]]
                same_hsv.append(sim_hsv(a, b))
                same_deep.append(sim_deep(a, b))
                same_comb.append(sim_comb(a, b))

    diff_hsv, diff_deep, diff_comb = [], [], []
    all_keys = list(feats.keys())
    n_samples = min(2000, len(all_keys) * (len(all_keys) - 1) // 2)
    for _ in range(n_samples):
        a_k, b_k = random.sample(all_keys, 2)
        if a_k[0] == b_k[0]:
            continue
        a, b = feats[a_k], feats[b_k]
        diff_hsv.append(sim_hsv(a, b))
        diff_deep.append(sim_deep(a, b))
        diff_comb.append(sim_comb(a, b))

    def report(name, same, diff):
        same, diff = np.array(same), np.array(diff)
        lo = np.percentile(same, 5)
        hi = np.percentile(diff, 95)
        gap = lo - hi
        print(f"\n=== {name} ===")
        print(f"  same: mean={same.mean():.4f}  5%={lo:.4f}  50%={np.percentile(same,50):.4f}")
        print(f"  diff: mean={diff.mean():.4f}  95%={hi:.4f}  50%={np.percentile(diff,50):.4f}")
        print(f"  gap = {gap:+.4f}  recommended thresh = {(lo+hi)/2:.2f}")

    report("HSV only",      same_hsv,  diff_hsv)
    report("Deep only",     same_deep, diff_deep)
    report("Combined 0.4/0.6", same_comb, diff_comb)

if __name__ == "__main__":
    main()
"""
Download SoccerNet ReID dataset and filter to ~30K high-quality samples.

Filtering strategy:
  1. Only keep 'Player' class (skip referees — we filter them at detection)
  2. Only keep crops with height >= 50px (skip tiny distant players)
  3. Only keep identities with >= 3 samples (need variety to learn from)
  4. Cap at 10 samples per identity (prevent overfitting on one player)
  5. Balanced sampling across games (diversity > volume)
"""
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── Config ─────────────────────────────────────────────
SOCCERNET_DIR = Path("data/soccernet_reid")
FILTERED_DIR = Path("data/rf")
TARGET_SAMPLES = 30000
MIN_HEIGHT = 50
MIN_SAMPLES_PER_ID = 3
MAX_SAMPLES_PER_ID = 10
SEED = 42


def download_soccernet():
    """Download SoccerNet ReID train split."""
    from SoccerNet.Downloader import SoccerNetDownloader as SNdl

    SOCCERNET_DIR.mkdir(parents=True, exist_ok=True)
    downloader = SNdl(LocalDirectory=str(SOCCERNET_DIR))
    downloader.downloadDataTask(task="reid", split=["train", "valid"])
    print(f"[Download] SoccerNet ReID saved to {SOCCERNET_DIR}")


def parse_filename(filename: str) -> dict:
    """Parse SoccerNet ReID filename convention.
    Format: <bbox_idx>-<action_idx>-<person_uid>-<frame_idx>-<class>-<ID>-<UAI>-<height>x<width>.png
    """
    try:
        parts = filename.replace(".png", "").split("-")
        if len(parts) < 8:
            return None
        # Height x Width is the last part
        hw = parts[-1].split("x")
        if len(hw) != 2:
            return None
        return {
            "bbox_idx": parts[0],
            "action_idx": parts[1],
            "person_uid": parts[2],
            "frame_idx": parts[3],
            "class": parts[4],
            "id": parts[5],
            "uai": parts[6],
            "height": int(hw[0]),
            "width": int(hw[1]),
            "filename": filename,
        }
    except (ValueError, IndexError):
        return None


def collect_and_filter():
    """Walk SoccerNet ReID directory, filter and organize samples."""
    print("[Filter] Scanning SoccerNet ReID files...")

    # Collect all image paths
    all_images = []
    for root, dirs, files in os.walk(SOCCERNET_DIR):
        for f in files:
            if f.endswith(".png"):
                all_images.append(Path(root) / f)

    print(f"[Filter] Found {len(all_images)} total images")

    # Parse and filter
    by_identity = defaultdict(list)
    skipped_class = 0
    skipped_small = 0
    skipped_parse = 0

    for img_path in all_images:
        info = parse_filename(img_path.name)
        if info is None:
            skipped_parse += 1
            continue

        # Filter 1: Only players
        if "Player" not in info["class"]:
            skipped_class += 1
            continue

        # Filter 2: Min height
        if info["height"] < MIN_HEIGHT:
            skipped_small += 1
            continue

        # Group by unique identity (person_uid is unique across dataset)
        identity_key = info["person_uid"]
        by_identity[identity_key].append({
            "path": img_path,
            "info": info,
        })

    print(f"[Filter] After filtering:")
    print(f"  Skipped (not player): {skipped_class}")
    print(f"  Skipped (too small):  {skipped_small}")
    print(f"  Skipped (parse error):{skipped_parse}")
    print(f"  Unique identities:    {len(by_identity)}")
    total_kept = sum(len(v) for v in by_identity.values())
    print(f"  Total samples:        {total_kept}")

    # Filter 3: Min samples per identity
    filtered_ids = {k: v for k, v in by_identity.items() if len(v) >= MIN_SAMPLES_PER_ID}
    print(f"[Filter] After min {MIN_SAMPLES_PER_ID} samples/ID: {len(filtered_ids)} identities")

    # Filter 4: Cap samples per identity + random sample to target
    random.seed(SEED)
    selected = []
    for identity, samples in filtered_ids.items():
        if len(samples) > MAX_SAMPLES_PER_ID:
            samples = random.sample(samples, MAX_SAMPLES_PER_ID)
        selected.extend([(identity, s) for s in samples])

    print(f"[Filter] After capping {MAX_SAMPLES_PER_ID}/ID: {len(selected)} samples")

    # Organize into torchreid format: filtered_dir/<id>/<image>.png
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    id_counter = 0
    id_map = {}

    for identity, sample in selected:
        if identity not in id_map:
            id_map[identity] = id_counter
            id_counter += 1

        numeric_id = id_map[identity]
        id_dir = FILTERED_DIR / f"{numeric_id:05d}"
        id_dir.mkdir(exist_ok=True)

        dst = id_dir / f"{sample['info']['bbox_idx']}.png"
        if not dst.exists():
            # Windows long path fix
            src = "\\\\?\\" + str(sample["path"].resolve())
            dst_str = "\\\\?\\" + str(dst.resolve())
            shutil.copy2(src, dst_str)

    # Count results
    total_files = sum(1 for _ in FILTERED_DIR.rglob("*.png"))
    total_ids = len(list(FILTERED_DIR.iterdir()))
    print(f"\n=== Final Dataset ===")
    print(f"  Directory: {FILTERED_DIR}")
    print(f"  Identities: {total_ids}")
    print(f"  Total images: {total_files}")
    print(f"  Avg samples/ID: {total_files / max(total_ids, 1):.1f}")

    return total_ids, total_files


if __name__ == "__main__":
    # Step 1: Download
    if not SOCCERNET_DIR.exists() or not any(SOCCERNET_DIR.rglob("*.png")):
        print("=== Downloading SoccerNet ReID ===")
        download_soccernet()
    else:
        print(f"[Skip] SoccerNet data already at {SOCCERNET_DIR}")

    # Step 2: Filter
    print("\n=== Filtering dataset ===")
    n_ids, n_imgs = collect_and_filter()
    print(f"\nDone! Ready to train OSNet on {n_imgs} images / {n_ids} identities")
"""
Fine-tune OSNet on filtered SoccerNet ReID dataset.

Research-backed training recipe (SportsReID, SoccerNet Challenge 2022-2024):
  - OSNet x0_25 backbone, pretrained on ImageNet
  - Softmax + label smoothing (centroid-like effect)
  - Hierarchical augmentation: random_flip + color_jitter + random_erase
  - 50 epochs, batch 32, lr 0.0003, cosine annealing
  - Color jitter is critical for soccer (shadow stripes on pitch)
  - 10% of SoccerNet data is enough to beat baseline OSNet
"""
import os
import sys
import shutil
from pathlib import Path

# ── Config ─────────────────────────────────────────────
FILTERED_DIR = Path("data/rf")
TRAIN_DIR = Path("data/reid_train")
QUERY_DIR = Path("data/reid_query")
GALLERY_DIR = Path("data/reid_gallery")
SAVE_DIR = Path("models/osnet_soccer")

EPOCHS = 50
BATCH_SIZE = 32
LR = 0.0003
MODEL_NAME = "osnet_x0_25"

# Train/val split: 85% train, 15% val (split into query + gallery)
TRAIN_RATIO = 0.85


def prepare_torchreid_structure():
    """Convert filtered data into torchreid train/query/gallery format.
    
    torchreid expects:
      train/  -> <pid>_<camid>_<imgidx>.png
      query/  -> same format
      gallery/ -> same format
    
    We fake camera IDs since all are broadcast crops.
    """
    import random
    random.seed(42)

    # Clean old splits
    for d in [TRAIN_DIR, QUERY_DIR, GALLERY_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    if not FILTERED_DIR.exists():
        print(f"[ERROR] Filtered data not found at {FILTERED_DIR}")
        print("Run download_reid_data.py first!")
        sys.exit(1)

    # Collect all identities
    id_dirs = sorted([d for d in FILTERED_DIR.iterdir() if d.is_dir()])
    print(f"[Prep] Found {len(id_dirs)} identities")

    total_train = 0
    total_val = 0

    for id_dir in id_dirs:
        pid = id_dir.name  # numeric ID like "00042"
        images = sorted(list(id_dir.glob("*.png")))
        if len(images) < 2:
            continue

        random.shuffle(images)
        split_idx = max(1, int(len(images) * TRAIN_RATIO))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Write train images
        for i, img_path in enumerate(train_imgs):
            # torchreid format: <pid>_c<camid>s1_<imgidx>.jpg
            dst_name = f"{pid}_c1s1_{i:04d}.png"
            shutil.copy2(img_path, TRAIN_DIR / dst_name)
            total_train += 1

        # Write val images — half as query, half as gallery
        for i, img_path in enumerate(val_imgs):
            if i % 2 == 0:
                dst_name = f"{pid}_c1s1_{i:04d}.png"
                shutil.copy2(img_path, QUERY_DIR / dst_name)
            else:
                dst_name = f"{pid}_c2s1_{i:04d}.png"
                shutil.copy2(img_path, GALLERY_DIR / dst_name)
            total_val += 1

    print(f"[Prep] Train: {total_train} images")
    print(f"[Prep] Val: {total_val} images (query + gallery)")
    print(f"[Prep] Directories: {TRAIN_DIR}, {QUERY_DIR}, {GALLERY_DIR}")


def register_dataset():
    """Register our custom dataset with torchreid."""
    import torchreid
    from torchreid.reid.data import ImageDataset

    class SoccerNetReID(ImageDataset):
        dataset_dir = ""

        def __init__(self, root="", **kwargs):
            self.train_dir = str(TRAIN_DIR)
            self.query_dir = str(QUERY_DIR)
            self.gallery_dir = str(GALLERY_DIR)

            train = self._process_dir(self.train_dir, relabel=True)
            query = self._process_dir(self.query_dir, relabel=False)
            gallery = self._process_dir(self.gallery_dir, relabel=False)

            super().__init__(train, query, gallery, **kwargs)

        def _process_dir(self, dir_path, relabel=False):
            """Parse filenames: <pid>_c<camid>s1_<imgidx>.png"""
            import glob
            img_paths = sorted(glob.glob(os.path.join(dir_path, "*.png")))
            if not img_paths:
                return []

            pid_container = set()
            for img_path in img_paths:
                fname = os.path.basename(img_path)
                pid = fname.split("_")[0]
                pid_container.add(pid)

            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

            data = []
            for img_path in img_paths:
                fname = os.path.basename(img_path)
                parts = fname.split("_")
                pid = parts[0]
                # Extract camid from c<X>s1
                camid_str = parts[1]  # e.g., "c1s1"
                camid = int(camid_str[1])  # extract digit after 'c'

                if relabel:
                    pid = pid2label[pid]
                else:
                    pid = int(pid)

                data.append((img_path, pid, camid))

            return data

    # Register
    torchreid.data.register_image_dataset("soccernet_reid", SoccerNetReID)
    print("[Dataset] Registered 'soccernet_reid' with torchreid")


def train():
    """Fine-tune OSNet on SoccerNet ReID."""
    import torchreid

    register_dataset()

    # Build data manager with soccer-optimized augmentations
    datamanager = torchreid.data.ImageDataManager(
        root="",
        sources="soccernet_reid",
        targets="soccernet_reid",
        height=256,
        width=128,
        batch_size_train=BATCH_SIZE,
        batch_size_test=100,
        transforms=["random_flip", "color_jitter", "random_erase"],
    )

    num_classes = datamanager.num_train_pids
    print(f"[Train] {num_classes} training identities")

    # Build OSNet model
    model = torchreid.models.build_model(
        name=MODEL_NAME,
        num_classes=num_classes,
        loss="softmax",
        pretrained=True,
    )

    model = model.cuda() if __import__("torch").cuda.is_available() else model

    # Optimizer: Adam with cosine LR
    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=LR,
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="cosine",
        max_epoch=EPOCHS,
    )

    # Engine: softmax with label smoothing
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
    )

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    engine.run(
        save_dir=str(SAVE_DIR),
        max_epoch=EPOCHS,
        eval_freq=10,
        print_freq=20,
        test_only=False,
    )

    print(f"\n[Done] Model saved to {SAVE_DIR}")
    print(f"[Done] Best model: {SAVE_DIR}/model.pth.tar-{EPOCHS}")


if __name__ == "__main__":
    # Step 1: Prepare directory structure
    if not TRAIN_DIR.exists() or len(list(TRAIN_DIR.glob("*.png"))) == 0:
        print("=== Preparing torchreid dataset structure ===")
        prepare_torchreid_structure()
    else:
        print(f"[Skip] Train data already at {TRAIN_DIR}")

    # Step 2: Train
    print("\n=== Starting OSNet fine-tuning ===")
    train()
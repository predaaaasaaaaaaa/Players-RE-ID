"""
Appearance feature extraction for player re-identification.
Combines two signals:
  1. HSV color histogram  — captures jersey color (fast, robust)
  2. Deep embedding        — OSNet pretrained on MSMT17 person re-ID dataset
Both are L2-normalized and concatenated into a single feature vector.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchreid


class FeatureExtractor:
    """Extract appearance features from player crops using OSNet."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_deep_model()
        self._init_transform()
        print(f"[FeatureExtractor] OSNet re-ID on {self.device}")

    def _init_deep_model(self):
        """Load OSNet fine-tuned on SoccerNet ReID data."""
        import torch
        from pathlib import Path
        
        self.model = torchreid.models.build_model(
            name="osnet_x0_25",
            num_classes=1000,  # placeholder, we discard classifier
            loss="softmax",
            pretrained=False,  # we'll load our own weights
        )
        
        # Load our fine-tuned weights
        checkpoint_path = Path("models/osnet_soccer/model/model.pth.tar-50")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Strip classifier keys (num_classes mismatch)
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[FeatureExtractor] Loaded fine-tuned OSNet from {checkpoint_path}")
        else:
            print(f"[FeatureExtractor] WARNING: {checkpoint_path} not found, using ImageNet pretrained")
        
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.classifier = torch.nn.Identity()

    def _init_transform(self):
        """Preprocessing for OSNet input (256x128, standard re-ID size)."""
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract_hsv(self, crop: np.ndarray) -> np.ndarray:
        """Compute normalized HSV color histogram from a BGR crop."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        upper = hsv[:int(h * 0.6), :, :]
        hist_h = cv2.calcHist([upper], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([upper], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([upper], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist  # 78-dim

    @torch.no_grad()
    def extract_deep(self, crop: np.ndarray) -> np.ndarray:
        """Extract OSNet re-ID embedding from a BGR crop."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat  # 512-dim (OSNet)

    def extract(self, crop: np.ndarray) -> dict:
        """Extract both features from a player crop."""
        hsv = self.extract_hsv(crop)
        deep = self.extract_deep(crop)
        combined = np.concatenate([hsv, deep])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return {"hsv": hsv, "deep": deep, "combined": combined}


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    from detector import Detector
    import config

    detector = Detector()
    extractor = FeatureExtractor()

    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()

    dets = detector.detect(frame)
    print(f"[Test] {len(dets)} players detected")

    for i, d in enumerate(dets[:3]):
        feats = extractor.extract(d.crop)
        print(f"  Player {i}: hsv={feats['hsv'].shape}, "
              f"deep={feats['deep'].shape}, "
              f"combined={feats['combined'].shape}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(dets):
        path = config.OUTPUT_DIR / f"crop_{i}.jpg"
        cv2.imwrite(str(path), d.crop)
    print(f"[Saved] {len(dets)} crops to {config.OUTPUT_DIR}")
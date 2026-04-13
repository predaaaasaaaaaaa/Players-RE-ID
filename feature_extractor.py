"""
Appearance feature extraction for player re-identification.
Combines two signals:
  1. HSV color histogram  — captures jersey color (fast, robust)
  2. Deep embedding        — MobileNetV2 pretrained features (appearance detail)
Both are L2-normalized and concatenated into a single feature vector.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class FeatureExtractor:
    """Extract appearance features from player crops."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_deep_model()
        self._init_transform()
        print(f"[FeatureExtractor] Using device: {self.device}")

    def _init_deep_model(self):
        """Load MobileNetV2 as feature extractor (no classification head)."""
        weights = MobileNet_V2_Weights.DEFAULT
        model = mobilenet_v2(weights=weights)
        # Remove classifier, keep feature backbone only
        model.classifier = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

    def _init_transform(self):
        """Preprocessing for MobileNetV2 input."""
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # standard re-ID input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def extract_hsv(self, crop: np.ndarray) -> np.ndarray:
        """Compute normalized HSV color histogram from a BGR crop."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Focus on upper body (top 60%) to capture jersey, not shorts/grass
        h, w = hsv.shape[:2]
        upper = hsv[:int(h * 0.6), :, :]
        # H: 30 bins, S: 32 bins, V: 16 bins
        hist_h = cv2.calcHist([upper], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([upper], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([upper], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        # L2 normalize
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist  # 78-dim

    @torch.no_grad()
    def extract_deep(self, crop: np.ndarray) -> np.ndarray:
        """Extract deep feature embedding from a BGR crop."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat  # 1280-dim

    def extract(self, crop: np.ndarray) -> dict:
        """Extract both features from a player crop.
        Returns dict with 'hsv', 'deep', and 'combined' keys.
        """
        hsv = self.extract_hsv(crop)
        deep = self.extract_deep(crop)
        combined = np.concatenate([hsv, deep])
        # Re-normalize combined
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

    for i, d in enumerate(dets[:3]):  # test first 3
        feats = extractor.extract(d.crop)
        print(f"  Player {i}: hsv={feats['hsv'].shape}, "
              f"deep={feats['deep'].shape}, "
              f"combined={feats['combined'].shape}")

    # Save crops for visual inspection
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(dets):
        path = config.OUTPUT_DIR / f"step3_crop_{i}.jpg"
        cv2.imwrite(str(path), d.crop)
    print(f"[Saved] {len(dets)} crops to {config.OUTPUT_DIR}")
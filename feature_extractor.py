"""
Appearance feature extraction for player re-identification.
Combines two signals:
  1. HSV color histogram  — captures jersey color (fast, robust)
  2. Deep embedding        — OSNet-AIN x1.0, MSMT17-pretrained with instance
                             normalization for cross-domain robustness

Each signal is L2-normalized independently. Similarity is computed as a
weighted sum of per-signal cosine similarities (not concatenated),
preventing the 512-dim deep vector from drowning out the 78-dim HSV.
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchreid


class FeatureExtractor:
    """Extract appearance features from player crops using OSNet-AIN."""

    # Fusion weights — deep carries more signal, HSV anchors jersey color
    W_HSV = 0.4
    W_DEEP = 0.6

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_deep_model()
        self._init_transform()

    def _init_deep_model(self):
        """Load OSNet-AIN x1.0 — instance-normalized, MSMT17-pretrained.
        IN layers normalize per-image, reducing lighting/exposure variance
        across camera cuts. 6.7M params vs 2.2M for x0_25."""
        self.model = torchreid.models.build_model(
            name="osnet_ain_x1_0",
            num_classes=1000,
            loss="softmax",
            pretrained=True,
        )
        self.model.eval()
        self.model = self.model.to(self.device)
        self.model.classifier = torch.nn.Identity()
        print(f"[FeatureExtractor] OSNet-AIN x1.0 (MSMT17 pretrained) on {self.device}")

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
        """Extract OSNet-AIN re-ID embedding from a BGR crop."""
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        feat = self.model(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat  # 512-dim

    def extract(self, crop: np.ndarray) -> dict:
        """Extract both features from a player crop. No concatenation."""
        hsv = self.extract_hsv(crop)
        deep = self.extract_deep(crop)
        return {"hsv": hsv, "deep": deep}

    @staticmethod
    def similarity(feat_a: dict, feat_b: dict) -> float:
        """Weighted cosine similarity between two feature dicts.
        Each component is already L2-normalized, so dot product = cosine.
        """
        cos_hsv = float(np.dot(feat_a["hsv"], feat_b["hsv"]))
        cos_deep = float(np.dot(feat_a["deep"], feat_b["deep"]))
        return FeatureExtractor.W_HSV * cos_hsv + FeatureExtractor.W_DEEP * cos_deep


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
        print(f"  Player {i}: hsv={feats['hsv'].shape}, deep={feats['deep'].shape}")

    if len(dets) >= 2:
        f0 = extractor.extract(dets[0].crop)
        f1 = extractor.extract(dets[1].crop)
        sim = extractor.similarity(f0, f1)
        print(f"  Sim(player0, player1) = {sim:.4f}")
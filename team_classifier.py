"""
Team classification via KMeans clustering on jersey color.
Splits players into two teams based on upper-body HSV histograms.
Used as a hard constraint in re-ID: never match across teams.
"""
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """Classify players into two teams based on jersey color."""

    def __init__(self):
        self.kmeans = None
        self.fitted = False

    def _extract_jersey_color(self, crop: np.ndarray) -> np.ndarray:
        """Extract HSV histogram from upper body (jersey region)."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        # Top 50% = jersey, avoids shorts and grass
        upper = hsv[:int(h * 0.5), :, :]
        hist_h = cv2.calcHist([upper], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([upper], [1], None, [16], [0, 256]).flatten()
        hist = np.concatenate([hist_h, hist_s])
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist  # 32-dim

    def fit(self, crops: list[np.ndarray]):
        """Fit KMeans on a batch of player crops to find two team clusters."""
        if len(crops) < 4:
            print("[TeamClassifier] Not enough crops to fit, need at least 4")
            return

        features = []
        valid_indices = []
        for i, crop in enumerate(crops):
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            features.append(self._extract_jersey_color(crop))
            valid_indices.append(i)

        if len(features) < 4:
            print("[TeamClassifier] Not enough valid crops after filtering")
            return

        X = np.array(features)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(X)
        self.fitted = True

        # Print cluster distribution
        labels = self.kmeans.labels_
        print(f"[TeamClassifier] Fitted on {len(crops)} crops: "
              f"Team 0={np.sum(labels == 0)}, Team 1={np.sum(labels == 1)}")

    def predict(self, crop: np.ndarray) -> int:
        """Predict team label (0 or 1) for a single player crop."""
        if not self.fitted:
            return -1
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return -1
        feat = self._extract_jersey_color(crop).reshape(1, -1)
        return int(self.kmeans.predict(feat)[0])


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    from detector import Detector
    import config

    detector = Detector()
    cap = cv2.VideoCapture(str(config.VIDEO_PATH))
    ret, frame = cap.read()
    cap.release()

    dets = detector.detect(frame)
    crops = [d.crop for d in dets]
    print(f"[Test] {len(crops)} player crops from frame 0")

    tc = TeamClassifier()
    tc.fit(crops)

    for i, d in enumerate(dets):
        team = tc.predict(d.crop)
        print(f"  Player {i}: team={team}, bbox={d.bbox}")

    # Save team-colored visualization
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEAM_COLORS = {0: (255, 100, 50), 1: (50, 100, 255), -1: (128, 128, 128)}
    vis = frame.copy()
    for d in dets:
        team = tc.predict(d.crop)
        x1, y1, x2, y2 = d.bbox
        color = TEAM_COLORS[team]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"T{team}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = str(config.OUTPUT_DIR / "step6_team_classification.jpg")
    cv2.imwrite(out_path, vis)
    print(f"[Saved] {out_path}")
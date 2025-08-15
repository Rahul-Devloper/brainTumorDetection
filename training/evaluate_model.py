import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === SETTINGS ===
TEST_DIR = "../data/binary_split/test"
MODEL_PATH = "../notebooks/api/model/brain_mri_model.h5"

THRESHOLD = 0.05   # Replace with your tuned value
IMG_SIZE = (28, 28)

# === 1. Load model ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded.")

# === 2. Helper: load images & labels ===


def load_data(folder):
    X, y = [], []
    for label_name, label_id in [("no_tumor", 0), ("tumor", 1)]:
        path = Path(folder) / label_name
        for img_path in path.glob("*"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)           # Resize
            img = img.astype("float32") / 255.0       # Normalize
            img = np.expand_dims(img, axis=-1)        # Shape: (28,28,1)
            X.append(img)
            y.append(label_id)
    return np.array(X), np.array(y)


# === 3. Load test data ===
X_test, y_test = load_data(TEST_DIR)
print(f"[INFO] Loaded {len(X_test)} test images.")

# === 4. Predict in batches to save RAM ===
probs = model.predict(X_test, batch_size=64, verbose=1).ravel()
best = None
print("\nThreshold  Prec    Recall  F1     FP   FN")
for t in np.linspace(0.05, 0.95, 19):
    preds_t = (probs >= t).astype(int)
    prec = precision_score(y_test, preds_t, zero_division=0)
    rec = recall_score(y_test, preds_t, zero_division=0)
    f1 = f1_score(y_test, preds_t, zero_division=0)
    fp = int(((y_test == 0) & (preds_t == 1)).sum())
    fn = int(((y_test == 1) & (preds_t == 0)).sum())
    print(f"{t:8.2f}  {prec:6.3f}  {recall_score(y_test, preds_t):6.3f}  {f1:6.3f}  {fp:4d} {fn:4d}")
    if best is None or f1 > best[0]:
        best = (f1, t, prec, rec)

print(f"\nBest F1 on TEST: F1={best[0]:.3f} at threshold={best[1]:.2f} "
      f"(prec={best[2]:.3f}, recall={best[3]:.3f})")

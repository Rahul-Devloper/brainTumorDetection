import os
import json
import datetime
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score
)

# === SETTINGS ===
TEST_DIR = "../data/binary_split/test"
MODEL_PATH = "../notebooks/api/model/brain_mri_model.h5"
THRESHOLD = 0.05             # your locked threshold
IMG_SIZE = (28, 28)
BATCH_SIZE = 64
OUT_DIR = "eval_final"     # where to save outputs

# === 0) Path sanity prints (optional but helpful) ===
print("[PATH] TEST_DIR  =", Path(TEST_DIR).resolve())
print("[PATH] MODEL     =", Path(MODEL_PATH).resolve())

# === 1) Load model ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded.")

# === 2) Helper: load images & labels ===
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

# === 3) Load test data ===
X_test, y_test = load_data(TEST_DIR)
print(f"[INFO] Loaded {len(X_test)} test images.")

# === 4) Predict in batches to save RAM ===
probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1).ravel()

# === 5) Threshold sweep (for reference) ===
best = None
print("\nThreshold  Prec    Recall  F1     FP   FN")
for t in np.linspace(0.05, 0.95, 19):
    preds_t = (probs >= t).astype(int)
    prec = precision_score(y_test, preds_t, zero_division=0)
    rec = recall_score(y_test, preds_t, zero_division=0)
    f1 = f1_score(y_test, preds_t, zero_division=0)
    fp = int(((y_test == 0) & (preds_t == 1)).sum())
    fn = int(((y_test == 1) & (preds_t == 0)).sum())
    print(f"{t:8.2f}  {prec:6.3f}  {rec:6.3f}  {f1:6.3f}  {fp:4d} {fn:4d}")
    if best is None or f1 > best[0]:
        best = (f1, t, prec, rec)

print(f"\nBest F1 on TEST: F1={best[0]:.3f} at threshold={best[1]:.2f} "
      f"(prec={best[2]:.3f}, recall={best[3]:.3f})")

# === 6) Final metrics at the LOCKED THRESHOLD ===
preds = (probs >= THRESHOLD).astype(int)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)
f1 = f1_score(y_test, preds, zero_division=0)
cm = confusion_matrix(y_test, preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Extra: specificity (TNR) and NPV for no_tumor
specificity = tn / (tn + fp) if (tn + fp) else float("nan")
npv = tn / (tn + fn) if (tn + fn) else float("nan")

# Curves (probability-based)
try:
    roc_auc = roc_auc_score(y_test, probs)
except ValueError:
    roc_auc = float("nan")
pr_auc = average_precision_score(y_test, probs)

print("\n=== Final Metrics at LOCKED threshold {:.2f} ===".format(THRESHOLD))
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"Specific.: {specificity:.4f}")
print(f"NPV      : {npv:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print("\nConfusion Matrix (rows=true [0,1], cols=pred [0,1]):")
print(cm)

# === 7) Save artifacts ===
os.makedirs(OUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().isoformat(timespec="seconds")

metrics = {
    "timestamp": timestamp,
    "threshold_locked": THRESHOLD,
    "counts": {
        "total": int(len(y_test)),
        "tumor_positives": int(y_test.sum()),
        "no_tumor_negatives": int((y_test == 0).sum())
    },
    "metrics": {
        "accuracy": acc,
        "precision_tumor": prec,
        "recall_tumor": rec,
        "f1_tumor": f1,
        "specificity_no_tumor": specificity,
        "npv_no_tumor": npv,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    },
    "confusion_matrix": {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "format": "rows=true [no_tumor(0), tumor(1)], cols=pred [0,1]"
    },
    "best_f1_sweep": {
        "f1": best[0], "threshold": best[1],
        "precision": best[2], "recall": best[3]
    }
}

with open(Path(OUT_DIR) / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

md = []
md.append(f"# Final Test Evaluation\n")
md.append(f"- **Date**: {timestamp}")
md.append(f"- **Locked Threshold**: `{THRESHOLD}`\n")
md.append("## Summary\n")
md.append(f"- Test images: **{metrics['counts']['total']}**")
md.append(f"- Tumor (1): **{metrics['counts']['tumor_positives']}**")
md.append(f"- No tumor (0): **{metrics['counts']['no_tumor_negatives']}**\n")
md.append("## Metrics (at locked threshold)\n")
md.append(f"- Accuracy: **{acc:.4f}**")
md.append(f"- Precision (tumor=1): **{prec:.4f}**")
md.append(f"- Recall (tumor=1): **{rec:.4f}**")
md.append(f"- F1: **{f1:.4f}**")
md.append(f"- Specificity (no_tumor): **{specificity:.4f}**")
md.append(f"- NPV (no_tumor): **{npv:.4f}**")
md.append(f"- ROC-AUC (probs): **{roc_auc:.4f}**")
md.append(f"- PR-AUC  (probs): **{pr_auc:.4f}**\n")
md.append("## Confusion Matrix\n")
md.append("|            | Pred 0 | Pred 1 |")
md.append("|------------|--------:|-------:|")
md.append(f"| **True 0** | {tn:6d} | {fp:6d} |")
md.append(f"| **True 1** | {fn:6d} | {tp:6d} |\n")
md.append(
    "_Format: rows = true labels [no_tumor(0), tumor(1)] ; columns = predicted labels [0,1]._\n")
md.append("## Threshold Sweep (reference)\n")
md.append(f"- Best F1 on test: **{best[0]:.3f}** at threshold **{best[1]:.2f}** "
          f"(precision **{best[2]:.3f}**, recall **{best[3]:.3f}**)")
with open(Path(OUT_DIR) / "results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\n[SAVED] {Path(OUT_DIR) / 'metrics.json'}")
print(f"[SAVED] {Path(OUT_DIR) / 'results.md'}")

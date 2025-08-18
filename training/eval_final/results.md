# Final Test Evaluation

- **Date**: 2025-08-18T19:08:55
- **Locked Threshold**: `0.05`

## Summary

- Test images: **1315**
- Tumor (1): **908**
- No tumor (0): **407**

## Metrics (at locked threshold)

- Accuracy: **0.8000**
- Precision (tumor=1): **0.7783**
- Recall (tumor=1): **0.9934**
- F1: **0.8728**
- Specificity (no_tumor): **0.3686**
- NPV (no_tumor): **0.9615**
- ROC-AUC (probs): **0.9203**
- PR-AUC  (probs): **0.9553**

## Confusion Matrix

|            | Pred 0 | Pred 1 |
|------------|--------:|-------:|
| **True 0** |    150 |    257 |
| **True 1** |      6 |    902 |

_Format: rows = true labels [no_tumor(0), tumor(1)] ; columns = predicted labels [0,1]._

## Threshold Sweep (reference)

- Best F1 on test: **0.920** at threshold **0.20** (precision **0.906**, recall **0.935**)
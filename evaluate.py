# evaluate.py
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from pathlib import Path

def evaluate(scores: np.ndarray, labels: np.ndarray,
             model_name: str = "model", service: str = "") -> dict:
    """
    Follows LO2 paper protocol exactly:
    - AUCROC = primary metric (threshold-free, fair for unsupervised)
    - F1 reported at the threshold that maximises it (same as paper)
    """
    if len(np.unique(labels)) < 2:
        print(f"  [{service}] Only one class in test set — skipping.")
        return {}

    aucroc = roc_auc_score(labels, scores)

    _, _, thresholds = roc_curve(labels, scores)
    best_f1, best_t = 0.0, 0.0
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    print(f"  [{model_name} | {service}] "
          f"AUCROC={aucroc:.4f}  F1={best_f1:.4f}  "
          f"thresh={best_t:.4f}  "
          f"n_test={len(labels)}")

    return {
        "service":    service,
        "model":      model_name,
        "aucroc":     round(aucroc, 4),
        "f1":         round(best_f1, 4),
        "threshold":  round(best_t, 4),
        "n_test":     int(len(labels)),
        "n_anomaly":  int(labels.sum()),
    }

def save_results(all_results: list, path: str = "results.json"):
    """Save raw results + print a comparison table."""
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    if df.empty:
        print("No results to display.")
        return

    # Pivot: services as rows, models as columns
    pivot = df.pivot_table(
        index="service",
        columns="model",
        values=["aucroc", "f1"],
        aggfunc="first"
    ).round(4)
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(pivot.to_string())

    # Macro-average row
    print("\nMacro average:")
    print(df.groupby("model")[["aucroc", "f1"]].mean().round(4).to_string())
    print("="*60)
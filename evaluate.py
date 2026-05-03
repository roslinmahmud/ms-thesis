# evaluate.py
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from pathlib import Path

def evaluate(scores: np.ndarray, labels: np.ndarray,
             model_name: str = "model", service: str = "") -> dict:

    if len(np.unique(labels)) < 2:
        print(f"  [{service}] Only one class — skipping.")
        return {}

    # ── Auto-correct score direction ──────────────────────────────────────────
    # AUCROC < 0.5 means signal is inverted — flip and record it
    raw_aucroc = roc_auc_score(labels, scores)
    flipped = False
    if raw_aucroc < 0.5:
        scores  = -scores
        flipped = True
        print(f"  ⚠  [{model_name}] Inverted signal detected "
              f"(raw={raw_aucroc:.4f}) — scores negated")

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
          f"flipped={flipped}")

    return {
        "service":   service,
        "model":     model_name,
        "aucroc":    round(aucroc, 4),
        "f1":        round(best_f1, 4),
        "threshold": round(float(best_t), 4),
        "flipped":   flipped,
        "n_test":    int(len(labels)),
        "n_anomaly": int(labels.sum()),
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

def length_baseline_aucroc(test_seqs):
    """
    Baseline: predicts anomaly score = sequence length only.
    If this scores high AUCROC, your dataset has a length confound
    and your other models might be exploiting it.
    """
    scores = np.array([len(s["text"]) for s in test_seqs])
    labels = np.array([s["label"] for s in test_seqs])
    return roc_auc_score(labels, scores)
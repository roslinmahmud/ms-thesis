# experiment_baselines.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import time
import torch

from preprocessing import build_sequences, split, SERVICES
from baseline_if import run_if_all_representations
from baseline_logbert import fine_tune_logbert, score_all_logbert
from evaluate import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_FILE     = "results_baselines.json"
RESULTS          = []
EXPERIMENT_START = time.time()

def elapsed(since):
    secs = time.time() - since
    return f"{secs/3600:.1f}h" if secs > 3600 else f"{secs/60:.1f}m"

def save_checkpoint(results, path=RESULTS_FILE):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Checkpoint saved → {path}  ({len(results)} records so far)")

# ── Main loop ─────────────────────────────────────────────────────────────────
for service in SERVICES:
    service_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Service: {service}")
    print(f"  Experiment elapsed: {elapsed(EXPERIMENT_START)}")
    print(f"{'='*60}")

    # ── 1. Build sequences ────────────────────────────────────────────────────
    print(f"\n  [1/3] Building sequences...")
    t0 = time.time()
    seqs = build_sequences(service)

    if not seqs:
        print(f"  No sequences — skipping {service}.")
        continue

    train_seqs, test_seqs = split(seqs, method="balanced")
    print(f"  Built in {elapsed(t0)} | "
          f"train={len(train_seqs)} | test={len(test_seqs)}")

    # ── 2. Isolation Forest — all three representations ───────────────────────
    print(f"\n  [2/3] Isolation Forest...")
    t0 = time.time()
    if_results = run_if_all_representations(train_seqs, test_seqs, service)
    if_time = round((time.time() - t0) / 3600, 2)
    for r in if_results:
        r["n_train"] = len(train_seqs)
        r["service_time_hrs"] = if_time
        RESULTS.append(r)
    save_checkpoint(RESULTS)
    print(f"  IF complete in {elapsed(t0)}")

    # ── 3. LogBERT ───────────────────────────────────────────────────────────
    print(f"\n  [3/3] LogBERT...")
    t0 = time.time()

    try:
        fine_tune_logbert(service, train_seqs)
        lb_scores, lb_labels = score_all_logbert(test_seqs, service)
        lb_result = evaluate(
            lb_scores, lb_labels,
            model_name="logbert",
            service=service
        )
        if lb_result:
            lb_result["n_train"] = len(train_seqs)
            lb_result["service_time_hrs"] = round(
                (time.time() - service_start) / 3600, 2
            )
            RESULTS.append(lb_result)
            save_checkpoint(RESULTS)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ⚠  LogBERT failed for {service}: {e}")
        print(f"  Continuing with next service...")

    print(f"\n  Service '{service}' complete in {elapsed(service_start)}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"  BASELINES COMPLETE")
print(f"  Total time: {elapsed(EXPERIMENT_START)}")
print(f"{'='*60}\n")

if RESULTS:
    import pandas as pd
    df = pd.DataFrame(RESULTS)

    print("Results by model:")
    summary = df.groupby("model")[["aucroc", "f1"]].mean().round(4)
    print(summary.to_string())

    print("\nAUCROC per service:")
    pivot_auc = df.pivot_table(
        index="service",
        columns="model",
        values="aucroc",
        aggfunc="first"
    ).round(4)
    print(pivot_auc.to_string())

    print("\nF1 per service:")
    pivot_f1 = df.pivot_table(
        index="service",
        columns="model",
        values="f1",
        aggfunc="first"
    ).round(4)
    print(pivot_f1.to_string())

    print("\nPrecision per service:")
    pivot_p = df.pivot_table(
        index="service",
        columns="model",
        values="precision",
        aggfunc="first"
    ).round(4)
    print(pivot_p.to_string())

    print("\nRecall per service:")
    pivot_r = df.pivot_table(
        index="service",
        columns="model",
        values="recall",
        aggfunc="first"
    ).round(4)
    print(pivot_r.to_string())

    csv_path = "results_baselines.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFinal results saved → {csv_path}")
else:
    print("No results collected — check logs for errors.")

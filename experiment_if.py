# experiment_if.py
# ──────────────────────────────────────────────────────────────────────────────

import json
import os
import time

from preprocessing import build_sequences, split, SERVICES
from baseline_if import run_if_all_representations

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_FILE     = "results_if_skip50.json"
EXPERIMENT_START = time.time()

# ── Resume: load existing results and find already-completed services ─────────
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        RESULTS = json.load(f)
    DONE_SERVICES = {r["service"] for r in RESULTS}
    print(f"  Resuming — {len(DONE_SERVICES)} services already done: {sorted(DONE_SERVICES)}")
else:
    RESULTS = []
    DONE_SERVICES = set()

def elapsed(since):
    secs = time.time() - since
    return f"{secs/3600:.1f}h" if secs > 3600 else f"{secs/60:.1f}m"

def save_checkpoint(results, path=RESULTS_FILE):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Checkpoint saved → {path}  ({len(results)} records so far)")

# ── Main loop ─────────────────────────────────────────────────────────────────
for service in SERVICES:
    if service in DONE_SERVICES:
        print(f"  Skipping '{service}' — already in results.")
        continue

    service_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Service: {service}")
    print(f"  Experiment elapsed: {elapsed(EXPERIMENT_START)}")
    print(f"{'='*60}")

    print(f"\n  [1/2] Building sequences...")
    t0 = time.time()
    seqs = build_sequences(service, skip_init_events=50)

    if not seqs:
        print(f"  No sequences — skipping {service}.")
        continue

    train_seqs, test_seqs = split(seqs)
    print(f"  Built in {elapsed(t0)} | "
          f"train={len(train_seqs)} | test={len(test_seqs)}")

    print(f"\n  [2/2] Isolation Forest...")
    t0 = time.time()
    if_results = run_if_all_representations(train_seqs, test_seqs, service)
    if_time = round((time.time() - t0) / 3600, 2)
    for r in if_results:
        r["n_train"] = len(train_seqs)
        r["service_time_hrs"] = if_time
        RESULTS.append(r)
    save_checkpoint(RESULTS)
    print(f"  IF complete in {elapsed(t0)}")

    print(f"\n  Service '{service}' complete in {elapsed(service_start)}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"  ISOLATION FOREST COMPLETE")
print(f"  Total time: {elapsed(EXPERIMENT_START)}")
print(f"{'='*60}\n")

if RESULTS:
    import pandas as pd
    df = pd.DataFrame(RESULTS)

    print("Results by representation:")
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

    csv_path = "results_if_skip50.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFinal results saved → {csv_path}")
else:
    print("No results collected — check logs for errors.")

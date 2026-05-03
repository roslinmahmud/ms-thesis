# generate_stats.py
# Run this after preprocessing to collect statistics for thesis section 4.4.
# Output: stats_4_4.json  +  console summary
#
# Usage:
#   python generate_stats.py
#
# Requires the same environment as experiment_baselines.py

import json
import numpy as np
from collections import defaultdict
from preprocessing import build_sequences, split, SERVICES

stats = {}

for service in SERVICES:
    print(f"\nProcessing: {service}")
    seqs = build_sequences(service)

    if not seqs:
        print(f"  No sequences found — skipping.")
        continue

    train_seqs, test_seqs = split(seqs, method="balanced")
    all_seqs = train_seqs + test_seqs

    # ── Token lengths ──────────────────────────────────────────────────────────
    # Assumes each sequence object has a .tokens attribute (list of token ids or
    # strings). Adjust the attribute name if yours differs (e.g. .token_ids).
    lengths = [len(seq.tokens) for seq in all_seqs]
    train_lengths = [len(seq.tokens) for seq in train_seqs]

    # ── Vocabulary (training split only — no data leakage) ────────────────────
    vocab = set()
    for seq in train_seqs:
        vocab.update(seq.tokens)

    # ── Class counts ──────────────────────────────────────────────────────────
    n_normal_train  = sum(1 for s in train_seqs if not s.is_anomaly)
    n_anomaly_train = sum(1 for s in train_seqs if s.is_anomaly)
    n_normal_test   = sum(1 for s in test_seqs  if not s.is_anomaly)
    n_anomaly_test  = sum(1 for s in test_seqs  if s.is_anomaly)

    # ── Per-error-type coverage ────────────────────────────────────────────────
    error_types = defaultdict(int)
    for seq in all_seqs:
        if seq.is_anomaly:
            error_types[seq.scenario] += 1

    stats[service] = {
        "total_sequences":      len(all_seqs),
        "train_sequences":      len(train_seqs),
        "test_sequences":       len(test_seqs),

        "class_balance": {
            "train_normal":     n_normal_train,
            "train_anomaly":    n_anomaly_train,
            "test_normal":      n_normal_test,
            "test_anomaly":     n_anomaly_test,
        },

        "token_lengths": {
            "mean":             round(float(np.mean(lengths)), 1),
            "median":           round(float(np.median(lengths)), 1),
            "std":              round(float(np.std(lengths)), 1),
            "min":              int(np.min(lengths)),
            "max":              int(np.max(lengths)),
            "p95":              round(float(np.percentile(lengths, 95)), 1),
            "p99":              round(float(np.percentile(lengths, 99)), 1),
        },

        "train_vocab_size":     len(vocab),
        "error_types_seen":     len(error_types),
        "error_type_counts":    dict(error_types),
    }

    print(f"  Sequences  — train: {len(train_seqs)}, test: {len(test_seqs)}")
    print(f"  Lengths    — mean: {stats[service]['token_lengths']['mean']}, "
          f"max: {stats[service]['token_lengths']['max']}, "
          f"p99: {stats[service]['token_lengths']['p99']}")
    print(f"  Vocab size — {len(vocab):,}")
    print(f"  Balance    — train {n_normal_train}N / {n_anomaly_train}A  |  "
          f"test {n_normal_test}N / {n_anomaly_test}A")

# ── Save ───────────────────────────────────────────────────────────────────────
output_path = "stats_4_4.json"
with open(output_path, "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nSaved → {output_path}")

# ── Cross-service summary table ────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Service':<16} {'Total':>7} {'Vocab':>8} {'MeanLen':>9} {'MaxLen':>8}")
print("="*70)
for svc, s in stats.items():
    print(f"{svc:<16} "
          f"{s['total_sequences']:>7} "
          f"{s['train_vocab_size']:>8,} "
          f"{s['token_lengths']['mean']:>9.1f} "
          f"{s['token_lengths']['max']:>8}")
print("="*70)

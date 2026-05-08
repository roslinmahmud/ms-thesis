import time

import pandas as pd

import torch, json
from preprocessing import build_sequences, split, SERVICES
from train import fine_tune, LORA_RANK
from score import load_finetuned, score_sequences
from evaluate import evaluate, save_results
from baseline_if import run_isolation_forest

from huggingface_hub import login
login("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")

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_FILE = "results_llm_lora_skip50.json"
RESULTS      = []

def elapsed(since):
    secs = time.time() - since
    return f"{secs/3600:.1f}h" if secs > 3600 else f"{secs/60:.1f}m"

def save_checkpoint(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {RESULTS_FILE}  ({len(results)} services complete)")

# ── Main loop ─────────────────────────────────────────────────────────────────
start = time.time()

for service in SERVICES:
    t_service = time.time()
    print(f"\n{'='*55}\n  Service: {service}\n{'='*55}")

    seqs = build_sequences(service, skip_init_events=50)
    if not seqs:
        print("  No data — skipping.")
        continue

    train_seqs, test_seqs = split(seqs)

    if len(train_seqs) < 5:
        print(f"  Too few training sequences ({len(train_seqs)}) — skipping.")
        continue

    # Fine-tune
    print(f"\n  [1/3] Training (rank={LORA_RANK})...")
    fine_tune(service, train_seqs, rank=LORA_RANK)

    # Score
    print(f"\n  [2/3] Scoring...")
    model, tokenizer = load_finetuned(service, rank=LORA_RANK)
    scores, labels   = score_sequences(test_seqs, model, tokenizer)
    del model, tokenizer
    torch.cuda.empty_cache()

    # Evaluate
    print(f"\n  [3/3] Evaluating...")
    result = evaluate(scores, labels,
                      model_name="llm_lora", service=service)
    if result:
        result["time_hrs"] = round((time.time() - t_service) / 3600, 2)
        RESULTS.append(result)
        save_checkpoint(RESULTS)

    print(f"  Done in {elapsed(t_service)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Total time: {elapsed(start)}")
print(f"{'='*55}\n")

df = pd.DataFrame(RESULTS)
print(df[["service", "aucroc", "f1", "flipped"]].to_string(index=False))
print(f"\nMacro AUCROC: {df['aucroc'].mean():.4f}")
print(f"Macro F1:     {df['f1'].mean():.4f}")
df.to_csv("results_llm_lora_skip50.csv", index=False)
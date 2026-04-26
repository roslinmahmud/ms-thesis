# baseline_logbert.py
import torch
import numpy as np
from pathlib import Path
from transformers import (
    BertTokenizer, BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments, Trainer,
    set_seed
)
from datasets import Dataset
from evaluate import evaluate
from preprocessing import build_sequences, split, SERVICES

set_seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
# bert-base-uncased: 110M params, no GPU quantization needed (much smaller than LLM)
# If even that's too heavy, swap for "prajjwal1/bert-tiny" (4.4M params)
MODEL_ID     = "bert-base-uncased"
CKPT_BASE    = Path("./checkpoints_logbert")
MAX_LEN      = 512          # BERT hard limit is 512 tokens
MLM_PROB     = 0.15         # mask 15% of tokens — standard BERT setting
TOP_K        = 9            # LogBERT paper default: anomaly if true token not in top-9
EPOCHS       = 5            # BERT needs more epochs than LLM (smaller model)
BATCH_SIZE   = 8
GRAD_ACCUM   = 2


# ── 1. Fine-tuning on normal sequences (MLM objective) ───────────────────────

def fine_tune_logbert(service: str, train_seqs: list) -> Path:
    ckpt_dir = CKPT_BASE / service
    if ckpt_dir.exists():
        print(f"  Checkpoint exists at {ckpt_dir} — skipping training.")
        return ckpt_dir

    tokenizer = BertTokenizer.from_pretrained(MODEL_ID)
    model     = BertForMaskedLM.from_pretrained(MODEL_ID)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )

    ds = Dataset.from_list([{"text": s["text"]} for s in train_seqs])
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])

    # Standard MLM collator — works reliably across all transformers versions
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        lr_scheduler_type="linear",
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"  Training LogBERT for {service} "
          f"({len(train_seqs)} normal sequences)...")
    trainer.train()
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"  Saved to {ckpt_dir}")

    del model, trainer
    torch.cuda.empty_cache()
    return ckpt_dir


# ── 2. Anomaly scoring ────────────────────────────────────────────────────────

@torch.no_grad()
def score_sequence_logbert(
    text: str,
    model: BertForMaskedLM,
    tokenizer: BertTokenizer,
    top_k: int = TOP_K,
) -> float:
    """
    LogBERT anomaly score for one sequence.

    Protocol (from Guo et al., 2021):
      For each token position:
        1. Mask that token
        2. Ask BERT: what are the top-k predictions here?
        3. If the true token is NOT in top-k → that position is anomalous
      Score = fraction of positions that are anomalous
              (higher = more anomalous)

    Intuition: a model trained on normal logs will confidently predict
    normal tokens in context. Anomalous sequences contain tokens the
    model doesn't expect → they fall outside the top-k.
    """
    model.eval()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    ).to(model.device if hasattr(model, 'device') else next(model.parameters()).device)

    input_ids = enc["input_ids"][0]         # shape: (seq_len,)
    n_tokens  = len(input_ids)

    # Skip [CLS] and [SEP] — they're always predictable, not informative
    token_positions = range(1, n_tokens - 1)
    if len(token_positions) == 0:
        return 0.0

    anomalous_count = 0

    for pos in token_positions:
        true_token = input_ids[pos].item()

        # Build masked input: replace position pos with [MASK]
        masked_ids = input_ids.clone()
        masked_ids[pos] = tokenizer.mask_token_id

        masked_enc = {
            "input_ids":      masked_ids.unsqueeze(0),
            "attention_mask": enc["attention_mask"],
        }
        if "token_type_ids" in enc:
            masked_enc["token_type_ids"] = enc["token_type_ids"]

        # Get model predictions at masked position
        logits   = model(**masked_enc).logits          # (1, seq_len, vocab)
        top_k_ids = torch.topk(logits[0, pos], k=top_k).indices.tolist()

        if true_token not in top_k_ids:
            anomalous_count += 1

    return anomalous_count / len(token_positions)      # fraction anomalous


def score_all_logbert(
    sequences: list,
    service: str,
    top_k: int = TOP_K,
) -> tuple:
    """Loads checkpoint and scores all sequences. Returns (scores, labels)."""
    ckpt_dir = CKPT_BASE / service
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No LogBERT checkpoint at {ckpt_dir}. Run fine_tune_logbert() first."
        )

    tokenizer = BertTokenizer.from_pretrained(str(ckpt_dir))
    model     = BertForMaskedLM.from_pretrained(str(ckpt_dir))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    model.eval()

    scores, labels = [], []
    for i, seq in enumerate(sequences):
        s = score_sequence_logbert(seq["text"], model, tokenizer, top_k)
        scores.append(s)
        labels.append(seq["label"])
        if (i + 1) % 10 == 0:
            print(f"    Scored {i+1}/{len(sequences)}  "
                  f"(last={s:.4f}, label={seq['label']})")

    del model
    torch.cuda.empty_cache()
    return np.array(scores), np.array(labels)


# ── 3. Top-k ablation (optional but useful for thesis) ───────────────────────

def run_topk_ablation(sequences: list, service: str, top_k_values=[1, 3, 9, 15]):
    """
    Tests different top-k thresholds on the same checkpoint.
    The LogBERT paper found k=9 optimal — verify this holds for your data.
    """
    _, test_seqs = split(sequences)
    results = []

    ckpt_dir = CKPT_BASE / service
    tokenizer = BertTokenizer.from_pretrained(str(ckpt_dir))
    model     = BertForMaskedLM.from_pretrained(str(ckpt_dir))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    # Score once per sequence (expensive), then apply different k thresholds
    # Here we store full logit ranks to avoid re-running the model per k
    print("  Scoring sequences once for all k values...")
    all_ranks = []   # for each sequence: list of (true_token_rank) per position

    for seq in test_seqs:
        enc = tokenizer(
            seq["text"], return_tensors="pt",
            truncation=True, max_length=MAX_LEN
        ).to(device)
        input_ids = enc["input_ids"][0]
        ranks = []

        for pos in range(1, len(input_ids) - 1):
            true_token = input_ids[pos].item()
            masked     = input_ids.clone()
            masked[pos] = tokenizer.mask_token_id
            logits = model(
                input_ids=masked.unsqueeze(0),
                attention_mask=enc["attention_mask"]
            ).logits
            sorted_ids = torch.argsort(logits[0, pos], descending=True).tolist()
            rank = sorted_ids.index(true_token) if true_token in sorted_ids else MAX_LEN
            ranks.append(rank)

        all_ranks.append(ranks)

    labels = np.array([s["label"] for s in test_seqs])

    for k in top_k_values:
        # Score = fraction of tokens with rank >= k (not in top-k)
        scores = np.array([
            sum(r >= k for r in ranks) / max(len(ranks), 1)
            for ranks in all_ranks
        ])
        res = evaluate(scores, labels,
                       model_name=f"logbert_k{k}", service=service)
        results.append(res)
        print(f"  k={k:>2}  AUCROC={res['aucroc']:.4f}  F1={res['f1']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


# ── 4. Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    service = "token"
    print(f"\n{'='*55}\n  LogBERT — {service}\n{'='*55}")

    seqs = build_sequences(service)
    train_seqs, test_seqs = split(seqs)

    # Train
    fine_tune_logbert(service, train_seqs)

    # Score & evaluate
    print("\n  Scoring test sequences...")
    scores, labels = score_all_logbert(test_seqs, service)
    result = evaluate(scores, labels, model_name="logbert", service=service)
    
    # Compare all three models on the same test set
    print("\n── Full comparison (token service) ──────────────────────")
    print(f"  logbert              AUCROC={result['aucroc']:.4f}  F1={result['f1']:.4f}")
    print(f"  IF trigrams          AUCROC=0.6002  F1=0.6706   (previous run)")
    print(f"  LLM+LoRA (Gemma-2B)  AUCROC=0.5337  F1=0.6746   (previous run)")
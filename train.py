import os, torch
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from preprocessing import build_sequences, split, SERVICES

set_seed(42)

# ── Config — tweak these per experiment ───────────────────────────────────────
MODEL_ID      = "meta-llama/Meta-Llama-3-8B"
CKPT_BASE     = Path("./checkpoints")
MAX_LEN       = 512    # tokens; reduce to 512 or 256 if you hit OOM
LORA_RANK     = 16     # ablate: 4, 8, 16, 32
LORA_ALPHA    = 32     # rule of thumb: 2 × rank
EPOCHS        = 3
BATCH_SIZE    = 1
GRAD_ACCUM    = 16     # effective batch = 16

# ── 4-bit quantization config (QLoRA) ─────────────────────────────────────────
def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # saves ~0.4 bits per param extra
    )

# ── Model + tokenizer loader ──────────────────────────────────────────────────
def load_base_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # required for causal LM training

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_bnb_config(),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    # Required step before attaching LoRA to a quantized model
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

# ── LoRA adapter ──────────────────────────────────────────────────────────────
def attach_lora(model, rank: int = LORA_RANK):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        # Target all attention projections for best coverage
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

# ── Dataset preparation ───────────────────────────────────────────────────────
def make_hf_dataset(train_seqs: list, tokenizer, max_len: int) -> Dataset:
    """
    Converts normal-only sequences into a HuggingFace Dataset.
    Labels = input_ids (causal LM: predict next token at every position).
    """
    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = Dataset.from_list([{"text": s["text"]} for s in train_seqs])
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds

# ── Training ──────────────────────────────────────────────────────────────────
def fine_tune(service: str, train_seqs: list, rank: int = LORA_RANK):
    """
    Fine-tunes LLM+LoRA on normal-only log sequences for a given service.
    Saves the LoRA adapter (not the full model) to checkpoints/<service>/r<rank>/
    """
    ckpt_dir = CKPT_BASE / service / f"r{rank}"
    if ckpt_dir.exists():
        print(f"  Checkpoint exists at {ckpt_dir} — skipping training.")
        return ckpt_dir

    print(f"\n  Loading base model...")
    model, tokenizer = load_base_model(MODEL_ID)
    model = attach_lora(model, rank)

    print(f"  Preparing dataset ({len(train_seqs)} sequences)...")
    dataset = make_hf_dataset(train_seqs, tokenizer, MAX_LEN)

    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,         # keep only the best checkpoint
        load_best_model_at_end=False,
        report_to="none",           # swap to "wandb" for loss curve tracking
        run_name=f"{service}-r{rank}",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"  Training {service} (rank={rank}, epochs={EPOCHS})...")
    trainer.train()

    # Save only the lightweight LoRA adapter weights (~10-50 MB vs 16 GB)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    print(f"  Saved adapter to {ckpt_dir}")

    # Free VRAM before next service
    del model, trainer
    torch.cuda.empty_cache()

    return ckpt_dir


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for service in SERVICES:
        print(f"\n{'='*55}")
        print(f"  Service: {service}")
        print(f"{'='*55}")

        seqs = build_sequences(service)
        if not seqs:
            print(f"  No sequences found — skipping.")
            continue

        train_seqs, _ = split(seqs)

        if len(train_seqs) < 5:
            print(f"  Too few training sequences ({len(train_seqs)}) — skipping.")
            continue

        fine_tune(service, train_seqs, rank=LORA_RANK)
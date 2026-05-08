# score.py
import torch, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# Make sure MODEL_ID matches your training notebook!
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
MAX_LEN   = 1024
CKPT_BASE = Path("./checkpoints_skip50")

def load_finetuned(service: str, rank: int = 16):
    ckpt_dir = CKPT_BASE / service / f"r{rank}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_dir}. Run train.py first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Define the EXACT SAME 4-bit config used during training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Load the base model in 4-bit
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # 3. Attach LoRA
    model = PeftModel.from_pretrained(base, str(ckpt_dir))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def perplexity(text: str, model, tokenizer, max_len: int = 1024) -> float:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(model.device)

    n_tokens = enc["input_ids"].shape[1]

    # Raw loss from model (already mean over tokens internally)
    outputs = model(**enc, labels=enc["input_ids"])
    raw_loss = outputs.loss.item()

    # Length penalty: penalise short sequences
    # log(n_tokens) normalisation — standard in perplexity literature
    length_penalty = np.log(max(n_tokens, 1))
    normalised_loss = raw_loss / length_penalty

    return normalised_loss

def score_sequences(sequences: list, model, tokenizer) -> tuple:
    """Returns (scores array, labels array) for a list of sequences."""
    scores, labels = [], []
    for i, seq in enumerate(sequences):
        s = perplexity(seq["text"], model, tokenizer)
        scores.append(s)
        labels.append(seq["label"])
        if (i + 1) % 10 == 0:
            print(f"    Scored {i+1}/{len(sequences)}  "
                  f"(last: {s:.4f}, label={seq['label']})")

    return np.array(scores), np.array(labels)
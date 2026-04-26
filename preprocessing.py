import re, os, random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── Constants matching your dataset spec ──────────────────────────────────────
DATASET_ROOT  = Path("light-oauth2-logs")
RUN_PREFIX    = "oauth2_logs_LO2_run_"
SERVICES      = ["client", "code", "key", "refresh-token",
                 "service", "token", "user"]

def log_filename(service: str) -> str:
    """Maps service name → actual filename in each scenario folder."""
    return f"light-oauth2-oauth2-{service}-1.log"

EVENT_START = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\b")

# ── Multiline event parser ────────────────────────────────────────────────────
def parse_events(log_path: Path) -> List[str]:
    """
    Groups raw lines into logical events.
    A new event starts when a line matches the timestamp pattern.
    Stack traces and continuations are appended to the previous event.
    """
    events, current = [], []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    for line in text.splitlines():
        if EVENT_START.match(line):
            if current:
                events.append("\n".join(current))
            current = [line]
        elif current:          # continuation of previous event
            current.append(line)
        # lines before the first event (rare) are silently skipped

    if current:
        events.append("\n".join(current))
    return events

# ── Log line cleaning ─────────────────────────────────────────────────────────
def clean_event(event: str) -> str:
    """Strip high-cardinality noise while preserving semantic content."""
    # Timestamps at line start
    event = re.sub(r"^\d{2}:\d{2}:\d{2}\.\d{3}", "", event, flags=re.MULTILINE)
    # Thread names like [XNIO-1 task-5]
    event = re.sub(r"\[[\w\s\-]+\]", "", event)
    # Hex addresses and UUIDs
    event = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HEX>", event)
    # IP addresses
    event = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b", "<IP>", event)
    # Long numeric IDs
    event = re.sub(r"\b\d{6,}\b", "<NUM>", event)
    return event.strip()

# ── Run discovery ─────────────────────────────────────────────────────────────
def discover_runs(root: Path) -> List[Path]:
    """Find all run folders by prefix — robust to partial/missing runs."""
    return sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name.startswith(RUN_PREFIX)
    ])

def extract_run_id(run_folder: Path) -> str:
    return run_folder.name.replace(RUN_PREFIX, "")

# ── Build sequences ───────────────────────────────────────────────────────────
def build_sequences(service: str, root: Path = DATASET_ROOT,
                    max_events: int = 256) -> List[dict]:
    """
    Returns list of dicts, one per (run, scenario) pair:
      {
        run_id:     str,
        scenario:   str,          # exact folder name, case-preserved
        service:    str,
        is_normal:  bool,
        label:      int,          # 0 = normal, 1 = anomaly
        text:       str,          # joined event texts for the LLM
        n_events:   int
      }
    """
    filename  = log_filename(service)
    sequences = []
    runs      = discover_runs(root)

    if not runs:
        raise FileNotFoundError(f"No run folders found under {root}")

    for run_folder in runs:
        run_id = extract_run_id(run_folder)

        # Each sub-folder of the run is a scenario (correct / error_type)
        scenario_folders = [
            d for d in run_folder.iterdir() if d.is_dir()
        ]
        if not scenario_folders:
            continue   # skip empty runs gracefully

        for scenario_folder in scenario_folders:
            scenario  = scenario_folder.name        # exact, case-sensitive
            is_normal = (scenario == "correct")
            label     = 0 if is_normal else 1

            log_path  = scenario_folder / filename
            if not log_path.exists():
                continue   # this service missing for this scenario — skip

            events = parse_events(log_path)
            if not events:
                continue   # empty log file — skip

            # Cap events, clean, then join with a separator the LLM can learn
            clean = [clean_event(e) for e in events[:max_events]]
            text  = "\n---\n".join(clean)   # \n---\n is clearer than [SEP]

            sequences.append({
                "run_id":    run_id,
                "scenario":  scenario,
                "service":   service,
                "is_normal": is_normal,
                "label":     label,
                "text":      text,
                "n_events":  len(events),
            })

    print(f"[{service}] {len(sequences)} sequences "
          f"({sum(s['label']==0 for s in sequences)} normal, "
          f"{sum(s['label']==1 for s in sequences)} anomalous)")
    return sequences

# ── Train / test split ────────────────────────────────────────────────────────
def split(sequences: List[dict], train_ratio: float = 0.5,
          seed: int = 42):
    """
    50/50 split following the LO2 paper protocol.
    Training set = NORMAL ONLY (self-supervised).
    Test set = balanced normal + anomalous.
    """
    random.seed(seed)
    normal  = [s for s in sequences if s["label"] == 0]
    anomaly = [s for s in sequences if s["label"] == 1]
    random.shuffle(normal)
    random.shuffle(anomaly)

    n_train = len(normal) // 2
    train   = normal[:n_train]
    normal_test = normal[n_train:]

    # Use first half of normals for training, rest go into test
    n_test_each = len(normal_test)
    anomaly_test = anomaly[:n_test_each]   # same count as normal in test

    test = normal_test + anomaly_test
    random.shuffle(test)

    print(f"  Train (normal only): {len(train)}")
    print(f"  Test  (mixed):       {len(test)}  "
          f"({sum(s['label']==0 for s in test)} normal, "
          f"{sum(s['label']==1 for s in test)} anomalous)")
    return train, test

# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    for svc in SERVICES:
        seqs = build_sequences(svc)
        train, test = split(seqs)
        # Print a sample to verify parsing looks correct
        sample = train[0]
        print(f"\nSample from {svc} / run {sample['run_id']}:")
        print(sample["text"][:400])
        print("...")
        break  # remove to run all services
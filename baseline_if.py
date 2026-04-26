# baseline_if.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from evaluate import evaluate
from preprocessing import build_sequences, split, SERVICES

# ── Feature extraction options ────────────────────────────────────────────────
# The LO2 paper tested "words", "trigrams", and "event IDs" as representations.
# We implement all three so you can compare in your ablation.

def make_word_vectorizer(max_features=2000):
    """Unigrams — simplest, most interpretable."""
    return TfidfVectorizer(
        analyzer="word",
        max_features=max_features,
        sublinear_tf=True,       # log(1+tf) — dampens very frequent terms
        min_df=2,                # ignore terms appearing in only 1 sequence
        token_pattern=r"[A-Za-z0-9_\.\-\/]{2,}",  # skip 1-char noise
    )

def make_trigram_vectorizer(max_features=2000):
    """Character 3-grams — captures partial log templates robustly."""
    return TfidfVectorizer(
        analyzer="char_wb",      # char n-grams within word boundaries
        ngram_range=(3, 3),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )

def make_event_id_vectorizer(max_features=2000):
    """
    Treats each log event header as a token (poor man's event ID).
    Extracts the logger class name — most stable part of a log line.
    e.g. 'c.n.openapi.ApiNormalisedPath' from the example in DATASET.md
    """
    import re
    def extract_loggers(text):
        # Match Java-style class names: letters, dots, camelCase
        loggers = re.findall(r'\b[a-z]+(?:\.[a-z]+)+(?:\.[A-Z][A-Za-z]+)+\b', text)
        return " ".join(loggers) if loggers else text

    return TfidfVectorizer(
        preprocessor=extract_loggers,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )

REPRESENTATIONS = {
    "words":     make_word_vectorizer,
    "trigrams":  make_trigram_vectorizer,
    "event_ids": make_event_id_vectorizer,
}

# ── Core runner ───────────────────────────────────────────────────────────────
def run_isolation_forest(
    train_seqs: list,
    test_seqs:  list,
    service:    str = "",
    representation: str = "words",   # "words" | "trigrams" | "event_ids"
    n_estimators: int = 100,
    contamination: float = "auto",   # "auto" = assume training is pure normal
    seed: int = 42,
) -> dict:
    """
    Trains an Isolation Forest on normal-only sequences and scores the test set.
    Returns evaluation dict with aucroc and f1.
    """
    if not train_seqs or not test_seqs:
        print(f"  [{service}] Empty split — skipping IF.")
        return {}

    # ── Vectorize ─────────────────────────────────────────────────────────────
    vectorizer = REPRESENTATIONS[representation](max_features=2000)

    train_texts = [s["text"] for s in train_seqs]
    test_texts  = [s["text"] for s in test_seqs]
    labels      = np.array([s["label"] for s in test_seqs])

    # Fit vectorizer on training data only — no leakage
    X_train = vectorizer.fit_transform(train_texts)
    X_test  = vectorizer.transform(test_texts)

    # L2 normalise — makes cosine-like distances, helps IF on sparse TF-IDF
    X_train = normalize(X_train, norm="l2")
    X_test  = normalize(X_test,  norm="l2")

    # ── Fit Isolation Forest ──────────────────────────────────────────────────
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,  # "auto" → decision boundary at 0.5
        random_state=seed,
        n_jobs=-1,                    # use all CPU cores
    )
    clf.fit(X_train)

    # ── Score ─────────────────────────────────────────────────────────────────
    # decision_function: higher = more normal → flip sign so higher = anomaly
    scores = -clf.decision_function(X_test)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model_name = f"isolation_forest_{representation}"
    result = evaluate(scores, labels, model_name=model_name, service=service)
    return result


# ── Run all representations for ablation (matches LO2 paper Table 3) ─────────
def run_if_all_representations(
    train_seqs: list,
    test_seqs:  list,
    service:    str = "",
) -> list:
    """
    Runs IF with all three representations and returns all results.
    Mirrors the LO2 paper's Table 3 structure exactly.
    """
    results = []
    for rep in REPRESENTATIONS:
        print(f"  [IF | {rep}]")
        res = run_isolation_forest(
            train_seqs, test_seqs, service=service, representation=rep
        )
        if res:
            results.append(res)
    return results


# ── Entry point — test standalone ────────────────────────────────────────────
if __name__ == "__main__":
    service = "token"
    print(f"\n{'='*55}\n  Isolation Forest — {service}\n{'='*55}")

    seqs = build_sequences(service)
    train_seqs, test_seqs = split(seqs)

    results = run_if_all_representations(train_seqs, test_seqs, service)

    print("\n── Summary ──")
    for r in results:
        print(f"  {r['model']:<40} "
              f"AUCROC={r['aucroc']:.4f}  F1={r['f1']:.4f}")
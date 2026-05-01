import time
from preprocessing import build_sequences, split, SERVICES

print(f"{'Service':<20} {'Normal':>8} {'Anomaly':>10} "
      f"{'Train':>8} {'Test':>8} {'Time':>8}")
print('-' * 65)

for svc in SERVICES:
    t0 = time.time()
    seqs = build_sequences(svc)
    train, test = split(seqs)
    elapsed = time.time() - t0

    n = sum(1 for s in seqs if s['label'] == 0)
    a = sum(1 for s in seqs if s['label'] == 1)
    mins = elapsed / 60

    print(f"{svc:<20} {n:>8} {a:>10} "
          f"{len(train):>8} {len(test):>8} {mins:>6.1f}m")

    # Flush immediately so you can tail -f and see progress
    import sys; sys.stdout.flush()
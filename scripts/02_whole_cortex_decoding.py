# Step 2: within-participant whole-cortex decoding, Chinese vs English
# (Methods 2.5, Fig. 1B; paper: 59.6% balanced accuracy, 95% CI [58.4, 60.9]).
#
# For each participant: train on one run, test on the other, then swap,
# and average the two balanced accuracies.
# Each trial = 800 parcel means. Z-scoring is fitted on the training run only.
#
# Usage:  python scripts/02_whole_cortex_decoding.py   (after step 1)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[1] / "results"
data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
print(len(subjects), "participants")


# leave-one-run-out decoding for one participant
def decode_across_runs(X, y, runs):
    scores = []
    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run

        # 1. z-score each parcel with the training run's mean and std
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])

        # 2. L2-regularized logistic regression
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y[train])

        # 3. score on the held-out run
        scores.append(balanced_accuracy_score(y[test], clf.predict(X_test)))
    return np.mean(scores)


rows = []
for s in subjects:
    X = data[f"X_{s}"]
    runs = data[f"run_{s}"]
    y = np.array([t[:2] for t in data[f"trial_type_{s}"]])   # "L1NS" -> "L1" (Chinese), "L2S" -> "L2" (English)

    acc = decode_across_runs(X, y, runs)
    rows.append({"subject": s, "balanced_accuracy": acc})
    print(f"{s}: {acc:.3f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "whole_cortex_decoding.csv", index=False)


# group mean with 95% CI: resample the 77 participants 5,000 times
acc = df["balanced_accuracy"].to_numpy()
rng = np.random.default_rng(42)
boot_means = [rng.choice(acc, size=len(acc), replace=True).mean() for _ in range(5000)]
low, high = np.percentile(boot_means, [2.5, 97.5])

print(f"\nmean balanced accuracy = {100 * acc.mean():.1f}%  95% CI [{100 * low:.1f}, {100 * high:.1f}]")
print(f"range {100 * acc.min():.1f}% to {100 * acc.max():.1f}%  (chance = 50%)")

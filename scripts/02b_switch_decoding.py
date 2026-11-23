# Step 2b: decoding across switching contexts (Methods 2.5, Fig. 1B).
#
# Same within-participant decoder as step 2 (train on one run, test on the other),
# on different subsets of trials:
#   repeat         Chinese vs English on non-switch trials only  (L1NS vs L2NS, 80 trials)
#   switch         Chinese vs English on switch trials only      (L1S  vs L2S,  80 trials)
#   switch_status  switch vs non-switch, all trials              (L1S+L2S vs L1NS+L2NS)
#
# Paper: repeat > switch by 2.8 pp [0.6, 5.1]; switch status 49.7% (no decoding).
#
# Usage:  python scripts/02b_switch_decoding.py   (after step 1)

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


# leave-one-run-out decoding, same as step 2
def decode_across_runs(X, y, runs):
    scores = []
    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y[train])
        scores.append(balanced_accuracy_score(y[test], clf.predict(X_test)))
    return np.mean(scores)


rows = []
for s in subjects:
    X = data[f"X_{s}"]
    runs = data[f"run_{s}"]
    trial_type = data[f"trial_type_{s}"]

    language = np.array([t[:2] for t in trial_type])                               # "L1" or "L2"
    is_repeat = np.array([t.endswith("NS") for t in trial_type])                   # L1NS, L2NS
    switch_status = np.where(is_repeat, "nonswitch", "switch")

    row = {
        "subject": s,
        "repeat": decode_across_runs(X[is_repeat], language[is_repeat], runs[is_repeat]),
        "switch": decode_across_runs(X[~is_repeat], language[~is_repeat], runs[~is_repeat]),
        "switch_status": decode_across_runs(X, switch_status, runs),
    }
    rows.append(row)
    print(f"{s}: repeat {row['repeat']:.3f}  switch {row['switch']:.3f}  switch status {row['switch_status']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "switch_decoding.csv", index=False)


# group means with 95% CI: resample participants 5,000 times
rng = np.random.default_rng(42)
boot_samples = [rng.integers(0, len(df), len(df)) for _ in range(5000)]


def mean_ci(values):
    boot = [values[sample].mean() for sample in boot_samples]
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


print("\nbalanced accuracy (%), mean [95% CI]")
for col in ["repeat", "switch", "switch_status"]:
    m, low, high = mean_ci(df[col].to_numpy())
    print(f"{col:<15}{100 * m:5.1f} [{100 * low:.1f}, {100 * high:.1f}]")

# paired difference: is decoding stronger on repeat than on switch trials?
m, low, high = mean_ci((df["repeat"] - df["switch"]).to_numpy())
print(f"repeat - switch {100 * m:+5.1f} pp [{100 * low:+.1f}, {100 * high:+.1f}]  (reliable if the CI excludes 0)")

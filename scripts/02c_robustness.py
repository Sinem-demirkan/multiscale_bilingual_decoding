# Step 2c: robustness of whole-cortex decoding to how the cortex is represented
# (Methods 2.5 "Robustness to cortical representation", SI Fig. S2).
#
# Same decoder as step 2 (Chinese vs English, train on one run, test on the other),
# with two changes to the features:
#   granularity   Schaefer atlases from 100 to 1000 parcels instead of 800
#   PCA           Schaefer-800, then PCA inside each fold: fitted on the training
#                 run, keeping the components that explain 90% of its variance
# Each is compared with the main analysis (Schaefer-800, no PCA) on the same participants.
#
# Granularity needs the parcel files of the other atlases: run script 01 with
# N_PARCELS = 100, 200, ..., 1000 first. Atlases without a file are skipped.
#
# Usage:  python scripts/02c_robustness.py   (after step 1)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[1] / "results"
GRANULARITIES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


# leave-one-run-out decoding, same as step 2, with optional PCA inside each fold
def decode_across_runs(X, y, runs, use_pca=False):
    scores = []
    for test_run in np.unique(runs):
        train = runs != test_run
        test = runs == test_run

        # 1. z-score each parcel with the training run's mean and std
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])

        # 2. optional: PCA on the training run, keep 90% of its variance
        if use_pca:
            pca = PCA(n_components=0.90, svd_solver="full")
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # 3. classifier, scored on the held-out run
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y[train])
        scores.append(balanced_accuracy_score(y[test], clf.predict(X_test)))
    return np.mean(scores)


def decode_file(n_parcels, use_pca=False):
    data = np.load(RESULTS / f"parcels_schaefer{n_parcels}.npz")
    subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
    scores = {}
    for s in subjects:
        y = np.array([t[:2] for t in data[f"trial_type_{s}"]])   # "L1" or "L2"
        scores[s] = decode_across_runs(data[f"X_{s}"], y, data[f"run_{s}"], use_pca)
    return pd.Series(scores)


# run every representation that has a parcel file; one column per representation
results = {}
for n in GRANULARITIES:
    if not (RESULTS / f"parcels_schaefer{n}.npz").exists():
        print(f"Schaefer-{n}: no parcel file, skipped (run script 01 with N_PARCELS = {n})")
        continue
    results[f"schaefer{n}"] = decode_file(n)
    print(f"Schaefer-{n} done")
results["schaefer800_pca90"] = decode_file(800, use_pca=True)
print("Schaefer-800 + PCA 90% done")

df = pd.DataFrame(results).rename_axis("subject")
df.to_csv(RESULTS / "robustness_by_subject.csv")


# group mean with 95% CI, and paired difference from the main analysis (Schaefer-800)
rng = np.random.default_rng(42)
boot_samples = [rng.integers(0, len(df), len(df)) for _ in range(5000)]


def mean_ci(values):
    boot = [values[sample].mean() for sample in boot_samples]
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


rows = []
print("\nbalanced accuracy (%), mean [95% CI]; difference from Schaefer-800 (pp)")
for col in df.columns:
    m, low, high = mean_ci(df[col].to_numpy())
    d, d_low, d_high = mean_ci((df[col] - df["schaefer800"]).to_numpy())
    rows.append({"representation": col, "accuracy": m, "ci_low": low, "ci_high": high,
                 "diff_from_800": d, "diff_ci_low": d_low, "diff_ci_high": d_high})
    print(f"{col:<20}{100 * m:5.1f} [{100 * low:.1f}, {100 * high:.1f}]"
          f"   {100 * d:+5.1f} [{100 * d_low:+.1f}, {100 * d_high:+.1f}]")
pd.DataFrame(rows).to_csv(RESULTS / "robustness_summary.csv", index=False)

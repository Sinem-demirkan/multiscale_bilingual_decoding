# Step 7: which shared dimensions carry transfer? (Methods 2.8, Fig. 2C)
#
# Same transfer as step 4 (teacher -> learner, PCA fitted without both), but the
# classifier gets ONE shared dimension at a time: dimension 1, 2, ..., 25 alone.
#
# Group mean per dimension with the participant bootstrap (as in step 3).
# One-sided p = share of bootstrap means at or below chance (50%),
# Benjamini-Hochberg across the 25 dimensions.
#
# Paper: 12 of 25 above chance; largest: dimensions 5, 10, 12, 3 (+4.3, +3.3, +3.2, +2.7 pp).
#
# Usage:  python scripts/07_single_dimension_transfer.py   (after step 4, which saves the PCA spaces)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

N_DIMS = 25
RESULTS = Path(__file__).resolve().parents[1] / "results"

# participants, z-scored on their own trials (as in step 4)
data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))
subject_X = {s: StandardScaler().fit_transform(data[f"X_{s}"]) for s in subjects}
subject_y = {s: np.array([t[:2] for t in data[f"trial_type_{s}"]]) for s in subjects}

# the PCA spaces saved by step 4: components[i] was fitted without the two people of pair i
pairs = pd.read_csv(RESULTS / "shared_dimensions_pairs.csv")
components = np.load(RESULTS / "shared_dimensions_components.npy", mmap_mode="r")
print(len(subjects), "participants,", len(pairs), "PCA spaces")


def transfer_score(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return balanced_accuracy_score(y_test, clf.predict(X_test))


rows = []
for i, (a, b) in enumerate(zip(pairs["subject_a"], pairs["subject_b"])):
    C = np.asarray(components[i, :N_DIMS])          # (25, 800)
    for teacher, learner in [(a, b), (b, a)]:
        scores_t = subject_X[teacher] @ C.T          # (n_trials, 25)
        scores_l = subject_X[learner] @ C.T
        row = {"teacher": teacher, "learner": learner}
        for d in range(N_DIMS):
            row[f"dim_{d + 1}"] = transfer_score(scores_t[:, [d]], subject_y[teacher],
                                                 scores_l[:, [d]], subject_y[learner])
        rows.append(row)
    if (i + 1) % 100 == 0:
        print(f"pair {i + 1}/{len(pairs)}", flush=True)

transfer = pd.DataFrame(rows)
transfer.to_csv(RESULTS / "single_dimension_transfer_by_pair.csv", index=False)


# ---- summary: participant bootstrap over teacher-learner pairs ---------------
index = {s: n for n, s in enumerate(subjects)}
t_idx = transfer["teacher"].map(index).to_numpy()
l_idx = transfer["learner"].map(index).to_numpy()
rng = np.random.default_rng(42)
pair_weights = []
for _ in range(5000):
    count = np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
    w = count[t_idx] * count[l_idx]
    if w.sum() > 0:                       # skip samples that drew a single person
        pair_weights.append(w)
pair_weights = np.array(pair_weights, dtype=float)

summary = []
for d in range(1, N_DIMS + 1):
    acc = transfer[f"dim_{d}"].to_numpy()
    boot = pair_weights @ acc / pair_weights.sum(axis=1)
    summary.append({"dimension": d, "accuracy": acc.mean(),
                    "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                    "p_one_sided": np.mean(boot <= 0.5)})
summary = pd.DataFrame(summary)
summary["q_bh"] = multipletests(summary["p_one_sided"], method="fdr_bh")[1]
summary["above_chance"] = summary["q_bh"] < 0.05
summary.to_csv(RESULTS / "single_dimension_transfer_summary.csv", index=False)

print("\ntransfer with one shared dimension alone (pp above chance), mean [95% CI]")
for _, r in summary.iterrows():
    star = "*" if r["above_chance"] else ""
    print(f"dim {int(r['dimension']):>2}  {100 * (r['accuracy'] - 0.5):+5.1f} "
          f"[{100 * (r['ci_low'] - 0.5):+.1f}, {100 * (r['ci_high'] - 0.5):+.1f}]  q = {r['q_bh']:.3f} {star}")
print(f"\n{summary['above_chance'].sum()} of {N_DIMS} dimensions above chance (BH q < 0.05)")

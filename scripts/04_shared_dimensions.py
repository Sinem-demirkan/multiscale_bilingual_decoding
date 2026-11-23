# Step 4: shared dimensions (Methods 2.8, Figs. 2B and 3A).
#
# For every PAIR of participants:
#   1. fit PCA on the pooled trials of the OTHER 75 participants (no language labels)
#      -> the "shared dimensions", fitted without the two people we test
#   2. represent each trial three ways:
#        full      all 800 parcels
#        shared    scores on the first k shared dimensions        X @ C.T
#        residual  what is left after removing them               X - X @ C.T @ C
#   3. test each representation
#        between people: train on one person (teacher), test on the other (learner),
#                        in both directions
#        within a person: leave-one-run-out, for both people in the pair
#
# Each participant is z-scored once across all of their own trials (no extra
# scaling inside the classifier), as in Methods 2.8.
# Within-person accuracies are later averaged over the 76 PCA spaces that exclude
# that person.
#
# Usage:  python scripts/04_shared_dimensions.py   (after step 1; takes 1-2 hours)

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

K_LIST = [1, 5, 10, 25, 50, 100, 200]

RESULTS = Path(__file__).resolve().parents[1] / "results"
data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))

# load everyone, z-score each participant on their own trials
subject_X, subject_y, subject_runs = {}, {}, {}
for s in subjects:
    subject_X[s] = StandardScaler().fit_transform(data[f"X_{s}"])
    subject_y[s] = np.array([t[:2] for t in data[f"trial_type_{s}"]])   # "L1" or "L2"
    subject_runs[s] = data[f"run_{s}"]
print(len(subjects), "participants")


def residualize(X, components):
    return X - X @ components.T @ components


# train on one person, test on another
def transfer_score(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return balanced_accuracy_score(y_test, clf.predict(X_test))


# within one person: train on one run, predict the other, score all predictions together
def loo_run_score(X, y, runs):
    preds = np.empty(len(y), dtype=object)
    for run in np.unique(runs):
        test = runs == run
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X[~test], y[~test])
        preds[test] = clf.predict(X[test])
    return balanced_accuracy_score(y, preds)


transfer_rows = []
within_rows = []
pairs = list(combinations(subjects, 2))   # 77 * 76 / 2 = 2,926 pairs

for i, (a, b) in enumerate(pairs, start=1):
    # 1. PCA on everyone except a and b
    pool = np.vstack([subject_X[s] for s in subjects if s not in (a, b)])
    pca = PCA(n_components=max(K_LIST), svd_solver="randomized", random_state=42)
    pca.fit(pool)
    all_components = pca.components_   # (200, 800), one row per shared dimension

    # 2. between people, both directions
    for teacher, learner in [(a, b), (b, a)]:
        X_t, y_t = subject_X[teacher], subject_y[teacher]
        X_l, y_l = subject_X[learner], subject_y[learner]
        row = {"teacher": teacher, "learner": learner, "full": transfer_score(X_t, y_t, X_l, y_l)}
        for k in K_LIST:
            C = all_components[:k]
            row[f"shared_{k}"] = transfer_score(X_t @ C.T, y_t, X_l @ C.T, y_l)
            row[f"residual_{k}"] = transfer_score(residualize(X_t, C), y_t, residualize(X_l, C), y_l)
        transfer_rows.append(row)

    # 3. within each person of the pair
    for s in (a, b):
        X, y, runs = subject_X[s], subject_y[s], subject_runs[s]
        row = {"subject": s, "space": f"{a}_{b}", "full": loo_run_score(X, y, runs)}
        for k in K_LIST:
            C = all_components[:k]
            row[f"shared_{k}"] = loo_run_score(X @ C.T, y, runs)
            row[f"residual_{k}"] = loo_run_score(residualize(X, C), y, runs)
        within_rows.append(row)

    print(f"pair {i}/{len(pairs)}: {a} {b}", flush=True)

transfer = pd.DataFrame(transfer_rows)
within = pd.DataFrame(within_rows)
transfer.to_csv(RESULTS / "shared_dimensions_transfer_by_pair.csv", index=False)
within.to_csv(RESULTS / "shared_dimensions_within_by_space.csv", index=False)


# ---- summary ---------------------------------------------------------------
# Bootstrap: resample participants; for transfer, keep the teacher-learner pairs
# formed by the resampled participants (a participant drawn twice counts twice).
rng = np.random.default_rng(42)
boot_samples = [rng.integers(0, len(subjects), len(subjects)) for _ in range(5000)]
index = {s: n for n, s in enumerate(subjects)}
t_idx = transfer["teacher"].map(index).to_numpy()
l_idx = transfer["learner"].map(index).to_numpy()


def transfer_ci(values):
    boot = []
    for sample in boot_samples:
        count = np.bincount(sample, minlength=len(subjects))
        weights = count[t_idx] * count[l_idx]
        if weights.sum() == 0:   # sample drew a single person: no pairs to average
            continue
        boot.append(np.average(values, weights=weights))
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


def subject_ci(values):
    boot = [values[sample].mean() for sample in boot_samples]
    return values.mean(), *np.percentile(boot, [2.5, 97.5])


within_per_subject = within.drop(columns="space").groupby("subject").mean().loc[subjects]

columns = ["full"] + [f"{rep}_{k}" for rep in ("shared", "residual") for k in K_LIST]
summary = []
for col in columns:
    t_mean, t_low, t_high = transfer_ci(transfer[col].to_numpy())
    w_mean, w_low, w_high = subject_ci(within_per_subject[col].to_numpy())
    summary.append({"representation": col,
                    "transfer": t_mean, "transfer_ci_low": t_low, "transfer_ci_high": t_high,
                    "within": w_mean, "within_ci_low": w_low, "within_ci_high": w_high})
summary = pd.DataFrame(summary)
summary.to_csv(RESULTS / "shared_dimensions_summary.csv", index=False)

print("\nbalanced accuracy (%), mean [95% CI]")
print(f"{'':<14}{'between people':>24}{'within a person':>26}")
for _, r in summary.iterrows():
    print(f"{r['representation']:<14}"
          f"{100 * r['transfer']:>10.1f} [{100 * r['transfer_ci_low']:.1f}, {100 * r['transfer_ci_high']:.1f}]"
          f"{100 * r['within']:>12.1f} [{100 * r['within_ci_low']:.1f}, {100 * r['within_ci_high']:.1f}]")

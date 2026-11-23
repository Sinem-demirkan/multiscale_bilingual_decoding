# Step 3: cross-participant transfer with the full cortical pattern
# (Methods 2.8, Fig. 2B black line; paper: 56.9%, 95% CI [56.0, 57.9]).
#
# For every ordered pair of participants: train on ALL trials of one person
# (the teacher), test on ALL trials of another (the learner).
# 77 x 76 = 5,852 classifiers. Each trial = 800 parcel means.
# Each participant is z-scored once across all of their own trials.
#
# Usage:  python scripts/03_cross_participant_transfer.py   (after step 1)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parents[1] / "results"
data = np.load(RESULTS / "parcels_schaefer800.npz")
subjects = sorted(key[2:] for key in data.files if key.startswith("X_"))

# load everyone, z-score each participant on their own trials
subject_X, subject_y = {}, {}
for s in subjects:
    subject_X[s] = StandardScaler().fit_transform(data[f"X_{s}"])
    subject_y[s] = np.array([t[:2] for t in data[f"trial_type_{s}"]])   # "L1" or "L2"
print(len(subjects), "participants")


# train on teacher, test on every other participant
rows = []
for teacher in subjects:
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(subject_X[teacher], subject_y[teacher])

    for learner in subjects:
        if learner == teacher:
            continue
        y_pred = clf.predict(subject_X[learner])
        rows.append({"teacher": teacher, "learner": learner,
                     "balanced_accuracy": balanced_accuracy_score(subject_y[learner], y_pred)})
    print(teacher, "done")

pairs = pd.DataFrame(rows)
pairs.to_csv(RESULTS / "cross_participant_transfer_by_pair.csv", index=False)


# per participant: as learner (mean over 76 teachers) and as teacher (mean over 76 learners)
per_subject = pd.DataFrame({
    "learner_transfer": pairs.groupby("learner")["balanced_accuracy"].mean(),
    "teacher_transfer": pairs.groupby("teacher")["balanced_accuracy"].mean(),
}).rename_axis("subject")
per_subject.to_csv(RESULTS / "cross_participant_transfer_by_subject.csv")


# group mean with 95% CI: resample participants 5,000 times and keep the
# teacher-learner pairs they form (a participant drawn twice counts twice)
index = {s: n for n, s in enumerate(subjects)}
t_idx = pairs["teacher"].map(index).to_numpy()
l_idx = pairs["learner"].map(index).to_numpy()
acc = pairs["balanced_accuracy"].to_numpy()

rng = np.random.default_rng(42)
boot_means = []
for _ in range(5000):
    count = np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
    weights = count[t_idx] * count[l_idx]
    if weights.sum() == 0:   # sample drew a single person: no pairs to average
        continue
    boot_means.append(np.average(acc, weights=weights))
low, high = np.percentile(boot_means, [2.5, 97.5])

print(f"\n{len(pairs)} teacher -> learner pairs")
print(f"mean transfer = {100 * acc.mean():.1f}%  95% CI [{100 * low:.1f}, {100 * high:.1f}]  (chance = 50%)")

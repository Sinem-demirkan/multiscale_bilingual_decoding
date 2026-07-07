"""
Cross-subject whole-cortex Schaefer parcel-mean transfer decoding.

Rows in the output matrix are teacher/training participants. Columns are
learner/test participants. Off-diagonal entries are cross-subject transfer
accuracies.
"""

# NOTE: teacher_score / learner_score below are simple descriptive averages
# of the raw off-diagonal transfer matrix, NOT the mixed-model teacher/learner
# effects used for the final paper analyses.
#
#   teacher_score for subject i = mean(T_i,j) across all learners j != i
#   learner_score for subject j = mean(T_i,j) across all teachers i != j
#
# These are confounded: a high teacher_score could mean subject i is a good
# "teacher", or that it happened to be paired with easier learners. Same
# issue in reverse for learner_score. The crossed mixed-effects model
# separates teacher identity, learner identity, fixed covariates, and
# pairwise residual variance.

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


# Path to the root of your duo-cogcon dataset folder
# (should contain the sub-XXX single-trial beta images and events.tsv files)
DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
OUT_DIR = Path("/home/sdemirka/fmri/duo-cogcon/outputs")

TASK = "LanguageControl"
TRIAL_COLUMN = "trial_type"
TARGET = "language"  # "language" or "switch"
START_SUB = 1
END_SUB = 2

# Also train and test on the same participant. The paper analyses use
# off-diagonal pairs only, so leave this False unless you specifically want it.
INCLUDE_DIAGONAL = False

N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2
RANDOM_STATE = 42


def subject_id(n):
    return f"sub-{n:03d}"


def make_labels(trial_labels, target):
    trial_labels = pd.Series(trial_labels).astype(str)

    if target == "language":
        labels = trial_labels.map(
            lambda x: "L1" if x.startswith("L1") else ("L2" if x.startswith("L2") else np.nan)
        )
    elif target == "switch":
        labels = trial_labels.map(
            lambda x: "switch" if x.endswith("S") else ("nonswitch" if x.endswith("NS") else np.nan)
        )
    else:
        raise ValueError("TARGET must be 'language' or 'switch'")

    keep = labels.notna().to_numpy()
    return labels.to_numpy(), keep


def transfer_score(train_X, train_y, test_X, test_y):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(train_X)
    X_test_z = scaler.transform(test_X)

    clf = LogisticRegression(
        penalty="l2",
        C=0.01,  # strong L2 regularization; tune here if needed
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_z, train_y)
    y_pred = clf.predict(X_test_z)
    return float(balanced_accuracy_score(test_y, y_pred))


OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build the Schaefer-atlas masker once, reused across subjects.
atlas = fetch_atlas_schaefer_2018(
    n_rois=N_ROIS,
    yeo_networks=YEO_NETWORKS,
    resolution_mm=RESOLUTION_MM,
)

masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=False,
    strategy="mean",
    resampling_target="data",
    verbose=0,
)

subject_data = {}

for sub in range(START_SUB, END_SUB + 1):
    subject = subject_id(sub)

    beta_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    )

    if len(beta_files) == 0 or len(event_files) == 0:
        print(subject, "missing files")
        continue

    beta_by_run = {
        int(p.name.split("run-")[1].split("_")[0]): p
        for p in beta_files
    }
    event_by_run = {
        int(p.name.split("run-")[1].split("_")[0]): p
        for p in event_files
    }
    runs = sorted(set(beta_by_run) & set(event_by_run))

    trial_imgs = []
    labels = []

    for run in runs:
        beta_file = beta_by_run[run]
        event_file = event_by_run[run]

        events = pd.read_csv(event_file, sep="\t")
        img = nib.load(beta_file)

        if TRIAL_COLUMN not in events.columns:
            print(subject, f"missing column {TRIAL_COLUMN} in {event_file.name}")
            continue

        if img.shape[-1] != len(events):
            print(subject, f"run {run} mismatch")
            continue

        y_full, keep = make_labels(events[TRIAL_COLUMN], TARGET)
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            continue

        for idx in keep_idx:
            trial_imgs.append(img.slicer[..., idx])
        labels.extend(y_full[keep_idx].tolist())

    if len(trial_imgs) == 0:
        print(subject, "no usable trials")
        continue

    y = np.asarray(labels)
    if len(np.unique(y)) < 2:
        print(subject, "only one class found")
        continue

    # One mean value per Schaefer parcel per trial -> (n_trials, n_rois) feature matrix.
    X = masker.fit_transform(image.concat_imgs(trial_imgs))

    subject_data[subject] = {"X": X, "y": y}
    print(subject, "loaded", flush=True)

rows = []
subjects = sorted(subject_data)

for teacher in subjects:
    for learner in subjects:
        if teacher == learner and not INCLUDE_DIAGONAL:
            continue

        score = transfer_score(
            subject_data[teacher]["X"],
            subject_data[teacher]["y"],
            subject_data[learner]["X"],
            subject_data[learner]["y"],
        )
        rows.append({
            "train_subject": teacher,
            "test_subject": learner,
            "balanced_accuracy": score,
            "same_subject": teacher == learner,
            "target": TARGET,
            "n_rois": N_ROIS,
            "yeo_networks": YEO_NETWORKS,
        })
    print(teacher, "done", flush=True)

results = pd.DataFrame(rows)
results.to_csv(OUT_DIR / "cross_subject_transfer_long.csv", index=False)

matrix = results.pivot(
    index="train_subject",
    columns="test_subject",
    values="balanced_accuracy",
).reset_index()
matrix.to_csv(OUT_DIR / "cross_subject_transfer_matrix_accuracy.csv", index=False)

off_diag = results.loc[~results["same_subject"]].copy()
diagonal = results.loc[results["same_subject"]].copy()

teacher_scores = (
    off_diag.groupby("train_subject", as_index=False)["balanced_accuracy"]
    .mean()
    .rename(columns={"train_subject": "subject", "balanced_accuracy": "teacher_score"})
)
learner_scores = (
    off_diag.groupby("test_subject", as_index=False)["balanced_accuracy"]
    .mean()
    .rename(columns={"test_subject": "subject", "balanced_accuracy": "learner_score"})
)
teacher_scores.merge(learner_scores, on="subject", how="outer").to_csv(
    OUT_DIR / "cross_subject_teacher_learner_scores.csv",
    index=False,
)

summary = pd.DataFrame([{
    "n_subjects": len(subjects),
    "n_pairs_off_diagonal": len(off_diag),
    "mean_off_diagonal_accuracy": off_diag["balanced_accuracy"].mean(),
    "sd_off_diagonal_accuracy": off_diag["balanced_accuracy"].std(ddof=1),
    "n_pairs_diagonal": len(diagonal),
    "mean_diagonal_accuracy": diagonal["balanced_accuracy"].mean() if len(diagonal) else np.nan,
    "sd_diagonal_accuracy": diagonal["balanced_accuracy"].std(ddof=1) if len(diagonal) else np.nan,
}])
summary.to_csv(OUT_DIR / "cross_subject_transfer_summary.csv", index=False)

results

"""
Within-subject whole-cortex Schaefer parcel-mean decoding.

This script computes the main self-decoding measure used as whole-cortex
decoding strength. It extracts one mean feature per Schaefer parcel from each
single-trial beta image and evaluates L1-vs-L2 language decoding with
leave-one-run-out cross-validation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler





DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")  # Path to the root of your duo-cogcon dataset folder, (should contain the sub-XXX single-trial beta images and events.tsv files)

TASK = "LanguageControl"
START_SUB = 1
END_SUB = 2

TRIAL_COLUMN = "trial_type"
N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2
RANDOM_STATE = 42


def subject_id(n):
    return f"sub-{n:03d}"


def make_labels(trial_labels):
    trial_labels = pd.Series(trial_labels).astype(str)
    labels = trial_labels.map(
        lambda x: "L1" if x.startswith("L1") else ("L2" if x.startswith("L2") else np.nan)
    )
    keep = labels.notna().to_numpy()
    return labels.to_numpy(), keep


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

rows = []
fold_rows = []

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
    groups = []

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

        y_full, keep = make_labels(events[TRIAL_COLUMN])
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            continue

        for idx in keep_idx:
            trial_imgs.append(img.slicer[..., idx])
        labels.extend(y_full[keep_idx].tolist())
        groups.extend([run] * len(keep_idx))

    if len(trial_imgs) == 0:
        print(subject, "no usable trials")
        continue

    y = np.asarray(labels)
    if set(np.unique(y)) != {"L1", "L2"}:
        print(subject, "missing one language class")
        continue

    # One mean value per Schaefer parcel per trial -> (n_trials, n_rois) feature matrix.
    X = masker.fit_transform(image.concat_imgs(trial_imgs))
    groups = np.asarray(groups)

    clf = LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    logo = LeaveOneGroupOut()
    scores = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        score = balanced_accuracy_score(y[test_idx], y_pred)
        scores.append(score)

        fold_rows.append({
            "subject": subject,
            "fold": fold,
            "test_run": int(groups[test_idx][0]),
            "balanced_accuracy": float(score),
        })

    mean_score = float(np.mean(scores))

    rows.append({
        "subject": subject,
        "balanced_accuracy": mean_score,
        "accuracy_minus_chance": mean_score - 0.5,
        "n_trials": int(len(y)),
        "n_L1": int(np.sum(y == "L1")),
        "n_L2": int(np.sum(y == "L2")),
        "n_runs": int(len(np.unique(groups))),
        "n_rois": N_ROIS,
        "yeo_networks": YEO_NETWORKS,
    })
    print(subject, round(mean_score, 5))

results = pd.DataFrame(rows).sort_values("subject") if rows else pd.DataFrame()
folds = pd.DataFrame(fold_rows).sort_values(["subject", "fold"]) if fold_rows else pd.DataFrame()

results.to_csv("distributed_parcel_mean_decoding_by_subject.csv", index=False)
folds.to_csv("distributed_parcel_mean_decoding_folds.csv", index=False)

results

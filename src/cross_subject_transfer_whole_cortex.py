"""
Cross-subject whole-cortex Schaefer-800 parcel-mean transfer decoding for the
Guo et al. bilingual picture-naming dataset (OpenNeuro ds005455).

For each subject, the script extracts Schaefer-800 parcel means from all
single-trial beta images. It then trains a classifier on one subject
(the teacher) and tests it on every other subject (the learner), producing a
77 x 76 cross-subject transfer table of balanced accuracies.

This script is for the descriptive transfer matrix itself. If teacher/learner
effects are later used to predict AoA, those effects should be re-estimated
inside each held-out fold rather than taken directly from the full matrix.
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
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77
TRIAL_COLUMN = "trial_type"

TARGET = "language"   # "language" or "switch"

N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2
RANDOM_STATE = 42


def subject_id(n):
    return f"sub-{n:03d}"


def parse_run_number(path):
    return int(path.name.split("run-")[1].split("_")[0])


def make_labels(trial_labels, target):
    trial_labels = pd.Series(trial_labels).astype(str)

    if target == "language":
        return trial_labels.str.contains("L1").astype(int).to_numpy()
    if target == "switch":
        return trial_labels.str.endswith("S").astype(int).to_numpy()

    raise ValueError("TARGET must be 'language' or 'switch'")


def load_subject_data(subject, masker):
    beta_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    )

    if len(beta_files) == 0 or len(event_files) == 0:
        return None, None

    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}
    runs = sorted(set(beta_by_run) & set(event_by_run))

    trial_imgs = []
    trial_labels = []

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

        for idx in range(img.shape[-1]):
            trial_imgs.append(img.slicer[..., idx])

        trial_labels.extend(events[TRIAL_COLUMN].tolist())

    if len(trial_imgs) == 0:
        return None, None

    X = masker.fit_transform(image.concat_imgs(trial_imgs))
    y = make_labels(trial_labels, TARGET)

    if len(np.unique(y)) < 2:
        print(subject, "only one class found")
        return None, None

    return X, y


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
    X, y = load_subject_data(subject, masker)

    if X is None:
        print(subject, "missing or unusable data")
        continue

    subject_data[subject] = {"X": X, "y": y}
    print(subject, "loaded")


rows = []

for teacher, teacher_data in subject_data.items():
    X_train = teacher_data["X"]
    y_train = teacher_data["y"]

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)

    clf = LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_z, y_train)

    for learner, learner_data in subject_data.items():
        if learner == teacher:
            continue

        X_test = scaler.transform(learner_data["X"])
        y_test = learner_data["y"]
        y_pred = clf.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)

        rows.append({
            "train_subject": teacher,
            "test_subject": learner,
            "balanced_accuracy": score,
            "same_subject": False,
            "target": TARGET,
            "n_rois": N_ROIS,
            "yeo_networks": YEO_NETWORKS,
        })

    print(teacher, "done")


results = pd.DataFrame(rows)
results.to_csv("cross_subject_transfer_long.csv", index=False)
print(results)

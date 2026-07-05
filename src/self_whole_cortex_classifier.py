from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77 # you may change this

TARGET = "language"   # or "switch"
TRIAL_COLUMN = "trial_type"


def subject_id(n):
    return f"sub-{n:03d}"


def relabel_trials(trial_labels, target):
    trial_labels = pd.Series(trial_labels).astype(str)

    if target == "language":
        mapping = {
            "L1S": "L1",
            "L1NS": "L1",
            "L2S": "L2",
            "L2NS": "L2",
        }
    elif target == "switch":
        mapping = {
            "L1S": "S",
            "L2S": "S",
            "L1NS": "NS",
            "L2NS": "NS",
        }
    else:
        raise ValueError("TARGET must be 'language' or 'switch'")

    y = trial_labels.map(mapping)
    keep = y.notna().to_numpy()
    return y, keep


rows = []

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

    imgs = []
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

        y_full, keep = relabel_trials(events[TRIAL_COLUMN], TARGET)
        keep_idx = np.where(keep)[0]

        for idx in keep_idx:
            imgs.append(img.slicer[..., idx])

        labels.extend(y_full.iloc[keep_idx].tolist())
        groups.extend([run] * len(keep_idx))

    if len(imgs) == 0:
        print(subject, "no usable trials")
        continue

    y = np.array(labels)
    if len(np.unique(y)) < 2:
        print(subject, "only one class left after relabeling")
        continue

    masker = NiftiMasker(mask_strategy="background", standardize=False)
    X = masker.fit_transform(imgs)
    groups = np.array(groups)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="liblinear"),
    )

    cv = LeaveOneGroupOut()
    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        groups=groups,
        scoring="balanced_accuracy",
    )

    rows.append({
        "subject": subject,
        "target": TARGET,
        "balanced_accuracy": scores.mean(),
        "n_trials": len(y),
    })
    print(subject, round(scores.mean(), 4))

results = pd.DataFrame(rows)
results.to_csv(f"self_whole_cortex_{TARGET}_results.csv", index=False)
print(results)


"""
Cross-subject whole-cortex Schaefer parcel-mean transfer decoding.

Rows in the output matrix are teacher/training participants. Columns are
learner/test participants. Off-diagonal entries are cross-subject transfer
accuracies.
"""

from __future__ import annotations

import argparse
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


RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Run cross-subject transfer decoding.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", default="LanguageControl")
    parser.add_argument("--trial-column", default="trial_type")
    parser.add_argument("--target", choices=["language", "switch"], default="language")
    parser.add_argument("--start-sub", type=int, default=1)
    parser.add_argument("--end-sub", type=int, default=77)
    parser.add_argument(
        "--include-diagonal",
        action="store_true",
        help="Also train and test on the same participant. The paper analyses use off-diagonal pairs only.",
    )
    parser.add_argument("--n-rois", type=int, default=800)
    parser.add_argument("--yeo-networks", type=int, default=7, choices=[7, 17])
    parser.add_argument("--resolution-mm", type=int, default=2)
    return parser.parse_args()


def subject_id(n: int) -> str:
    return f"sub-{n:03d}"


def parse_run_number(path: Path) -> int:
    return int(path.name.split("run-")[1].split("_")[0])


def make_labels(trial_labels, target: str):
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
        raise ValueError("target must be 'language' or 'switch'")

    keep = labels.notna().to_numpy()
    return labels.to_numpy(), keep


def load_subject_data(subject: str, masker, args):
    beta_files = sorted(
        args.data_root.glob(f"**/{subject}_task-{args.task}_run-*_singletrial-Act.nii.gz")
    )
    event_files = sorted(
        args.data_root.glob(f"**/{subject}_task-{args.task}_run-*_events.tsv")
    )

    if len(beta_files) == 0 or len(event_files) == 0:
        return None, None

    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}
    runs = sorted(set(beta_by_run) & set(event_by_run))

    trial_imgs = []
    labels = []

    for run in runs:
        beta_file = beta_by_run[run]
        event_file = event_by_run[run]

        events = pd.read_csv(event_file, sep="\t")
        img = nib.load(beta_file)

        if args.trial_column not in events.columns:
            print(subject, f"missing column {args.trial_column} in {event_file.name}")
            continue

        if img.shape[-1] != len(events):
            print(subject, f"run {run} mismatch")
            continue

        y_full, keep = make_labels(events[args.trial_column], args.target)
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            continue

        for idx in keep_idx:
            trial_imgs.append(img.slicer[..., idx])
        labels.extend(y_full[keep_idx].tolist())

    if len(trial_imgs) == 0:
        return None, None

    X = masker.fit_transform(image.concat_imgs(trial_imgs))
    y = np.asarray(labels)

    if len(np.unique(y)) < 2:
        print(subject, "only one class found")
        return None, None

    return X, y


def transfer_score(train_X, train_y, test_X, test_y):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(train_X)
    X_test_z = scaler.transform(test_X)

    clf = LogisticRegression(
        penalty="l2",
        C=0.01,
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_z, train_y)
    y_pred = clf.predict(X_test_z)
    return float(balanced_accuracy_score(test_y, y_pred))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    atlas = fetch_atlas_schaefer_2018(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.resolution_mm,
    )

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize=False,
        strategy="mean",
        resampling_target="data",
        verbose=0,
    )

    subject_data = {}
    skipped = []

    for sub in range(args.start_sub, args.end_sub + 1):
        subject = subject_id(sub)
        X, y = load_subject_data(subject, masker, args)

        if X is None:
            skipped.append({"subject": subject, "reason": "missing_or_unusable_data"})
            print(subject, "missing or unusable data")
            continue

        subject_data[subject] = {"X": X, "y": y}
        print(subject, "loaded", flush=True)

    rows = []
    subjects = sorted(subject_data)

    for teacher in subjects:
        for learner in subjects:
            if teacher == learner and not args.include_diagonal:
                continue

            score = transfer_score(
                subject_data[teacher]["X"],
                subject_data[teacher]["y"],
                subject_data[learner]["X"],
                subject_data[learner]["y"],
            )
            rows.append(
                {
                    "train_subject": teacher,
                    "test_subject": learner,
                    "balanced_accuracy": score,
                    "same_subject": teacher == learner,
                    "target": args.target,
                    "n_rois": args.n_rois,
                    "yeo_networks": args.yeo_networks,
                }
            )
        print(teacher, "done", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "cross_subject_transfer_long.csv", index=False)

    matrix = results.pivot(
        index="train_subject",
        columns="test_subject",
        values="balanced_accuracy",
    ).reset_index()
    matrix.to_csv(args.out_dir / "cross_subject_transfer_matrix_accuracy.csv", index=False)

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
        args.out_dir / "cross_subject_teacher_learner_scores.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "n_subjects": len(subjects),
                "n_pairs_off_diagonal": len(off_diag),
                "mean_off_diagonal_accuracy": off_diag["balanced_accuracy"].mean(),
                "sd_off_diagonal_accuracy": off_diag["balanced_accuracy"].std(ddof=1),
                "n_pairs_diagonal": len(diagonal),
                "mean_diagonal_accuracy": diagonal["balanced_accuracy"].mean() if len(diagonal) else np.nan,
                "sd_diagonal_accuracy": diagonal["balanced_accuracy"].std(ddof=1) if len(diagonal) else np.nan,
            }
        ]
    )
    summary.to_csv(args.out_dir / "cross_subject_transfer_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out_dir / "cross_subject_transfer_skipped_subjects.csv", index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUB_RE = re.compile(
    r"(sub-[0-9]+)_task-LanguageControl_run-(0[12])_singletrial-Act\.nii\.gz"
)


def canonical_subject(subject: str) -> str:
    return f"sub-{int(subject.split('-')[1]):03d}"


def language_label(trial_type: str) -> str | float:
    trial_type = str(trial_type)
    if trial_type.startswith("L1"):
        return "L1"
    if trial_type.startswith("L2"):
        return "L2"
    return np.nan


def valid_events_file(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "r") as f:
        return f.readline().startswith("onset\t")


def matching_events_path(data_root: Path, image_path: Path) -> Path:
    return data_root / "events" / image_path.name.replace("_singletrial-Act.nii.gz", "_events.tsv")


def collect_subject_runs(data_root: Path) -> dict[str, dict[str, Path]]:
    subjects: dict[str, dict[str, list[Path]]] = {}
    singletrial_dir = data_root / "singletrial"
    for path in singletrial_dir.glob("*_singletrial-Act.nii.gz"):
        match = SUB_RE.match(path.name)
        if not match:
            continue
        subject, run = match.groups()
        canonical = canonical_subject(subject)
        subjects.setdefault(canonical, {}).setdefault(run, [])
        subjects[canonical][run].append(path)

    chosen: dict[str, dict[str, Path]] = {}
    for canonical, runs in subjects.items():
        chosen[canonical] = {}
        for run, paths in runs.items():
            # Prefer zero-padded filenames if duplicates exist.
            paths = sorted(paths, key=lambda p: (len(p.name), p.name), reverse=True)
            chosen[canonical][run] = paths[0]
    return chosen


def load_template_and_atlas(n_rois: int = 800, resolution_mm: int = 2):
    template = datasets.load_mni152_template(resolution=resolution_mm)
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=7,
        resolution_mm=resolution_mm,
    )
    atlas_img = image.resample_to_img(
        nib.load(atlas.maps),
        template,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16).ravel()
    return template, atlas_img, atlas_data


def load_subject_trials(data_root: Path, subject: str, run_paths: dict[str, Path], template):
    run_arrays = []
    run_labels = []
    run_groups = []

    for run in ["01", "02"]:
        img_path = run_paths.get(run)
        if img_path is None:
            continue
        events_path = matching_events_path(data_root, img_path)
        if not valid_events_file(events_path):
            continue

        img = image.resample_to_img(
            nib.load(img_path),
            template,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
        events = pd.read_csv(events_path, sep="\t")
        events["language"] = events["trial_type"].map(language_label)
        events = events.dropna(subset=["language"]).reset_index(drop=True)

        if len(events) != img.shape[-1]:
            raise ValueError(
                f"{subject} {run}: {len(events)} events but {img.shape[-1]} image volumes"
            )

        data = np.asarray(img.dataobj, dtype=np.float32)
        run_arrays.append(data.reshape(-1, data.shape[-1]).T)
        run_labels.append(events["language"].to_numpy())
        run_groups.append(np.repeat(run, len(events)))

    if len(run_arrays) < 2:
        raise ValueError("fewer_than_two_valid_runs")

    X = np.vstack(run_arrays)
    y = np.concatenate(run_labels)
    groups = np.concatenate(run_groups)
    if set(np.unique(y)) != {"L1", "L2"}:
        raise ValueError(f"labels_{sorted(np.unique(y))}")
    return X, y, groups


def make_estimator(C: float = 0.01):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        ),
    )


def cv_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator):
    cv = LeaveOneGroupOut()
    y_true_all = []
    y_pred_all = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        clf = clone(estimator)
        clf.fit(X[train_idx], y[train_idx])
        yhat = clf.predict(X[test_idx])
        y_true = y[test_idx]
        bal_acc = balanced_accuracy_score(y_true, yhat)
        fold_rows.append(
            {
                "fold": fold,
                "test_run": str(np.unique(groups[test_idx])[0]),
                "balanced_accuracy": float(bal_acc),
            }
        )
        y_true_all.append(y_true)
        y_pred_all.append(yhat)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return y_true_all, y_pred_all, fold_rows


def balanced_accuracy_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def l1_l2_sensitivity(y_true: np.ndarray, y_pred: np.ndarray):
    y_true_bin = (y_true == "L1").astype(int)
    y_pred_bin = (y_pred == "L1").astype(int)
    l1 = recall_score(y_true_bin, y_pred_bin, pos_label=1)
    l2 = recall_score(y_true_bin, y_pred_bin, pos_label=0)
    return float(l1), float(l2)


"""
Within-subject parcelwise Schaefer-800 decoding for the Guo et al. bilingual
picture-naming dataset (OpenNeuro ds005455).

This script reproduces the local Schaefer-800 multivoxel analysis. For each
subject, single-trial beta images are resampled to MNI space, and decoding is
run separately within each Schaefer parcel using leave-one-run-out
cross-validation.

The output is:
- schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv

This file can then be used to compute subject-level local extent measures.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77
TRIAL_COLUMN = "trial_type"

N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2
RANDOM_STATE = 42

OUT_CSV = "schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv"
SKIPPED_CSV = "schaefer800_parcelwise_multivoxel_skipped_subjects.csv"


def subject_id(n):
  return f"sub-{n:03d}"


def parse_run_number(path):
  return int(path.name.split("run-")[1].split("_")[0])


def language_label(trial_type):
  trial_type = str(trial_type)
  if trial_type.startswith("L1"):
      return "L1"
  if trial_type.startswith("L2"):
      return "L2"
  return np.nan


def cv_balanced_accuracy(X, y, run_groups, estimator):
  cv = LeaveOneGroupOut()
  scores = []

  for train_idx, test_idx in cv.split(X, y, groups=run_groups):
      clf = clone(estimator)
      clf.fit(X[train_idx], y[train_idx])
      yhat = clf.predict(X[test_idx])
      scores.append(balanced_accuracy_score(y[test_idx], yhat))

  return float(np.mean(scores))


def load_subject_data(subject):
  beta_files = sorted(
      DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
  )
  event_files = sorted(
      DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
  )

  if len(beta_files) == 0 or len(event_files) == 0:
      return None, None, None

  beta_by_run = {parse_run_number(p): p for p in beta_files}
  event_by_run = {parse_run_number(p): p for p in event_files}
  runs = sorted(set(beta_by_run) & set(event_by_run))

  template = datasets.load_mni152_template(resolution=RESOLUTION_MM)

  run_arrays = []
  run_labels = []
  run_groups = []

  for run in runs:
      beta_file = beta_by_run[run]
      event_file = event_by_run[run]

      img = image.resample_to_img(
          nib.load(beta_file),
          template,
          interpolation="continuous",
          force_resample=True,
          copy_header=True,
      )
      events = pd.read_csv(event_file, sep="\t")

      if TRIAL_COLUMN not in events.columns:
          print(subject, f"missing column {TRIAL_COLUMN} in {event_file.name}")
          continue

      events["language"] = events[TRIAL_COLUMN].map(language_label)
      events = events.dropna(subset=["language"]).reset_index(drop=True)

      if len(events) != img.shape[-1]:
          print(subject, f"run {run} mismatch")
          continue

      data = np.asarray(img.dataobj, dtype=np.float32)
      run_arrays.append(data.reshape(-1, data.shape[-1]).T)
      run_labels.append(events["language"].to_numpy())
      run_groups.append(np.repeat(run, len(events)))

  if len(run_arrays) < 2:
      return None, None, None

  X = np.vstack(run_arrays)
  y = np.concatenate(run_labels)
  groups = np.concatenate(run_groups)

  if set(np.unique(y)) != {"L1", "L2"}:
      return None, None, None

  return X, y, groups


template = datasets.load_mni152_template(resolution=RESOLUTION_MM)
atlas = datasets.fetch_atlas_schaefer_2018(
  n_rois=N_ROIS,
  yeo_networks=YEO_NETWORKS,
  resolution_mm=RESOLUTION_MM,
)
atlas_img = image.resample_to_img(
  nib.load(atlas.maps),
  template,
  interpolation="nearest",
  force_resample=True,
  copy_header=True,
)
atlas_data = np.asarray(atlas_img.dataobj).astype(np.int16).ravel()

clf = make_pipeline(
  StandardScaler(),
  LogisticRegression(
      penalty="l2",
      C=0.01,
      solver="lbfgs",
      max_iter=1000,
      random_state=RANDOM_STATE,
  ),
)

rows = []
skipped = []

for sub in range(START_SUB, END_SUB + 1):
  subject = subject_id(sub)
  X, y, groups = load_subject_data(subject)

  if X is None:
      skipped.append({"subject": subject, "reason": "missing_or_unusable_data"})
      print(subject, "missing or unusable data")
      continue

  print(
      f"Running {subject}: {len(y)} trials, "
      f"L1={np.sum(y == 'L1')}, L2={np.sum(y == 'L2')}"
  )

  for parcel in range(1, N_ROIS + 1):
      vox = np.flatnonzero(atlas_data == parcel)

      if len(vox) < 2:
          acc = np.nan
      else:
          acc = cv_balanced_accuracy(X[:, vox], y, groups, clf)

      rows.append({
          "subject": subject,
          "parcel": parcel,
          "accuracy": acc,
          "accuracy_minus_chance": acc - 0.5 if np.isfinite(acc) else np.nan,
          "n_voxels": int(len(vox)),
          "n_trials": int(len(y)),
          "n_L1": int(np.sum(y == "L1")),
          "n_L2": int(np.sum(y == "L2")),
      })

  pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
pd.DataFrame(skipped).to_csv(SKIPPED_CSV, index=False)

print(f"Wrote {OUT_CSV}")

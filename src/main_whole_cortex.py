"""
Within-subject whole-cortex Schaefer-800 parcel-mean decoding for the Guo et al.
bilingual picture-naming dataset (OpenNeuro ds005455).

This script reproduces the main self whole-cortex language decoding analysis:
single-trial beta images are converted to Schaefer-800 parcel means, and a
logistic regression classifier is evaluated within subject using leave-one-run-out
cross-validation.
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


DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77
TRIAL_COLUMN = "trial_type"

N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2
RANDOM_STATE = 42


def subject_id(n):
  return f"sub-{n:03d}"


def parse_run_number(path):
  return int(path.name.split("run-")[1].split("_")[0])


def make_labels(trial_labels):
  trial_labels = pd.Series(trial_labels).astype(str)
  return trial_labels.str.contains("L1").astype(int).to_numpy()


def load_subject_data(subject, masker):
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

  trial_imgs = []
  trial_labels = []
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

      for idx in range(img.shape[-1]):
          trial_imgs.append(img.slicer[..., idx])

      trial_labels.extend(events[TRIAL_COLUMN].tolist())
      groups.extend([run] * len(events))

  if len(trial_imgs) == 0:
      return None, None, None

  X = masker.fit_transform(image.concat_imgs(trial_imgs))
  y = make_labels(trial_labels)
  groups = np.array(groups)

  return X, y, groups


def run_subject_cv(X, y, groups):
  clf = LogisticRegression(
      penalty="l2",
      C=0.01,
      solver="lbfgs",
      max_iter=1000,
      random_state=RANDOM_STATE,
  )

  logo = LeaveOneGroupOut()
  scores = []

  for train_idx, test_idx in logo.split(X, y, groups=groups):
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X[train_idx])
      X_test = scaler.transform(X[test_idx])

      clf.fit(X_train, y[train_idx])
      y_pred = clf.predict(X_test)
      score = balanced_accuracy_score(y[test_idx], y_pred)
      scores.append(score)

  return float(np.mean(scores))


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

for sub in range(START_SUB, END_SUB + 1):
  subject = subject_id(sub)
  X, y, groups = load_subject_data(subject, masker)

  if X is None:
      print(subject, "missing or unusable data")
      continue

  score = run_subject_cv(X, y, groups)

  rows.append({
      "subject": subject,
      "balanced_accuracy": score,
      "n_trials": len(y),
      "n_runs": len(np.unique(groups)),
  })
  print(subject, round(score, 5))

results = pd.DataFrame(rows)
results.to_csv("distributed_parcel_mean_decoding_by_subject.csv", index=False)
print(results)


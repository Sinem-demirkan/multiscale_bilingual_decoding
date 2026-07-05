"""
Minimal within-subject Schaefer-parcel decoding script for the Guo et al.
bilingual picture-naming dataset (OpenNeuro ds005455).

For each subject, the script loads single-trial beta images, converts them to
Schaefer parcel means, relabels trials for language or switch decoding, and
evaluates either a raw parcel-mean model or a PCA-reduced parcel-mean model
using leave-one-run-out cross-validation.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77  # change if needed

TARGET = "language"   # or "switch"
TRIAL_COLUMN = "trial_type"

# Main atlas options
N_ROIS = 800
YEO_NETWORKS = 7
RESOLUTION_MM = 2

# Variant options
FEATURE_VARIANT = "raw"   # "raw" or "pca"
PCA_COMPONENTS = 0.90     # proportion variance kept if FEATURE_VARIANT == "pca"


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


atlas = fetch_atlas_schaefer_2018(
  n_rois=N_ROIS,
  yeo_networks=YEO_NETWORKS,
  resolution_mm=RESOLUTION_MM,
)

masker = NiftiLabelsMasker(
  labels_img=atlas.maps,
  standardize=False,
)

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

  X_runs = []
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
      if len(keep_idx) == 0:
          continue

      run_imgs = [img.slicer[..., idx] for idx in keep_idx]
      X_run = masker.fit_transform(run_imgs)

      X_runs.append(X_run)
      labels.extend(y_full.iloc[keep_idx].tolist())
      groups.extend([run] * len(keep_idx))

  if len(X_runs) == 0:
      print(subject, "no usable trials")
      continue

  X = np.vstack(X_runs)
  y = np.array(labels)
  groups = np.array(groups)

  if len(np.unique(y)) < 2:
      print(subject, "only one class left after relabeling")
      continue

  if FEATURE_VARIANT == "raw":
      clf = make_pipeline(
          StandardScaler(),
          LogisticRegression(max_iter=5000, solver="liblinear"),
      )
  elif FEATURE_VARIANT == "pca":
      clf = make_pipeline(
          StandardScaler(),
          PCA(n_components=PCA_COMPONENTS),
          LogisticRegression(max_iter=5000, solver="liblinear"),
      )
  else:
      raise ValueError("FEATURE_VARIANT must be 'raw' or 'pca'")

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
      "n_rois": N_ROIS,
      "feature_variant": FEATURE_VARIANT,
      "balanced_accuracy": scores.mean(),
      "n_trials": len(y),
  })
  print(subject, round(scores.mean(), 4))

results = pd.DataFrame(rows)
results.to_csv(
  f"self_schaefer{N_ROIS}_{FEATURE_VARIANT}_{TARGET}_results.csv",
  index=False,
)
print(results)


from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_ROOT = Path("/path/to/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77

# Choose target:
# "language" -> L1 vs L2
# "switch"   -> S vs NS
TARGET = "language"

TRIAL_COLUMN = "trial_type"   # change if needed


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
  (DATA_ROOT / "derivatives" / "singletrial").glob(
      f"{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz"
  )
)
event_files = sorted(
  (DATA_ROOT / "events").glob(
      f"{subject}_task-{TASK}_run-*_events.tsv"
  )
)

if len(beta_files) == 0 or len(event_files) == 0:
  print(subject, "missing files")
  continue

imgs = []
labels = []
groups = []

for beta_file, event_file in zip(beta_files, event_files):
  run = int(beta_file.name.split("run-")[1].split("_")[0])

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

  imgs.extend(nib.four_to_three(img.slicer[..., keep_idx]))
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

mean_score = scores.mean()
rows.append(
  {
      "subject": subject,
      "target": TARGET,
      "balanced_accuracy": mean_score,
      "n_trials": len(y),
      "classes": ",".join(sorted(np.unique(y))),
  }
)
print(subject, round(mean_score, 4))

results = pd.DataFrame(rows)

if TARGET == "language":
out_csv = "self_whole_cortex_language_results.csv"
else:
out_csv = "self_whole_cortex_switch_results.csv"

results.to_csv(out_csv, index=False)
print(results)

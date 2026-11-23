# Step 1: turn each participant's single-trial images into a trials x 800 table
# (Methods 2.5). This is the slow step, so we do it once and save the result.
#
# For every participant we save:
#   X          (n_trials, 800)  mean activity in each Schaefer parcel, NOT z-scored
#   run        (n_trials,)      run number of each trial (for leave-one-run-out)
#   trial_type (n_trials,)      L1S, L1NS, L2S or L2NS
#
# Usage:  python scripts/01_extract_parcels.py

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")   # folder with the ds005455 single-trial files
OUT_FILE = Path(__file__).resolve().parents[1] / "results" / "parcels_schaefer800.npz"

TASK = "LanguageControl"
TRIAL_COLUMN = "trial_type"
N_PARCELS = 800


# filename parser: ".../sub-001_task-LanguageControl_run-03_events.tsv" -> 3
def parse_run_number(path):
    after_run = path.name.split("run-")[1]   # "03_events.tsv"
    return int(after_run.split("_")[0])      # 3


# atlas + masker: average the voxels inside each of the 800 parcels
atlas = fetch_atlas_schaefer_2018(n_rois=N_PARCELS, yeo_networks=7, resolution_mm=2)
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=False,          # keep raw values, the decoding scripts z-score
    strategy="mean",
    resampling_target="data",   # resample the atlas to the beta images' grid
    verbose=0,
)


def load_subject_data(subject):
    # 1. find this subject's beta and events files, matched by run number
    beta_files = DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_singletrial-Act.nii.gz")
    event_files = DATA_ROOT.glob(f"**/{subject}_task-{TASK}_run-*_events.tsv")
    beta_by_run = {parse_run_number(p): p for p in beta_files}
    event_by_run = {parse_run_number(p): p for p in event_files}
    runs = sorted(set(beta_by_run) & set(event_by_run))   # runs that have both

    trial_imgs, run_of_trial, trial_types = [], [], []
    for run in runs:
        events = pd.read_csv(event_by_run[run], sep="\t")
        img = nib.load(beta_by_run[run])   # 4D image: one volume per trial

        # 2. safety check: one volume per event
        if img.shape[-1] != len(events):
            print(subject, "run", run, "mismatch:", img.shape[-1], "volumes vs", len(events), "events")
            continue

        # 3. keep the naming trials (L1S, L1NS, L2S, L2NS)
        for i, trial_type in enumerate(events[TRIAL_COLUMN].astype(str)):
            if trial_type.startswith(("L1", "L2")):
                trial_imgs.append(img.slicer[..., i])
                run_of_trial.append(run)
                trial_types.append(trial_type)

    if not trial_imgs:
        return None

    # 4. brain images -> parcel means
    X = masker.fit_transform(image.concat_imgs(trial_imgs))   # (n_trials, 800)
    return X, np.array(run_of_trial), np.array(trial_types)


# run it for every participant
subjects = sorted({p.name.split("_")[0] for p in DATA_ROOT.glob(f"**/*_task-{TASK}_*_events.tsv")})
print(len(subjects), "participants found")

saved = {}
for s in subjects:
    result = load_subject_data(s)
    if result is None:
        print(s, "no files found")
        continue
    X, run, trial_type = result
    saved[f"X_{s}"] = X
    saved[f"run_{s}"] = run
    saved[f"trial_type_{s}"] = trial_type
    print(s, X.shape)

OUT_FILE.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT_FILE, **saved)
print("saved", OUT_FILE)

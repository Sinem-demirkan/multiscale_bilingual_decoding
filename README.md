# Multiscale Bilingual Decoding

This repository contains the core code used to compute the two participant-level measures from our bilingual fMRI decoding study:

- `cortical extent`: local parcelwise language decoding summarized as the proportion of parcels carrying reliable language information
- `cortical strength`: whole-cortex language decoding summarized as cross-run balanced accuracy

The code expects the single-trial beta images and event files from the Guo et al. Chinese-English bilingual picture-naming dataset, organized as:

```text
DATA_ROOT/
  singletrial/
    sub-001_task-LanguageControl_run-01_singletrial-Act.nii.gz
    sub-001_task-LanguageControl_run-02_singletrial-Act.nii.gz
    ...
  events/
    sub-001_task-LanguageControl_run-01_events.tsv
    sub-001_task-LanguageControl_run-02_events.tsv
    ...
```

## What the scripts do

### 1. Local extent

`src/compute_local_extent.py`:

- projects single-trial beta estimates onto the Schaefer-800 atlas
- trains one parcelwise logistic-regression classifier per parcel
- evaluates each parcel with leave-one-run-out cross-validation
- saves parcelwise balanced accuracies
- summarizes participant-level local extent in two ways:
  - `prop_accuracy_gt_0p60`: proportion of parcels with balanced accuracy > 0.60
  - `prop_z_gt_1p64`: proportion of parcels with one-sided binomial `z > 1.64`, matching the paper release

### 2. Global strength

`src/compute_global_strength.py`:

- computes one mean-activity feature per parcel
- trains one whole-cortex logistic-regression classifier per participant
- evaluates it with leave-one-run-out cross-validation
- saves participant-level balanced accuracy as `cortical strength`

## Installation

Create an environment with the dependencies in `requirements.txt`.

## Included files

```text
multiscale_bilingual_decoding/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  configs/
    example_paths.yaml
  src/
    common.py
    compute_local_extent.py
    compute_global_strength.py
```

## Example commands

Run the local measure:

```bash
python src/compute_local_extent.py \
  --data-root /path/to/ds005455_data \
  --out-dir outputs/local
```

Run the global measure:

```bash
python src/compute_global_strength.py \
  --data-root /path/to/ds005455_data \
  --out-dir outputs/global
```

Run a small subset of subjects:

```bash
python src/compute_local_extent.py \
  --data-root /path/to/ds005455_data \
  --out-dir outputs/local \
  --subjects sub-001 sub-002
```

## Suggested repository usage

- keep raw data outside this repository
- save generated outputs under a separate `outputs/` directory
- version only the code, documentation, and lightweight configs

## Main outputs

### Local extent

- `parcelwise_local_accuracy_long.csv`
- `local_extent_by_subject.csv`
- `local_extent_summary.csv`
- `skipped_subjects.csv`

### Global strength

- `whole_cortex_strength_by_subject.csv`
- `whole_cortex_strength_folds.csv`
- `whole_cortex_strength_summary.csv`
- `skipped_subjects.csv`

## Notes

- Language labels are derived from `trial_type`: entries beginning with `L1` are labeled `L1`, and entries beginning with `L2` are labeled `L2`.
- Cross-run evaluation means one run is used for training and the other for testing, then the folds are averaged.
- The local `z > 1.64` extent measure uses a one-sided binomial p-value derived from the observed parcelwise decoding score, matching the analysis release.

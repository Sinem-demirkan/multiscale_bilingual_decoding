# Multiscale Bilingual Decoding

This repository contains the analysis code for the multiscale bilingual fMRI
decoding study. Each script is configured by editing a small block of
variables near the top of the file (`DATA_ROOT`, `OUT_DIR`, `TASK`, subject
range, etc.).

## Data

This code uses the single-trial beta images and event files from the publicly available **Guo et al. Chinese–English bilingual picture-naming dataset**:

> Guo, T., Liu, X., Chen, M., Fu, Y., & Guo, T. (2025). *An fMRI dataset for investigating language control and cognitive control in bilinguals*. *Scientific Data, 12*(1). 

The dataset is available from **OpenNeuro**:

* **Dataset:** ds005455
* **DOI:** https://doi.org/10.18112/openneuro.ds005455.v1.1.5

The dataset is **not included** in this repository. Please download it from OpenNeuro.

After downloading, keep the dataset in any local directory and point `DATA_ROOT`
at it in each script. The scripts search recursively for matching beta and
event files, so the dataset can use the OpenNeuro directory layout. The
required files are:

```text
duo-cogcon/
├── derivatives/
│   └── singletrial/
│       ├── sub-001_task-LanguageControl_run-01_singletrial-Act.nii.gz
│       ├── sub-001_task-LanguageControl_run-02_singletrial-Act.nii.gz
│       └── ...
└── events/
    ├── sub-001_task-LanguageControl_run-01_events.tsv
    ├── sub-001_task-LanguageControl_run-02_events.tsv
    └── ...
```

Steps 4 and 5 additionally expect a directory of per-run tSNR maps, named:

```text
duo-cogcon/derivatives/tsnr/
├── sub-001_task-LanguageControl_run-01_tsnr.nii.gz
├── sub-001_task-LanguageControl_run-02_tsnr.nii.gz
└── ...
```

## What the scripts do

The main scripts should be run in this order.

### 1. Whole-cortex self-decoding

Open `src/whole_cortex_main_model.py` and set:

```python
DATA_ROOT = Path("/path/to/duo-cogcon")
TASK = "LanguageControl"
START_SUB = 1
END_SUB = 77
```

Then run:

```bash
python src/whole_cortex_main_model.py
```

This extracts Schaefer parcel means from each single-trial beta image and runs
within-subject leave-one-run-out L1-vs-L2 decoding.

Outputs (written to the current working directory):

- `distributed_parcel_mean_decoding_by_subject.csv`
- `distributed_parcel_mean_decoding_folds.csv`

The subject-level file is the self-decoding reference used by later transfer
models. Note this script does not have an `OUT_DIR` variable, unlike the
others below — run it from whatever directory you want the CSVs to land in,
or move them afterward.

### 2. Local parcelwise decoding and local extent

Open `src/local_parcelwise_multivoxel.py` and set `DATA_ROOT`, `OUT_DIR`,
`START_SUB`/`END_SUB` as needed, then run:

```bash
python src/local_parcelwise_multivoxel.py
```

This runs one classifier per Schaefer parcel and summarizes local cortical
extent as the proportion of parcels with one-sided binomial `z > 1.64`.

Outputs:

- `outputs/schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv`
- `outputs/local_z_extent_by_subject.csv`

### 3. Cross-subject transfer matrix

Open `src/whole_cortex_cross_subject_transfer.py` and set `DATA_ROOT`,
`OUT_DIR`, `TARGET`, `INCLUDE_DIAGONAL`, etc., then run:

```bash
python src/whole_cortex_cross_subject_transfer.py
```

Rows are teacher/training participants and columns are learner/test
participants. With `INCLUDE_DIAGONAL = False` (the default), the script writes
only off-diagonal cross-subject transfer values. The within-subject
self-decoding reference comes from step 1.

Outputs:

- `outputs/cross_subject_transfer_long.csv`
- `outputs/cross_subject_transfer_matrix_accuracy.csv`
- `outputs/cross_subject_teacher_learner_scores.csv`
- `outputs/cross_subject_transfer_summary.csv`

`cross_subject_teacher_learner_scores.csv` contains simple descriptive
averages of the raw off-diagonal transfer matrix (mean accuracy per subject as
teacher, and separately as learner). These are confounded — a high
teacher_score can reflect either a good "teacher" or an easy set of
learner-pairings, and vice versa — so treat them as a QC/descriptive
convenience only. Use the crossed mixed-effects teacher/learner effects from
step 4 for any adjusted result reported in the paper.

### 4. Mixed-effects transfer models

The mixed-effects transfer models connect the cross-subject transfer output
from step 3 with the self-decoding output from step 1 and the per-subject
mean tSNR computed from the tSNR maps described above.

Open `src/transfer_mixed_models.py` and set:

```python
TRANSFER = Path("outputs/cross_subject_transfer_long.csv")
SELF_DECODING = Path("distributed_parcel_mean_decoding_by_subject.csv")
TSNR_DIR = Path("/path/to/duo-cogcon/derivatives/tsnr")
TASK = "LanguageControl"
OUT_DIR = Path("outputs/mixed_models")
```

Then run:

```bash
python src/transfer_mixed_models.py
```

Outputs:

- `outputs/mixed_models/mean_tsnr_by_subject.csv` (the computed QC table)
- `outputs/mixed_models/mixed_transfer_design.csv`
- `outputs/mixed_models/mixed_unadjusted_variance_components.csv`
- `outputs/mixed_models/mixed_unadjusted_subject_effects_long.csv`
- `outputs/mixed_models/mixed_variance_components.csv`
- `outputs/mixed_models/mixed_subject_effects_long.csv`
- `outputs/mixed_models/mixed_fixed_effects.csv`

The unadjusted model estimates teacher, learner, and pairwise residual variance
from transfer accuracy. The adjusted model additionally includes teacher and
learner self-decoding accuracy, teacher and learner mean tSNR, and their
side-specific interactions.

These models are fit from Python using `pymer4` (imported at the top of the
script, so pymer4 and its R backend must be installed and working before you
run or import this file), so the analysis can be run from the Python pipeline
while retaining formula-based mixed-effects syntax.

### 5. Nested AoA prediction from transfer effects

Age-of-acquisition prediction uses leave-one-subject-out nesting. For each held
out participant, the mixed transfer model is fit only on transfer pairs among
the remaining training participants. The held-out participant's teacher and
learner effects are then estimated only from transfer pairs between that
participant and the training participants. Finally, AoA prediction models are
fit on the training participants and evaluated on the held-out participant.

Open `src/age_of_acquisition_prediction.py` and set:

```python
PARTICIPANTS = Path("/path/to/duo-cogcon/participants.tsv")
AOA_COLUMN = None  # auto-detects AoA / aoa 
TRANSFER = Path("outputs/cross_subject_transfer_long.csv")
SELF_DECODING = Path("distributed_parcel_mean_decoding_by_subject.csv")
LOCAL_EXTENT = None  # or Path("outputs/local_z_extent_by_subject.csv") to include that baseline
TSNR_DIR = Path("/path/to/duo-cogcon/derivatives/tsnr")
TASK = "LanguageControl"
OUT_DIR = Path("outputs/nested_aoa")
```

Then run:

```bash
python src/age_of_acquisition_prediction.py
```


Outputs:

- `outputs/nested_aoa/age_of_acquisition_nested_mixed_predictions_wide.csv`
- `outputs/nested_aoa/age_of_acquisition_nested_mixed_predictions_long.csv`
- `outputs/nested_aoa/age_of_acquisition_nested_mixed_summary.csv`
- `outputs/nested_aoa/nested_transfer_effects_by_fold.csv`

This script also imports `pymer4` at the top, so it requires the same
pymer4/R setup as step 4.

### Optional scripts

- `src/local_searchlight_multivoxel.py` runs the voxelwise searchlight version
  of local decoding. Defaults: `DATA_ROOT = Path("duo-cogcon")`,
  `OUT_DIR = Path("outputs/local_searchlight_multivoxel")`.
- `src/whole_cortex_classifier_variants.py` contains atlas-size, PCA, and
  grouped permutation feature-importance variants, selected via the `ANALYSIS`
  variable (`"whole_cortex"`, `"size_matched_grouped_pfi"`, or
  `"network_pca_grouped_pfi"`). Defaults: `DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")`,
  `OUT_DIR = Path("/home/sdemirka/fmri/duo-cogcon/outputs")`.

As with the main pipeline scripts above, edit the config variables at the top
of each file before running.

## Notes

- Language labels are derived from `trial_type`: entries beginning with `L1` are labeled `L1`, and entries beginning with `L2` are labeled `L2`.
- Cross-run evaluation means one run is used for training and the other for testing, then the folds are averaged.
- The local `z > 1.64` extent measure uses a one-sided binomial p-value derived from the observed parcelwise decoding score.
- Cross-subject transfer analyses should use off-diagonal pairs. Self-decoding
  references should come from the whole-cortex self-decoding script, not from
  training and testing a cross-subject model on the same participant.

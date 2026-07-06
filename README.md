# Multiscale Bilingual Decoding

This repository contains the analysis code for the multiscale bilingual fMRI
decoding study. The scripts are written so that all paths are supplied at run
time; users should not need to edit hardcoded local directories.

## Data

This code uses the single-trial beta images and event files from the publicly available **Guo et al. Chinese–English bilingual picture-naming dataset**:

> Guo, T., Liu, X., Chen, M., Fu, Y., & Guo, T. (2025). *An fMRI dataset for investigating language control and cognitive control in bilinguals*. *Scientific Data, 12*(1). 

The dataset is available from **OpenNeuro**:

* **Dataset:** ds005455
* **DOI:** https://doi.org/10.18112/openneuro.ds005455.v1.1.5

The dataset is **not included** in this repository. Please download it from OpenNeuro.

After downloading, keep the dataset in any local directory and pass that
directory with `--data-root`. The scripts search recursively for matching beta
and event files, so the dataset can use the OpenNeuro directory layout. The
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

## What the scripts do

The main scripts should be run in this order.

### 1. Whole-cortex self-decoding

```bash
python src/whole_cortex_main_model.py \
  --data-root /path/to/duo-cogcon \
  --out-dir outputs/whole_cortex
```

This extracts Schaefer parcel means from each single-trial beta image and runs
within-subject leave-one-run-out L1-vs-L2 decoding.

Outputs:

- `outputs/whole_cortex/distributed_parcel_mean_decoding_by_subject.csv`
- `outputs/whole_cortex/distributed_parcel_mean_decoding_folds.csv`
- `outputs/whole_cortex/distributed_parcel_mean_skipped_subjects.csv`

The subject-level file is the self-decoding reference used by later transfer
models.

### 2. Local parcelwise decoding and local extent

```bash
python src/local_parcelwise_multivoxel.py \
  --data-root /path/to/duo-cogcon \
  --out-dir outputs/local_parcelwise
```

This runs one classifier per Schaefer parcel and summarizes local cortical
extent as the proportion of parcels with one-sided binomial `z > 1.64`.

Outputs:

- `outputs/local_parcelwise/schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv`
- `outputs/local_parcelwise/local_z_extent_by_subject.csv`
- `outputs/local_parcelwise/schaefer800_parcelwise_multivoxel_skipped_subjects.csv`

### 3. Cross-subject transfer matrix

```bash
python src/whole_cortex_cross_subject_transfer.py \
  --data-root /path/to/duo-cogcon \
  --out-dir outputs/cross_subject_transfer
```

Rows are teacher/training participants and columns are learner/test
participants. By default, the script writes only off-diagonal cross-subject
transfer values. The within-subject self-decoding reference comes from step 1.

Outputs:

- `outputs/cross_subject_transfer/cross_subject_transfer_long.csv`
- `outputs/cross_subject_transfer/cross_subject_transfer_matrix_accuracy.csv`
- `outputs/cross_subject_transfer/cross_subject_teacher_learner_scores.csv`
- `outputs/cross_subject_transfer/cross_subject_transfer_summary.csv`

### 4. Mixed-effects transfer models

The mixed-effects transfer models connect the cross-subject transfer output
from step 3 with the self-decoding output from step 1 and a subject-level QC
file containing `subject` and `mean_tsnr`.

```bash
Rscript src/transfer_lme4_models.R \
  --transfer outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --self-decoding outputs/whole_cortex/distributed_parcel_mean_decoding_by_subject.csv \
  --qc /path/to/languagecontrol_qc_by_subject.csv \
  --out-dir outputs/transfer_lme4
```

Outputs:

- `outputs/transfer_lme4/lme4_unadjusted_variance_components.csv`
- `outputs/transfer_lme4/lme4_unadjusted_subject_effects_long.csv`
- `outputs/transfer_lme4/lme4_variance_components.csv`
- `outputs/transfer_lme4/lme4_subject_effects_long.csv`
- `outputs/transfer_lme4/lme4_fixed_effects.csv`

The unadjusted model estimates teacher, learner, and pairwise residual variance
from transfer accuracy. The adjusted model additionally includes teacher and
learner self-decoding accuracy, teacher and learner mean tSNR, and their
side-specific interactions.

### 5. Nested AoA prediction from transfer effects

Age-of-acquisition prediction uses leave-one-subject-out nesting. For each held
out participant, the mixed transfer model is fit only on transfer pairs among
the remaining training participants. The held-out participant's teacher and
learner effects are then estimated only from transfer pairs between that
participant and the training participants. Finally, AoA prediction models are
fit on the training participants and evaluated on the held-out participant.

```bash
Rscript src/age_of_acquisition_prediction \
  --data-root /path/to/duo-cogcon \
  --transfer outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --self-decoding outputs/whole_cortex/distributed_parcel_mean_decoding_by_subject.csv \
  --local-extent outputs/local_parcelwise/local_z_extent_by_subject.csv \
  --qc /path/to/languagecontrol_qc_by_subject.csv \
  --out-dir outputs/age_of_acquisition
```

The script reads AoA from `participants.tsv` in the dataset root. If the AoA
column has a nonstandard name, add `--aoa-column column_name`. You can also pass
`--participants /path/to/participants.tsv` instead of `--data-root`.

Outputs:

- `outputs/age_of_acquisition/age_of_acquisition_nested_lme4_predictions_wide.csv`
- `outputs/age_of_acquisition/age_of_acquisition_nested_lme4_predictions_long.csv`
- `outputs/age_of_acquisition/age_of_acquisition_nested_lme4_summary.csv`
- `outputs/age_of_acquisition/nested_transfer_effects_by_fold.csv`

### Optional scripts

- `src/local_searchlight_multivoxel.py` runs the voxelwise searchlight version
  of local decoding. It defaults to `DATA_ROOT = Path("duo-cogcon")` and
  `OUT_DIR = Path("outputs/local_searchlight_multivoxel")`; edit these two
  constants if needed.
- `src/whole_cortex_classifier_variants.py` contains atlas-size, PCA, and
  grouped permutation feature-importance variants. It defaults to
  `DATA_ROOT = Path("duo-cogcon")` and writes to
  `outputs/whole_cortex_classifier_variants`.

## Notes

- Language labels are derived from `trial_type`: entries beginning with `L1` are labeled `L1`, and entries beginning with `L2` are labeled `L2`.
- Cross-run evaluation means one run is used for training and the other for testing, then the folds are averaged.
- The local `z > 1.64` extent measure uses a one-sided binomial p-value derived from the observed parcelwise decoding score.
- Cross-subject transfer analyses should use off-diagonal pairs. Self-decoding
  references should come from the whole-cortex self-decoding script, not from
  training and testing a cross-subject model on the same participant.

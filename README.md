# Multiscale Bilingual Decoding

Code for the manuscript **What Transfers Across Brains Is Not What Stays Within
Individuals: Evidence from Bilingual Language States**.

The code is meant for someone starting from the Guo et al. OpenNeuro dataset
(`ds005455`) and its single-trial derivative images. The data are not included
here.

## Setup

```bash
pip install -r requirements.txt
```

Use the local path to the downloaded dataset wherever `/path/to/ds005455`
appears below.

## Main Analysis Order

### 1. Whole-cortex decoding

```bash
python src/whole_cortex_main_model.py \
  --data-root /path/to/ds005455 \
  --out-dir outputs/whole_cortex
```

This makes the main within-participant decoding results. It also writes the
trialwise parcel-feature and label files used by the later PCA analyses.

### 2. Robustness and network analyses

```bash
python src/whole_cortex_classifier_variants.py
```

Set `DATA_ROOT`, `OUT_DIR`, and `ANALYSIS` at the top of the script. This file
contains the atlas-size checks, PCA robustness checks, switch/repeat analyses,
and network permutation-importance analyses.

### 3. Local multivoxel analyses

```bash
python src/local_parcelwise_multivoxel.py \
  --data-root /path/to/ds005455 \
  --out-dir outputs/local_parcelwise
```

For the searchlight analysis:

```bash
python src/local_searchlight_multivoxel.py
```

Set `DATA_ROOT` and `OUT_DIR` at the top of the searchlight script before
running.

### 4. Cross-subject transfer

```bash
python src/whole_cortex_cross_subject_transfer.py \
  --data-root /path/to/ds005455 \
  --out-dir outputs/cross_subject_transfer
```

This trains on one participant and tests on another across ordered
teacher-learner pairs.

### 5. Participant table for individual-difference analyses

```bash
python src/prepare_participant_table.py \
  --participants-tsv /path/to/ds005455/participants.tsv \
  --whole-cortex-csv outputs/whole_cortex/distributed_parcel_mean_decoding_by_subject.csv \
  --cross-subject-transfer-csv outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --local-extent-csv outputs/local_parcelwise/local_z_extent_by_subject.csv \
  --out-csv outputs/participants/participant_neural_measures.csv
```

If the AoA column is not detected automatically, add `--aoa-column COLUMN_NAME`.

### 6. Population PCA and residual analyses

```bash
python analysis_code/current_manuscript/run_shared_space_transfer_leave_pair_out.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --transfer-csv outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --participant-table-csv outputs/participants/participant_neural_measures.csv \
  --out-dir outputs/population_pca
```

Related within-participant PCA controls are in the same folder:

- `run_pair_space_self_decoding_leave_pair_out.py`
- `run_leave_one_subject_out_pca_self_decoding.py`

Both take `--features-csv`, `--labels-csv`, and `--out-dir`.

### 7. AoA prediction

```bash
python analysis_code/current_manuscript/run_aoa_prediction_with_whole_cortex_pca50.py \
  --participant-table-csv outputs/participants/participant_neural_measures.csv \
  --leave-pair-transfer-csv outputs/population_pca/shared_space_transfer_leave_pair_out_by_pair.csv \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --out-dir outputs/aoa_prediction
```

## Notes

- Language labels come from `trial_type`: `L1*` is Chinese and `L2*` is English.
- Decoding performance is balanced accuracy.
- Whole-cortex and local decoding use leave-one-run-out cross-validation.
- Cross-subject transfer uses off-diagonal teacher-learner pairs.
- Downstream CSVs used by the manuscript scripts are produced by the earlier
  scripts above; no private local files are required.

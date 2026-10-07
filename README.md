# Multiscale Bilingual Decoding

Code for the manuscript **How the brain distinguishes a bilingual's languages
is partly shared across people and partly their own**.

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

### 3. Parcelwise multivoxel analyses

```bash
python src/local_parcelwise_multivoxel.py \
  --data-root /path/to/ds005455 \
  --out-dir outputs/local_parcelwise
```

Group-level parcel tests with BH-FDR correction are run with:

```bash
python analysis_code/current_manuscript/analyze_group_level_local_parcels.py \
  --local-csv outputs/local_parcelwise/schaefer800_parcelwise_multivoxel_by_subject_and_parcel.csv \
  --mvpa-dir outputs/mvpa_schaefer800 \
  --out-all outputs/local_parcelwise/group_level_local_parcel_stats.csv \
  --out-sig outputs/local_parcelwise/group_level_local_parcel_stats_fdr05.csv
```

The mean-removed local multivoxel control is:

```bash
python src/local_mean_removed_parcelwise_multivoxel.py \
  --data-root /path/to/ds005455 \
  --mvpa-dir outputs/mvpa_schaefer800 \
  --out-dir outputs/local_mean_removed
```

### 4. Cross-participant transfer

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

### 6. Shared dimensions and residual analyses

```bash
python analysis_code/current_manuscript/run_shared_space_transfer_leave_pair_out.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --transfer-csv outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --participant-table-csv outputs/participants/participant_neural_measures.csv \
  --out-dir outputs/population_pca
```

Within-participant decoding in the same leave-pair-out PCA spaces is:

```bash
python analysis_code/current_manuscript/run_pair_space_self_decoding_leave_pair_out.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --out-dir outputs/population_pca
```

The same-preprocessing full cortical pattern transfer used as the black line in
the cross-participant transfer figure is computed by:

```bash
python analysis_code/current_manuscript/run_full_cortical_pattern_same_preprocessing_transfer.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --shared-transfer-csv outputs/population_pca/shared_space_transfer_leave_pair_out_by_pair.csv \
  --teacher-standardized-transfer-csv outputs/cross_subject_transfer/cross_subject_transfer_long.csv \
  --within-self-csv outputs/population_pca/pair_space_self_decoding_leave_pair_out_by_subject.csv \
  --out-dir outputs/full_cortical_pattern_same_preprocessing
```

The run-split residual language-state similarity analysis used for Fig. 3C is:

```bash
python analysis_code/current_manuscript/run_residual_language_contrast_run_split_similarity_leave_pair_out.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --out-dir outputs/population_pca
```

This writes the 77 x 77 run-1 by run-2 similarity matrices and their summary
CSV. For between-participant cells, residuals are computed in the
leave-both-out PCA space for that participant pair. For within-participant
cells, values are averaged across that participant's 76 leave-both-out spaces.

Single-PC transfer for PCs 1-25 is:

```bash
python analysis_code/current_manuscript/run_pc1_pc25_single_pc_transfer.py \
  --features-csv outputs/whole_cortex/whole_cortex_trialwise_parcel_features.csv \
  --labels-csv outputs/whole_cortex/whole_cortex_trialwise_labels.csv \
  --out-dir outputs/population_pca/pc1_pc25_single_pc_transfer
```

The shared-PC maps, network summaries, and BrainSMASH spatial null are:

```bash
python analysis_code/current_manuscript/prepare_shared_pc_maps_networks.py
python analysis_code/current_manuscript/run_shared_pc_brainsmash_network_null.py
```

These scripts expect the full-sample PC loading file and Schaefer/Lipkin network
tables in the manuscript output folders.

Manuscript helper scripts use the repository root by default. If the result
folders are stored somewhere else, pass `--root /path/to/project`.

The targeted Lipkin language and multiple-demand grouped PFI analysis is:

```bash
python analysis_code/current_manuscript/run_targeted_lipkin_pca_grouped_pfi.py \
  --data-root /path/to/ds005455 \
  --mvpa-dir outputs/mvpa_schaefer800 \
  --out-dir outputs/targeted_lipkin_pca_grouped_pfi
```

### 7. AoA prediction

```bash
python analysis_code/current_manuscript/run_aoa_prediction_43_predictors_max_stat.py \
  --participant-table-csv outputs/participants/participant_neural_measures.csv \
  --whole-cortex-by-subject-csv outputs/whole_cortex/distributed_parcel_mean_decoding_by_subject.csv \
  --self-decoding-csv outputs/population_pca/pair_space_self_decoding_leave_pair_out_by_subject.csv \
  --transfer-by-pair-csv outputs/population_pca/shared_space_transfer_leave_pair_out_by_pair.csv \
  --out-dir outputs/aoa_prediction_43_predictors
```

This is the manuscript AoA analysis: it builds the 43-predictor table, runs
leave-one-participant-out ridge prediction for each predictor, and performs the
max-statistic permutation test across predictors. By default it uses 10,000
permutations with seed 42. To rerun the model from a saved predictor table, pass
`--predictor-table-csv outputs/aoa_prediction_43_predictors/age_prediction_predictor_table_43_predictors.csv`.
In the summary CSV, `pearson_r` is the correlation between observed and
cross-validated predicted AoA. `raw_predictor_age_r` is the direct correlation
between the predictor and AoA.

## Notes

- Language labels come from `trial_type`: `L1*` is Chinese and `L2*` is English.
- Decoding performance is balanced accuracy.
- Whole-cortex and local decoding use leave-one-run-out cross-validation.
- Cross-participant transfer uses off-diagonal teacher-learner pairs.
- Downstream CSVs used by the manuscript scripts are produced by the earlier
  scripts above; no private local files are required.

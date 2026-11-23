# Multiscale Bilingual Decoding

Code for **How the brain distinguishes a bilingual's languages is partly shared
across people and partly their own** (Demirkan, Wassermann & Sagot).

Data: Guo et al. (2025), OpenNeuro [ds005455](https://openneuro.org/datasets/ds005455)
v1.1.5, using the single-trial (LSS) activation images released with the
dataset. The data are not included here.

## Setup

```bash
pip install -r requirements.txt
```

Two scripts read the raw data: `scripts/01_extract_parcels.py` and
`scripts/06_parcelwise_multivoxel.py`. Open them and set `DATA_ROOT` to the
folder where you downloaded the dataset, for example:

```python
DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")   # folder with the ds005455 single-trial files
```

Script 01 saves `results/parcels_schaefer800.npz`, and scripts 02-05 load that
file, so they need no data path.

Scripts 05 and 05b also need the probabilistic language and multiple-demand maps
(Lipkin et al., 2022, 2023) from the EvLab resources; set `LIPKIN_LANGUAGE_MAP`
and `LIPKIN_MD_MAP` at their top.

## Analyses

Each script is short and runs top to bottom. Run step 1 first; the others can
run in any order. A "b" script is a variant of the script with the same number.

| Step | Script | Paper |
|------|--------|-------|
| 1 | `scripts/01_extract_parcels.py` | Methods 2.5: each trial as 800 Schaefer parcel means |
| 2 | `scripts/02_whole_cortex_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English within each participant |
| 2b | `scripts/02b_switch_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English on repeat and on switch trials; switch-status decoding |
| 2c | `scripts/02c_robustness.py` | Methods 2.5, SI Fig. S2: same decoding with Schaefer 100-1000 parcels, and with PCA (90% variance) |
| 3 | `scripts/03_cross_participant_transfer.py` | Methods 2.8, Fig. 2B: train on one participant, test on another (full 800 parcels) |
| 4 | `scripts/04_shared_dimensions.py` | Methods 2.8, Figs. 2B, 3A: leave-pair-out PCA; transfer and within-person decoding in shared and residual dimensions |
| 5 | `scripts/05_network_importance.py` | Methods 2.6, Fig. 1D, SI S6B, S7B: permutation importance on network PCA scores (Yeo networks; Lipkin language and MD sets) |
| 5b | `scripts/05b_network_importance_size_matched.py` | Methods 2.6, SI S6A, S7A: same without PCA, networks size-matched to the smallest one |
| 6 | `scripts/06_parcelwise_multivoxel.py` | Methods 2.7, Fig. 1C: decoding within each parcel from voxels, demeaned voxels, parcel mean |

For script 2c, first run script 01 again with `N_PARCELS = 100, 200, ..., 1000`
(set at its top); each run saves `results/parcels_schaefer<N>.npz`.

Scripts 5 and 5b have two settings at the top: `TARGET = "language"` or `"switch"`
(switch-status decoding, SI Fig. S7) and `YEO = 7` or `17` (SI Fig. S6).

```bash
python scripts/01_extract_parcels.py                    # slow, run once -> results/parcels_schaefer800.npz
python scripts/02_whole_cortex_decoding.py              # -> results/whole_cortex_decoding.csv
python scripts/02b_switch_decoding.py                   # -> results/switch_decoding.csv
python scripts/02c_robustness.py                        # -> results/robustness_*.csv
python scripts/03_cross_participant_transfer.py         # -> results/cross_participant_transfer_*.csv
python scripts/04_shared_dimensions.py                  # 1-2 hours -> results/shared_dimensions_*.csv
python scripts/05_network_importance.py                 # -> results/network_importance_*.csv
python scripts/05b_network_importance_size_matched.py   # ~30 min -> results/network_importance_size_matched_*.csv
python scripts/06_parcelwise_multivoxel.py              # 1-2 hours, reads raw images -> results/parcelwise_*.csv
```

`src/` and `analysis_code/` hold the previous version of the code, kept for reference.

## Citation

If you use this code, please cite the preprint and the dataset (Guo et al., 2025).

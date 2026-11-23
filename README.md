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

Set `DATA_ROOT` at the top of `scripts/01_extract_parcels.py` to your copy of the data.

## Analyses

Each script is short and runs top to bottom. Run them in order.

| Step | Script | Paper |
|------|--------|-------|
| 1 | `scripts/01_extract_parcels.py` | Methods 2.5: each trial as 800 Schaefer parcel means |
| 2 | `scripts/02_whole_cortex_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English within each participant |
| 3 | `scripts/03_shared_dimensions.py` | Methods 2.8, Figs. 2B, 3A: leave-pair-out PCA; transfer and within-person decoding in shared and residual dimensions |

```bash
python scripts/01_extract_parcels.py      # slow, run once -> results/parcels_schaefer800.npz
python scripts/02_whole_cortex_decoding.py  # -> results/whole_cortex_decoding.csv
python scripts/03_shared_dimensions.py      # 1-2 hours -> results/shared_dimensions_*.csv
```

`src/` and `analysis_code/` hold the previous version of the code, kept for reference.

## Citation

If you use this code, please cite the preprint and the dataset (Guo et al., 2025).

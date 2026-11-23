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

```bash
python scripts/01_extract_parcels.py      # slow, run once -> results/parcels_schaefer800.npz
python scripts/02_whole_cortex_decoding.py  # -> results/whole_cortex_decoding.csv
```

`legacy/` holds the previous version of the code, kept for reference.

## Citation

If you use this code, please cite the preprint and the dataset (Guo et al., 2025).

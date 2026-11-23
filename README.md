# Multiscale Bilingual Decoding

Code for **How the brain distinguishes a bilingual's languages is partly shared
across people and partly their own** (Demirkan, Wassermann & Sagot).

Data: Guo et al. (2025), OpenNeuro [ds005455](https://openneuro.org/datasets/ds005455)
v1.1.5, using the single-trial (LSS) activation images released with the
dataset. 

## Setup

```bash
pip install -r requirements.txt
```

Set `DATA_ROOT` at the top of `scripts/01_extract_parcels.py` to your copy of the data.
Example: DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")   # folder with the ds005455 single-trial files

Script 01 saves `results/parcels_schaefer800.npz`, and all later scripts load that file, so they need no paths.

## Analyses

| Step | Script | Paper |
|------|--------|-------|
| 1 | `scripts/01_extract_parcels.py` | Methods 2.5: each trial as 800 Schaefer parcel means |
| 2 | `scripts/02_whole_cortex_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English within each participant |
| 3 | `scripts/03_cross_participant_transfer.py` | Methods 2.8, Fig. 2B: train on one participant, test on another (full 800 parcels) |
| 4 | `scripts/04_shared_dimensions.py` | Methods 2.8, Figs. 2B, 3A: leave-pair-out PCA; transfer and within-person decoding in shared and residual dimensions |

```bash
python scripts/01_extract_parcels.py    # run once and use for later, 800 parcel mean full representation -> results/parcels_schaefer800.npz
python scripts/02_whole_cortex_decoding.py  # within-participant language decoding using the full rep. -> results/whole_cortex_decoding.csv
python scripts/03_cross_participant_transfer.py  # cross-participant language decoding using the full rep. -> results/cross_participant_transfer_*.csv
python scripts/04_shared_dimensions.py      # cross and within-participant decoding using k dimensions and/or residual signal -> results/shared_dimensions_*.csv
```

`src/` and `analysis_code/` refer to previous version of the code, only kept for reference.

## Citation

Please cite the preprint and the original dataset as follows:

**Preprint:**

Demirkan, S., Wassermann, D., & Sagot, B. (2026). *How the brain distinguishes a bilingual’s languages is partly shared across people and partly their own*. bioRxiv. https://doi.org/10.64898/2026.09.18.752561

**Original dataset:**

Guo, T., Liu, X., Chen, M., Fu, Y., & Guo, T. (2025). An fMRI dataset for investigating language control and cognitive control in bilinguals. *Scientific Data, 12*, Article 889. https://doi.org/10.1038/s41597-025-05245-9

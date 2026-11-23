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

Two scripts read the raw data: `scripts/01_extract_parcels.py` and
`scripts/06_parcelwise_multivoxel.py`. Set `DATA_ROOT` at the top of both to your copy of the data, for example:

```python
DATA_ROOT = Path("/home/sdemirka/fmri/duo-cogcon")   # folder with the ds005455 single-trial files
```

Script 01 saves `results/parcels_schaefer800.npz`, and all other scripts except 06 load that file.

Scripts 05, 05b and 10 also need the probabilistic language and multiple-demand maps
(Lipkin et al., 2022, 2023) from the EvLab resources: set `LIPKIN_LANGUAGE_MAP`
and `LIPKIN_MD_MAP` at their top. Script 09 reads `participants.tsv` from the
dataset: set `PARTICIPANTS_FILE` at its top.

## Analyses

Run script 01 first. Scripts 07, 08 and 10 use the PCA spaces saved by script 04,
and script 09 uses the results of scripts 02 and 04, so run those first; the others
can run in any order. A "b" or "c" script is a
variant of the script with the same number.

| Step | Script | Paper |
|------|--------|-------|
| 1 | `scripts/01_extract_parcels.py` | Methods 2.5: each trial as 800 Schaefer parcel means |
| 2 | `scripts/02_whole_cortex_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English within each participant |
| 2b | `scripts/02b_switch_decoding.py` | Methods 2.5, Fig. 1B: Chinese vs English on repeat and on switch trials; switch-status decoding |
| 2c | `scripts/02c_robustness.py` | Methods 2.5, SI Fig. S2: same decoding with Schaefer 100-1000 parcels, and with PCA (90% variance) |
| 3 | `scripts/03_cross_participant_transfer.py` | Methods 2.8, Fig. 2B: train on one participant, test on another (full 800 parcels) |
| 4 | `scripts/04_shared_dimensions.py` | Methods 2.8, Figs. 2B, 3A: leave-pair-out PCA; transfer and within-person decoding in shared and residual dimensions |
| 5 | `scripts/05_network_importance.py` | Methods 2.6, Fig. 1D, SI Figs. S6B, S7B: permutation importance on network PCA scores (Yeo networks; Lipkin language and MD sets) |
| 5b | `scripts/05b_network_importance_size_matched.py` | Methods 2.6, SI Figs. S6A, S7A: same without PCA, networks size-matched to the smallest one |
| 6 | `scripts/06_parcelwise_multivoxel.py` | Methods 2.7, Fig. 1C: decoding within each parcel from voxels, demeaned voxels, parcel mean |
| 7 | `scripts/07_single_dimension_transfer.py` | Methods 2.8, Fig. 2C: transfer with each of the first 25 shared dimensions alone |
| 8 | `scripts/08_signatures_identification.py` | Methods 2.8, Fig. 3B-C: English-minus-Chinese signatures, similarity across runs and people, identification |
| 9 | `scripts/09_aoa_prediction.py` | Methods 2.9, Fig. 2D, SI Tables S2-S3: predicting age of English acquisition (and age) from 43 predictors, max-statistic permutation test |
| 10 | `scripts/10_pc_network_profiles.py` | Methods 2.8, Fig. 2E, SI Fig. S9: PCA on all participants, network enrichment of each dimension, BrainSMASH spatial null |

```bash
python scripts/01_extract_parcels.py    # run once and use for later, 800 parcel mean full representation -> results/parcels_schaefer800.npz
python scripts/02_whole_cortex_decoding.py  # within-participant language decoding using the full rep. -> results/whole_cortex_decoding.csv
python scripts/02b_switch_decoding.py   # within-participant decoding on repeat / switch trials, and of switch status -> results/switch_decoding.csv
python scripts/02c_robustness.py        # within-participant decoding with other parcellations and with PCA -> results/robustness_*.csv
python scripts/03_cross_participant_transfer.py  # cross-participant language decoding using the full rep. -> results/cross_participant_transfer_*.csv
python scripts/04_shared_dimensions.py      # cross and within-participant decoding using k dimensions and/or residual signal -> results/shared_dimensions_*.csv
python scripts/05_network_importance.py     # network importance on PCA scores -> results/network_importance_*.csv
python scripts/05b_network_importance_size_matched.py  # network importance without PCA, size-matched -> results/network_importance_size_matched_*.csv
python scripts/06_parcelwise_multivoxel.py  # decoding within each parcel, reads the raw images -> results/parcelwise_*.csv
python scripts/07_single_dimension_transfer.py  # cross-participant decoding with one shared dimension at a time -> results/single_dimension_transfer_*.csv
python scripts/08_signatures_identification.py  # person-specific signatures and identification -> results/signature*.npz/csv
python scripts/09_aoa_prediction.py         # age of English acquisition from transfer and decoding -> results/aoa_*.csv
python scripts/10_pc_network_profiles.py    # loading maps and network profiles of the shared dimensions -> results/pc_*.csv
```

For script 02c, first run script 01 again with `N_PARCELS = 100, 200, ..., 1000`
(set at its top); each run saves `results/parcels_schaefer<N>.npz`.

Scripts 05 and 05b have two settings at their top: `TARGET = "language"` or `"switch"`
(switch-status decoding, SI Fig. S7) and `YEO = 7` or `17` (SI Fig. S6).

## Figures

`figures/` has one script per main figure. Each reads the results saved by the
analysis scripts and writes a PDF and PNG to `figures/output/`.

| Figure | Script | Needs results from |
|--------|--------|--------------------|
| Fig. 1 | `figures/fig1.py` | scripts 02, 02b, 05, 06 |
| Fig. 2 | `figures/fig2.py` | scripts 04, 07, 09, 10 |
| Fig. 3 | `figures/fig3.py` | scripts 04, 08 |

```bash
python figures/fig1.py
python figures/fig2.py
python figures/fig3.py
```

The schematics (Fig. 1A, Fig. 2A) are drawings, not results: put them in
`figures/assets/fig1A.png` and `figures/assets/fig2A.png` and the scripts place
them in the figure. `figures/style.py` holds the shared colours and fonts.

`src/` and `analysis_code/` refer to previous version of the code, only kept for reference.

## Citation

Please cite the preprint and the original dataset as follows:

**Preprint:**

Demirkan, S., Wassermann, D., & Sagot, B. (2026). *How the brain distinguishes a bilingual’s languages is partly shared across people and partly their own*. bioRxiv. https://doi.org/10.64898/2026.09.18.752561

**Original dataset:**

Guo, T., Liu, X., Chen, M., Fu, Y., & Guo, T. (2025). An fMRI dataset for investigating language control and cognitive control in bilinguals. *Scientific Data, 12*, Article 889. https://doi.org/10.1038/s41597-025-05245-9

**Software and atlases used in this code:**

- **Nilearn** (feature extraction, atlases, brain maps): Nilearn contributors. (2026). *Nilearn* (Version 0.13.1). Zenodo. https://doi.org/10.5281/zenodo.8397156
- **scikit-learn** (classifiers, PCA): Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- **BrainSMASH** (spatial null, script 10): Burt, J. B., Helmer, M., Shinn, M., Anticevic, A., & Murray, J. D. (2020). Generative modeling of brain maps with spatial autocorrelation. *NeuroImage, 220*, Article 117038. https://doi.org/10.1016/j.neuroimage.2020.117038
- **Schaefer parcellation**: Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X.-N., Holmes, A. J., Eickhoff, S. B., & Yeo, B. T. T. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex, 28*, 3095–3114. https://doi.org/10.1093/cercor/bhx179
- **Yeo networks**: Yeo, B. T. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., Roffman, J. L., Smoller, J. W., Zöllei, L., Polimeni, J. R., Fischl, B., Liu, H., & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology, 106*, 1125–1165. https://doi.org/10.1152/jn.00338.2011
- **EvLab language atlas** (scripts 05, 05b, 10): Lipkin, B., Tuckute, G., Affourtit, J., Small, H., Mineroff, Z., Kean, H., Jouravlev, O., Rakocevic, L., Pritchett, B., Siegelman, M., Hoeflin, C., Pongos, A., Blank, I. A., Kline Struhl, M., Ivanova, A., Shannon, S., Sathe, A., Hoffmann, M., Nieto-Castañón, A., & Fedorenko, E. (2022). Probabilistic atlas for the language network based on precision fMRI data from >800 individuals. *Scientific Data, 9*, Article 529. https://doi.org/10.1038/s41597-022-01645-3
- **EvLab multiple-demand atlas** (scripts 05, 05b, 10): Lipkin, B., Blank, I., & Fedorenko, E. (2023). Probabilistic atlases for the multiple demand (MD) and theory of mind (ToM) networks based on large-scale precision localizers. *Figshare*. https://doi.org/10.6084/m9.figshare.22306348

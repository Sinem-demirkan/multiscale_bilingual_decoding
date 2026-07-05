# Multiscale Bilingual Decoding

This repository contains the code used to compute the two participant-level measures from our bilingual fMRI decoding study.

## Data

This code uses the single-trial beta images and event files from the publicly available **Guo et al. Chinese–English bilingual picture-naming dataset**:

> Guo, T., Liu, X., Chen, M., Fu, Y., & Guo, T. (2025). *An fMRI dataset for investigating language control and cognitive control in bilinguals*. *Scientific Data, 12*(1). https://doi.org/10.1038/s41597-025-05245-9

The dataset is available from **OpenNeuro**:

* **Dataset:** ds005455
* **DOI:** https://doi.org/10.18112/openneuro.ds005455.v1.1.5

The dataset is **not included** in this repository. Please download it from OpenNeuro.

After downloading, rename the dataset directory to `duo-cogcon`. The code expects the following directory structure:

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

## Notes

- Language labels are derived from `trial_type`: entries beginning with `L1` are labeled `L1`, and entries beginning with `L2` are labeled `L2`.
- Cross-run evaluation means one run is used for training and the other for testing, then the folds are averaged.
- The local `z > 1.64` extent measure uses a one-sided binomial p-value derived from the observed parcelwise decoding score.

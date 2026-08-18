# Time-Series Dynamical Regime Analysis

Analysis of 25 single-column time series:

- 12 `sac_ascf_*` SAC signals
- 12 `NLM_sac_ascf_*` NLM-SAC signals
- `lorenz.txt` as an ideal deterministic nonlinear reference

The analysis investigates whether the signals can be separated into groups with distinct dynamical characteristics.

## Pipeline

1. Extract statistical, temporal, spectral, entropy, DFA and nonlinear-prediction features.
2. Build the complete 25-signal feature matrix.
3. Standardize features.
4. Perform PCA.
5. Perform UMAP when `umap-learn` is available; otherwise the analysis script explicitly falls back to PCA for a reproducible 2-D projection.
6. Perform K-means and hierarchical clustering.
7. Test the proposed groups using cross-validation.
8. Use pair-aware leave-one-base-signal-out CV so SAC/NLM versions of the same signal are held out together.
9. Run a label-permutation test and compare within-SAC/NLM distances with between-signal distances.

## Main findings

- PCA PC1 + PC2 explain about 73.4% of standardized feature variance in the generated matrix.
- `sac_ascf_kai` and `NLM_sac_ascf_kai` form a strongly distinctive high-frequency/outlier regime in unsupervised analysis.
- The Persistent and Intermediate regimes remain reasonably separable under pair-aware CV: approximately 81.8% accuracy, 81.7% balanced accuracy and 81.7% macro-F1 in the two-class test excluding `kai`.
- A three-class supervised result cannot be considered fully validated because `kai` is the only independent base signal in the high-frequency class. Holding `kai` out leaves no training examples of that class.
- SAC/NLM versions of the same base signal are closer in feature space than unrelated base signals, indicating that the feature representation preserves meaningful dynamical identity through the NLM transformation.
- The Lorenz reference has low permutation entropy and very low nonlinear prediction error relative to the SAC signals, consistent with deterministic nonlinear structure.

## Important statistical caveat

There are only 12 independent base signal identities, and the third proposed class has only one independent identity (`kai`). Therefore, the analysis supports the existence of a distinctive `kai` regime but does **not** establish a generalizable three-class classifier. More independent examples are required for that claim.

## Directory layout

```text
.
├── data/raw/                 # original 25 input files
├── results/                  # feature matrices, plots and CV workbooks
├── src/
│   ├── feature_extraction.py
│   ├── pca_clustering.py
│   └── cross_validation.py
└── README.md
```

## Requirements

Core packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib openpyxl
```

Optional UMAP:

```bash
pip install umap-learn
```

## Reproduction

From the repository root:

```bash
python src/feature_extraction.py
python src/pca_clustering.py
python src/cross_validation.py
```

The scripts read from `data/raw/` and write to `results/`.

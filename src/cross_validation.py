"""Cross-validation of proposed dynamical regimes.

Primary validated task:
    Persistent_low_frequency vs Intermediate_broadband

Validation:
    1. Stratified 2-fold CV.
    2. Pair-aware leave-one-base-signal-out CV.

The High_frequency_outlier class (kai) is retained for exploratory
analysis but is NOT included in supervised validation because it has
only one independent base signal.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.base import clone


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FEATURE_FILE = OUT / "timeseries_feature_matrix_25.csv"

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature matrix not found:\n{FEATURE_FILE}\n\n"
        "Run src/feature_extraction.py first."
    )


# =====================================================================
# Load feature matrix
# =====================================================================

df = pd.read_csv(FEATURE_FILE)


# =====================================================================
# Regime labels
# =====================================================================

label_map = {
    "alpha": "Persistent_low_frequency",
    "beta": "Persistent_low_frequency",

    "delta": "Intermediate_broadband",
    "gamma": "Intermediate_broadband",

    "kai": "High_frequency_outlier",

    "kappa": "Persistent_low_frequency",
    "lambda": "Persistent_low_frequency",
    "mu": "Persistent_low_frequency",

    "nu": "Intermediate_broadband",
    "phi": "Intermediate_broadband",
    "rho": "Intermediate_broadband",
    "theta": "Intermediate_broadband",
}


def base_signal_name(signal_name):
    """Return common base name for SAC/NLM signal pairs."""

    return (
        str(signal_name)
        .replace("NLM_sac_ascf_", "")
        .replace("sac_ascf_", "")
    )


# =====================================================================
# Prepare experimental signals
# =====================================================================

d = df[df["dataset"] != "REFERENCE"].copy()

d["base_signal"] = d["signal"].map(base_signal_name)
d["group"] = d["base_signal"].map(label_map)

if d["group"].isna().any():
    unknown = sorted(
        d.loc[d["group"].isna(), "base_signal"].unique()
    )

    raise ValueError(
        f"Unknown regime labels for: {unknown}"
    )


# =====================================================================
# Validated two-class population
# =====================================================================

VALID_GROUPS = [
    "Persistent_low_frequency",
    "Intermediate_broadband",
]

d2 = d[d["group"].isin(VALID_GROUPS)].copy()


# =====================================================================
# Feature matrix
# =====================================================================

ids = [
    "signal",
    "dataset",
    "group",
    "source_file",
    "base_signal",
]

cols = [
    c
    for c in d2.columns
    if c not in ids + ["n_samples"]
]

X = d2[cols].apply(
    pd.to_numeric,
    errors="coerce",
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan,
)

y = d2["group"].to_numpy()
groups = d2["base_signal"].to_numpy()


# =====================================================================
# Models
# =====================================================================

logistic_pca = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "pca",
            PCA(
                n_components=0.90,
                svd_solver="full",
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ]
)


rbf_svm_pca = Pipeline(
    [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "pca",
            PCA(
                n_components=0.90,
                svd_solver="full",
            ),
        ),
        (
            "classifier",
            SVC(
                C=1.0,
                kernel="rbf",
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# =====================================================================
# Cross-validation prediction helper
# =====================================================================

def cross_validated_predictions(
    estimator,
    splitter,
    X_data,
    y_data,
    groups_data=None,
):
    """Return out-of-fold predictions."""

    predictions = np.empty(
        len(y_data),
        dtype=object,
    )

    if groups_data is None:
        split_iterator = splitter.split(
            X_data,
            y_data,
        )
    else:
        split_iterator = splitter.split(
            X_data,
            y_data,
            groups_data,
        )

    for train_idx, test_idx in split_iterator:

        fitted = clone(estimator)

        fitted.fit(
            X_data.iloc[train_idx],
            y_data[train_idx],
        )

        predictions[test_idx] = fitted.predict(
            X_data.iloc[test_idx]
        )

    return predictions


# =====================================================================
# Metric helper
# =====================================================================

def classification_metrics(y_true, y_pred):
    """Calculate metrics safely for aggregate or single-class folds."""

    result = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": np.nan,
        "macro_F1": np.nan,
    }

    # A single held-out base signal contains only one true class.
    # Balanced accuracy and two-class macro-F1 are therefore not
    # meaningful for that individual fold.
    if len(np.unique(y_true)) >= 2:

        result["balanced_accuracy"] = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        result["macro_F1"] = f1_score(
            y_true,
            y_pred,
            labels=VALID_GROUPS,
            average="macro",
            zero_division=0,
        )

    return result


# =====================================================================
# 1. Stratified 2-fold CV
# =====================================================================

stratified = StratifiedKFold(
    n_splits=2,
    shuffle=True,
    random_state=42,
)

stratified_rows = []

for name, estimator in [
    ("Logistic+PCA", logistic_pca),
    ("RBF-SVM+PCA", rbf_svm_pca),
]:

    predictions = cross_validated_predictions(
        estimator,
        stratified,
        X,
        y,
    )

    metrics = classification_metrics(
        y,
        predictions,
    )

    stratified_rows.append(
        [
            "2-fold stratified",
            name,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_F1"],
            (
                "Potentially optimistic: "
                "SAC/NLM pairs may split across folds"
            ),
        ]
    )


stratified_results = pd.DataFrame(
    stratified_rows,
    columns=[
        "validation",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_F1",
        "interpretation",
    ],
)


# =====================================================================
# 2. Pair-aware leave-one-base-signal-out CV
# =====================================================================

logo = LeaveOneGroupOut()

logo_rows = []

logo_truth = []
logo_predictions = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(
        X,
        y,
        groups,
    ),
    start=1,
):

    held_out = groups[test_idx][0]

    estimator = clone(logistic_pca)

    estimator.fit(
        X.iloc[train_idx],
        y[train_idx],
    )

    predictions = estimator.predict(
        X.iloc[test_idx]
    )

    metrics = classification_metrics(
        y[test_idx],
        predictions,
    )

    logo_truth.extend(y[test_idx])
    logo_predictions.extend(predictions)

    if len(np.unique(y[test_idx])) < 2:
        note = (
            "Single-class held-out fold; "
            "balanced accuracy and macro-F1 "
            "are not defined. Aggregate LOGO "
            "metrics are reported separately."
        )
    else:
        note = ""

    logo_rows.append(
        [
            fold,
            held_out,
            "TESTED",
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_F1"],
            note,
        ]
    )


logo_results = pd.DataFrame(
    logo_rows,
    columns=[
        "fold",
        "held_out_base_signal",
        "status",
        "accuracy",
        "balanced_accuracy",
        "macro_F1",
        "note",
    ],
)


# =====================================================================
# Aggregate pair-aware LOGO result
# =====================================================================

logo_truth = np.asarray(logo_truth)
logo_predictions = np.asarray(logo_predictions)

logo_metrics = classification_metrics(
    logo_truth,
    logo_predictions,
)


# =====================================================================
# Pair-level predictions
# =====================================================================

pair_predictions = []

for base_signal in sorted(
    np.unique(groups)
):

    mask = groups == base_signal

    true_values = y[mask]

    train_mask = ~mask

    estimator = clone(logistic_pca)

    estimator.fit(
        X.loc[train_mask],
        y[train_mask],
    )

    predicted_values = estimator.predict(
        X.loc[mask]
    )

    pair_predictions.append(
        [
            base_signal,
            true_values[0],
            predicted_values[0],
            predicted_values[1],
            bool(
                np.all(
                    predicted_values == true_values
                )
            ),
        ]
    )


pair_predictions = pd.DataFrame(
    pair_predictions,
    columns=[
        "base_signal",
        "true_group",
        "SAC_prediction",
        "NLM_prediction",
        "both_correct",
    ],
)


# =====================================================================
# Aggregate confusion matrix
# =====================================================================

cm = confusion_matrix(
    logo_truth,
    logo_predictions,
    labels=VALID_GROUPS,
)

confusion = pd.DataFrame(
    cm,
    index=[
        "Persistent",
        "Intermediate",
    ],
    columns=[
        "Persistent",
        "Intermediate",
    ],
)


# =====================================================================
# Main summary
# =====================================================================

summary_rows = []

for _, row in stratified_results.iterrows():
    summary_rows.append(row.tolist())


summary_rows.append(
    [
        "Leave-one-base-signal-out",
        "Logistic+PCA",
        logo_metrics["accuracy"],
        logo_metrics["balanced_accuracy"],
        logo_metrics["macro_F1"],
        (
            "Primary validation: SAC/NLM "
            "pairs kept together"
        ),
    ]
)

summary = pd.DataFrame(
    summary_rows,
    columns=[
        "validation",
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_F1",
        "interpretation",
    ],
)


# =====================================================================
# Exploratory three-class information
# =====================================================================

three_class_counts = (
    d.groupby("group")["base_signal"]
    .nunique()
    .reindex(
        [
            "Persistent_low_frequency",
            "Intermediate_broadband",
            "High_frequency_outlier",
        ],
        fill_value=0,
    )
    .reset_index()
)

three_class_counts.columns = [
    "group",
    "independent_base_signals",
]

three_class_note = pd.DataFrame(
    [
        [
            "High_frequency_outlier",
            "NOT VALIDATED",
            (
                "Only one independent base signal "
                "(kai); no independent held-out "
                "training example exists."
            ),
        ]
    ],
    columns=[
        "group",
        "status",
        "note",
    ],
)


# =====================================================================
# Save results
# =====================================================================

output_file = (
    OUT / "timeseries_cross_validation_results.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl",
) as writer:

    summary.to_excel(
        writer,
        index=False,
        sheet_name="Summary",
    )

    stratified_results.to_excel(
        writer,
        index=False,
        sheet_name="Stratified_2Class",
    )

    logo_results.to_excel(
        writer,
        index=False,
        sheet_name="Pair_Aware_LOGO",
    )

    pair_predictions.to_excel(
        writer,
        index=False,
        sheet_name="Pair_Predictions",
    )

    confusion.to_excel(
        writer,
        sheet_name="LOGO_Confusion",
    )

    three_class_counts.to_excel(
        writer,
        index=False,
        sheet_name="Three_Class_Counts",
    )

    three_class_note.to_excel(
        writer,
        index=False,
        sheet_name="Three_Class_Status",
    )


# =====================================================================
# Console report
# =====================================================================

print()
print("=" * 72)
print("CROSS-VALIDATION OF PROPOSED DYNAMICAL REGIMES")
print("=" * 72)

print()
print("PRIMARY VALIDATED TASK:")
print(
    "Persistent_low_frequency vs "
    "Intermediate_broadband"
)

print()
print(
    "Independent base signals in validated task:",
    len(np.unique(groups)),
)

print(
    "SAC/NLM observations in validated task:",
    len(y),
)

print(
    "Excluded exploratory base signal: kai "
    "(High_frequency_outlier; only one independent base signal)"
)

print()
print("SUMMARY:")
print(
    summary.to_string(index=False)
)

print()
print("PAIR-AWARE LEAVE-ONE-BASE-SIGNAL-OUT:")
print(
    logo_results.to_string(index=False)
)

print()
print("LOGO CONFUSION MATRIX:")
print(
    confusion.to_string()
)

print()
print("BASE-SIGNAL PREDICTIONS:")
print(
    pair_predictions.to_string(index=False)
)

print()
print("THREE-CLASS STATUS:")
print(
    three_class_counts.to_string(index=False)
)

print()
print(
    three_class_note.to_string(index=False)
)

print()
print("Results written to:")
print(output_file)

print("=" * 72)

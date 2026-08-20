"""PCA, UMAP, K-means and hierarchical clustering."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.cluster.hierarchy import linkage, dendrogram


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

FEATURE_FILE = OUT / "timeseries_feature_matrix_25.csv"


# ---------------------------------------------------------------------
# Load feature matrix
# ---------------------------------------------------------------------

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature matrix not found: {FEATURE_FILE}\n"
        "Run src/feature_extraction.py first."
    )

df = pd.read_csv(FEATURE_FILE)


# ---------------------------------------------------------------------
# Exploratory classification labels
# ---------------------------------------------------------------------
# These labels were established during the exploratory analysis and are
# included here for reproducibility.

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


for i, row in df.iterrows():

    if row["dataset"] != "REFERENCE":

        base = (
            str(row["signal"])
            .replace("NLM_sac_ascf_", "")
            .replace("sac_ascf_", "")
        )

        df.loc[i, "group"] = label_map.get(
            base,
            "UNASSIGNED"
        )


# ---------------------------------------------------------------------
# Select numerical features
# ---------------------------------------------------------------------

ids = [
    "signal",
    "dataset",
    "group",
    "source_file",
]

cols = [
    c
    for c in df.columns
    if c not in ids + ["n_samples"]
]

# Convert to numeric explicitly.
X = df[cols].apply(
    pd.to_numeric,
    errors="coerce"
)

# Replace infinities.
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation.
X = X.fillna(
    X.median()
)

# Standardization.
scaler = StandardScaler()
Z = scaler.fit_transform(X)


# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------

pca = PCA()
XP = pca.fit_transform(Z)

p2 = PCA(n_components=2)
PC = p2.fit_transform(Z)


# ---------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------

try:

    import umap.umap_ as umap

    reducer = umap.UMAP(
        n_neighbors=7,
        min_dist=0.15,
        n_components=2,
        random_state=42,
    )

    U = reducer.fit_transform(Z)

    umap_status = "UMAP"

except ImportError:

    raise ImportError(
        "The 'umap-learn' package is required for this analysis.\n"
        "Install it with:\n\n"
        "    pip install umap-learn\n"
    )

except Exception as exc:

    raise RuntimeError(
        f"UMAP failed: {exc}"
    ) from exc


# ---------------------------------------------------------------------
# K-means evaluation
# ---------------------------------------------------------------------

kmrows = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        n_init=50,
        random_state=42,
    )

    lab = model.fit_predict(Z)

    silhouette = silhouette_score(
        Z,
        lab
    )

    ari = adjusted_rand_score(
        df["group"],
        lab
    )

    kmrows.append(
        [
            k,
            silhouette,
            ari,
        ]
    )


kdf = pd.DataFrame(
    kmrows,
    columns=[
        "k",
        "silhouette",
        "ARI_vs_predefined_groups",
    ],
)

best = int(
    kdf.loc[
        kdf["silhouette"].idxmax(),
        "k"
    ]
)


# ---------------------------------------------------------------------
# Final K-means
# ---------------------------------------------------------------------

km_model = KMeans(
    n_clusters=best,
    n_init=100,
    random_state=42,
)

kl = km_model.fit_predict(Z)


# ---------------------------------------------------------------------
# Agglomerative clustering
# ---------------------------------------------------------------------

al = AgglomerativeClustering(
    n_clusters=3
).fit_predict(Z)


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------

res = df[ids].copy()

res[["PC1", "PC2"]] = PC

res[[
    "UMAP1",
    "UMAP2",
]] = U

res[f"KMeans_k{best}"] = kl

res["Agglomerative_k3"] = al


# ---------------------------------------------------------------------
# PCA loadings
# ---------------------------------------------------------------------

load = pd.DataFrame(
    p2.components_.T,
    index=cols,
    columns=[
        "PC1_loading",
        "PC2_loading",
    ],
)


# ---------------------------------------------------------------------
# PCA variance
# ---------------------------------------------------------------------

pca_variance = pd.DataFrame(
    {
        "PC": np.arange(
            1,
            len(
                pca.explained_variance_ratio_
            ) + 1,
        ),
        "explained_variance_ratio":
            pca.explained_variance_ratio_,
        "cumulative_variance":
            np.cumsum(
                pca.explained_variance_ratio_
            ),
    }
)


# ---------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------

excel_file = (
    OUT /
    "timeseries_PCA_UMAP_clustering.xlsx"
)

with pd.ExcelWriter(
    excel_file,
    engine="openpyxl",
) as writer:

    res.to_excel(
        writer,
        index=False,
        sheet_name="Embeddings_Clusters",
    )

    kdf.to_excel(
        writer,
        index=False,
        sheet_name="KMeans_Evaluation",
    )

    load.to_excel(
        writer,
        sheet_name="PCA_Loadings",
    )

    pca_variance.to_excel(
        writer,
        index=False,
        sheet_name="PCA_Variance",
    )

    pd.DataFrame(
        {
            "parameter": [
                "UMAP_status",
                "UMAP_n_neighbors",
                "UMAP_min_dist",
                "UMAP_random_state",
                "KMeans_best_k",
            ],
            "value": [
                umap_status,
                7,
                0.15,
                42,
                best,
            ],
        }
    ).to_excel(
        writer,
        index=False,
        sheet_name="Analysis_Settings",
    )


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------

def plot_embedding(
    a,
    b,
    title,
    xlab,
    ylab,
    outfile,
):

    plt.figure(
        figsize=(10, 7)
    )

    groups = df["group"].fillna(
        "UNASSIGNED"
    )

    for group in groups.unique():

        mask = (
            groups.to_numpy() == group
        )

        plt.scatter(
            a[mask],
            b[mask],
            label=group,
            s=55,
        )

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        OUT / outfile,
        dpi=180,
    )

    plt.close()


# ---------------------------------------------------------------------
# PCA plot
# ---------------------------------------------------------------------

plot_embedding(
    PC[:, 0],
    PC[:, 1],
    "PCA of 25 Time-Series Feature Vectors",
    (
        f"PC1 "
        f"({p2.explained_variance_ratio_[0] * 100:.1f}%)"
    ),
    (
        f"PC2 "
        f"({p2.explained_variance_ratio_[1] * 100:.1f}%)"
    ),
    "timeseries_PCA.png",
)


# ---------------------------------------------------------------------
# UMAP plot
# ---------------------------------------------------------------------

plot_embedding(
    U[:, 0],
    U[:, 1],
    "UMAP of 25 Time-Series Feature Vectors",
    "UMAP 1",
    "UMAP 2",
    "timeseries_UMAP.png",
)


# ---------------------------------------------------------------------
# Hierarchical clustering / dendrogram
# ---------------------------------------------------------------------

linkage_matrix = linkage(
    Z,
    method="ward",
)

# IMPORTANT:
# Convert the pandas Series to a NumPy array.
# scipy.cluster.hierarchy.dendrogram() performs positional indexing,
# while pandas Series indexing can interpret negative integers as
# labels, producing KeyError on recent pandas versions.

dendrogram_labels = (
    df["signal"]
    .astype(str)
    .str.replace(
        "sac_ascf_",
        "",
        regex=False,
    )
    .str.replace(
        "NLM_sac_ascf_",
        "NLM_",
        regex=False,
    )
    .to_numpy()
)


plt.figure(
    figsize=(14, 7)
)

dendrogram(
    linkage_matrix,
    labels=dendrogram_labels,
    leaf_rotation=90,
    leaf_font_size=8,
)

plt.title(
    "Hierarchical Clustering — Ward Linkage"
)

plt.ylabel(
    "Ward distance"
)

plt.tight_layout()

plt.savefig(
    OUT / "timeseries_dendrogram.png",
    dpi=180,
)

plt.close()


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print()
print("=" * 70)
print("PCA / UMAP / CLUSTERING ANALYSIS")
print("=" * 70)

print(
    f"Signals analyzed: {len(df)}"
)

print(
    f"Features used: {len(cols)}"
)

print(
    f"PCA PC1 variance: "
    f"{p2.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"PCA PC2 variance: "
    f"{p2.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"PCA PC1+PC2 variance: "
    f"{p2.explained_variance_ratio_.sum() * 100:.2f}%"
)

print(
    f"Best K-means k: {best}"
)

print(
    f"UMAP: {umap_status}"
)

print()
print("K-means evaluation:")
print(
    kdf.to_string(index=False)
)

print()
print(
    f"Results written to:\n"
    f"{excel_file}"
)

print(
    f"\nPCA plot:\n"
    f"{OUT / 'timeseries_PCA.png'}"
)

print(
    f"\nUMAP plot:\n"
    f"{OUT / 'timeseries_UMAP.png'}"
)

print(
    f"\nDendrogram:\n"
    f"{OUT / 'timeseries_dendrogram.png'}"
)

print("=" * 70)

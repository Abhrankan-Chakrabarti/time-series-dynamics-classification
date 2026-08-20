"""Extract the 25-signal feature matrix used in the analysis."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.neighbors import KNeighborsRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def autocorr(x, lag):
    if lag >= len(x):
        return np.nan

    a = x[:-lag]
    b = x[lag:]
    sa, sb = a.std(), b.std()

    return np.corrcoef(a, b)[0, 1] if sa and sb else 0.0


def permutation_entropy(x, m=5):
    x = np.asarray(x)

    if len(x) < m + 1:
        return np.nan

    patterns = {}

    for i in range(len(x) - m + 1):
        order = tuple(
            np.argsort(x[i:i + m], kind="mergesort")
        )
        patterns[order] = patterns.get(order, 0) + 1

    p = np.array(list(patterns.values()), dtype=float)
    p /= p.sum()

    return float(
        -(p * np.log(p)).sum()
        / np.log(math.factorial(m))
    )


def dfa(x):
    # Compact DFA implementation, scales chosen to match
    # the exploratory analysis.
    x = np.asarray(x, float)

    y = np.cumsum(x - x.mean())

    scales = np.unique(
        np.logspace(
            np.log10(8),
            np.log10(max(16, len(x) // 8)),
            12
        ).astype(int)
    )

    vals = []
    ss = []

    for s in scales:
        n = len(y) // s

        if n < 2:
            continue

        rms = []

        for j in range(n):
            seg = y[j * s:(j + 1) * s]
            t = np.arange(s)

            coef = np.polyfit(t, seg, 1)

            rms.append(
                np.sqrt(
                    np.mean(
                        (seg - np.polyval(coef, t)) ** 2
                    )
                )
            )

        if np.mean(rms) > 0:
            ss.append(s)
            vals.append(np.mean(rms))

    return (
        float(
            np.polyfit(
                np.log(ss),
                np.log(vals),
                1
            )[0]
        )
        if len(ss) >= 2
        else np.nan
    )


def nonlinear_prediction_error(x):
    # One-step local prediction from a short delay embedding;
    # normalized RMSE.
    x = np.asarray(x, float)

    if len(x) < 100:
        return np.nan

    # Subsample for tractability while retaining
    # deterministic local structure.
    maxn = 12000

    if len(x) > maxn:
        idx = np.linspace(
            0,
            len(x) - 1,
            maxn
        ).astype(int)

        x = x[idx]

    z = (x - x.mean()) / (x.std() or 1)

    lag = 1
    emb = 3

    Y = np.column_stack([
        z[
            i:
            len(z) - 1 - ((emb - 1) * lag) + i
        ]
        for i in range(0, emb * lag, lag)
    ])

    target = z[emb * lag:]

    n = len(target)

    if n < 100:
        return np.nan

    split = int(n * 0.8)

    knn = KNeighborsRegressor(
        n_neighbors=min(
            15,
            max(3, split // 50)
        ),
        weights="distance"
    )

    knn.fit(
        Y[:split],
        target[:split]
    )

    pred = knn.predict(Y[split:])

    return float(
        np.sqrt(
            np.mean(
                (pred - target[split:]) ** 2
            )
        )
    )


def features(x):
    x = np.asarray(x, float)

    std = np.std(x, ddof=1)

    f = {
        "n_samples": len(x),
        "mean": np.mean(x),
        "std": std,
        "median": np.median(x),
        "iqr": stats.iqr(x),
        "skewness": stats.skew(x),
        "excess_kurtosis": stats.kurtosis(x),
        "diff_std": np.std(np.diff(x), ddof=1),
        "roughness": (
            np.std(np.diff(x), ddof=1)
            / (std or 1)
        ),
    }

    for lag in [1, 5, 10, 20, 50, 100]:
        f[f"autocorr_lag{lag}"] = autocorr(x, lag)

    y = x - x.mean()

    spec = np.abs(np.fft.rfft(y)) ** 2
    freq = np.fft.rfftfreq(len(y))

    spec[0] = 0

    spec_sum = spec.sum()

    if spec_sum:
        p = spec / spec_sum
    else:
        p = spec

    f["spectral_entropy"] = float(
        -(
            p[p > 0] * np.log(p[p > 0])
        ).sum()
        / np.log(len(p))
    )

    f["dominant_frequency"] = float(
        freq[np.argmax(spec)]
    )

    bands = [
        (0, 0.01),
        (0.01, 0.05),
        (0.05, 0.20),
        (0.20, 0.50),
    ]

    for lo, hi in bands:
        mask = (
            (freq >= lo)
            & (freq < hi)
        )

        f[
            f"spectral_energy_{lo:.2f}_{hi:.2f}"
        ] = (
            float(spec[mask].sum() / spec_sum)
            if spec_sum
            else np.nan
        )

    f["zero_crossing_rate"] = float(
        np.mean(
            np.diff(
                np.signbit(y)
            ) != 0
        )
    )

    f["permutation_entropy_m5"] = (
        permutation_entropy(x, 5)
    )

    f["dfa_exponent"] = dfa(x)

    f["nonlinear_prediction_error"] = (
        nonlinear_prediction_error(x)
    )

    return f


rows = []

for path in sorted(DATA.glob("*.txt")):

    x = (
        pd.read_csv(
            path,
            header=None,
            sep=r"\s+"
        )
        .iloc[:, 0]
        .dropna()
        .to_numpy()
    )

    name = path.stem

    if name == "lorenz":
        dataset = "REFERENCE"
        group = "Lorenz_reference"

    elif name.startswith("NLM_"):
        dataset = "NLM"
        group = "UNASSIGNED"

    else:
        dataset = "SAC"
        group = "UNASSIGNED"

    row = {
        "signal": name,
        "dataset": dataset,
        "group": group,
        "source_file": path.name,
    }

    row.update(features(x))
    rows.append(row)


out = pd.DataFrame(rows)

out.to_csv(
    OUT / "timeseries_feature_matrix_25.csv",
    index=False
)

out.to_excel(
    OUT / "timeseries_feature_matrix_25.xlsx",
    index=False,
    sheet_name="Feature_Matrix"
)

print(out.to_string(index=False))

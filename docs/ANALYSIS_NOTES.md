# Analysis notes

## Feature matrix
25 signals × 25 columns (4 identifiers + sample count + 21 numerical features). The numerical features cover:

- distribution: mean, standard deviation, median, IQR, skewness, excess kurtosis
- roughness: first-difference standard deviation and normalized roughness
- temporal dependence: autocorrelation at lags 1, 5, 10, 20, 50, 100
- spectrum: spectral entropy, dominant frequency, and four frequency-band energy fractions
- zero-crossing rate
- permutation entropy (m=5)
- DFA exponent
- nonlinear one-step prediction error

## Clustering
The strongest unsupervised observation is that the `kai` SAC/NLM pair is highly distinctive. The data do not cleanly support three equally populated clusters; the third proposed regime is represented by only one independent base signal.

## Cross-validation
Pair-aware leave-one-base-signal-out validation is the preferred estimate because SAC and NLM versions of the same base signal are related. The high-frequency class cannot be tested as a generalizable class because `kai` is its sole independent example.

The two-class Persistent vs Intermediate test (excluding `kai`) is the cleanest current generalization experiment.

"""
gapfilling_utils.py

Utilities for EC flux gapfilling + uncertainty attachment.

Methodology (Modified for CV-Ensemble):
- Main Prediction: Model trained on 100% of data.
- Uncertainty Components:
    (a) Structural Uncertainty (sigma_ens): Standard deviation of predictions from K CV-fold models.
    (b) Residual Uncertainty (sigma_resid): Quantile-based scaling of OOF residuals.
- Total Uncertainty: sigma_gf(t) = sqrt(sigma_ens(t)^2 + sigma_resid(t)^2)

Notes
-----
- If you log-transform y, pass an inverse transform function `inv_y`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd

Idx = np.ndarray
Split = Tuple[Idx, Idx]


# -----------------------------------------------------------------------------
# Small numeric helpers
# -----------------------------------------------------------------------------

def _as_1d_float(x: Any) -> np.ndarray:
    """Convert array-like to 1D float array (keeps NaNs)."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return float("nan")
    y = y_true[m]
    yhat = y_pred[m]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _check_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {name}: {missing}")


# -----------------------------------------------------------------------------
# Transform helpers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Log1pShift:
    """log1p transform with optional shift if min < 0."""
    shift: float  # the min value if min<0 else 0

    def transform(self, x: Any) -> np.ndarray:
        x = _as_1d_float(x)
        if self.shift < 0:
            return np.log1p(x - self.shift)
        return np.log1p(x)

    def inverse(self, x: Any) -> np.ndarray:
        x = _as_1d_float(x)
        if self.shift < 0:
            return np.expm1(x) + self.shift
        return np.expm1(x)


def setup_log_transform(
    data: pd.DataFrame,
    target: str,
    *,
    apply: bool = False,
    plot: bool = False,
    bins: int = 20,
):
    _check_columns(data, [target], name="data")
    min_value = float(pd.to_numeric(data[target], errors="coerce").min())
    t = Log1pShift(shift=min_value if min_value < 0 else 0.0)

    def log_transform(x):
        return t.transform(x)

    def inverse_log_transform(x):
        return t.inverse(x)

    if plot:
        import matplotlib.pyplot as plt
        plt.hist(data[target].dropna(), bins=bins)
        plt.title("Original")
        plt.show()
        plt.hist(log_transform(data[target].dropna()), bins=bins)
        plt.title("Log-transformed")
        plt.show()

    data_out = data
    if apply:
        data_out = data.copy()
        data_out[target] = log_transform(data_out[target])

    return log_transform, inverse_log_transform, min_value, data_out


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def undersample_target(
    data: pd.DataFrame,
    target: str,
    *,
    quantile_cutoff: float = 0.8,
    fraction: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
):
    _check_columns(data, [target], name="data")

    if not (0.0 < quantile_cutoff < 1.0):
        raise ValueError("quantile_cutoff must be in (0, 1)")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")

    cutoff_value = float(pd.to_numeric(data[target], errors="coerce").quantile(quantile_cutoff))
    upper = data[data[target] > cutoff_value]
    lower = data[data[target] <= cutoff_value]

    lower_sampled = lower.sample(frac=fraction, random_state=random_state)
    out = pd.concat([upper, lower_sampled], axis=0).sort_index()

    if verbose:
        kept = len(out)
        total = len(data)
        print(
            f"Undersample {target}: cutoff q={quantile_cutoff:.2f} -> {cutoff_value:.6g}; "
            f"kept {kept}/{total} rows ({kept/total:.1%}); lower kept fraction={fraction:.2f}"
        )
    return out, cutoff_value


# -----------------------------------------------------------------------------
# Artificial gap sampling (Irvin et al. 2021) for realistic CV splits
# -----------------------------------------------------------------------------
#
# these helpers hold out windows positioned in REAL TIME on the original
# series, with lengths drawn from a distribution fitted to the site's own gap
# structure. Two consequences worth knowing:
#   * held-out windows have realistic depth in wall-clock terms, so OOF
#     residuals reflect how the model behaves mid-gap rather than always one
#     step from an observation;
#   * the resulting splits are NOT a partition -- each iteration masks
#     eval_frac independently, so some rows are held out several times and
#     some never. fit_cv_ensemble and fit_residual_scale below handle this
#     (residuals are pooled across folds, and min_per_bin counts DISTINCT
#     rows so repeated near-duplicate residuals don't inflate the bin count).

def get_gap_lengths(s: Any) -> np.ndarray:
    """Lengths of consecutive-NaN runs in a series/array (original index units)."""
    arr = pd.Series(np.asarray(s, dtype=float))
    isna = arr.isna()
    if not bool(isna.any()):
        return np.array([], dtype=int)
    run_id = isna.ne(isna.shift()).cumsum()
    run_len = isna.groupby(run_id).size()
    run_isna = isna.groupby(run_id).first()
    return run_len[run_isna].to_numpy().astype(int)


def _geom_pmf(p: float, support: int) -> np.ndarray:
    from scipy.stats import geom
    return np.asarray([geom.pmf(x, p) for x in range(support)], dtype=float)


def convex_combine_geom(pmf1: np.ndarray, p: float, alpha: float) -> np.ndarray:
    """alpha * pmf1 + (1 - alpha) * Geom(p), renormalized."""
    g = _geom_pmf(p, len(pmf1))
    cc = alpha * np.asarray(pmf1, dtype=float) + (1.0 - alpha) * g
    tot = float(cc.sum())
    return cc / tot if tot > 0 else cc


def compile_empirical_gap_dist(
    gap_lengths: np.ndarray,
    *,
    outlier_quant: float = 0.99,
    smooth_tail_start: float = 0.95,
    bandwidth: float = 5.0,
) -> np.ndarray:
    """
    Empirical PMF over gap lengths, with the sparse tail smoothed by a KDE.
    Gaps above `outlier_quant` are dropped (they cannot be validated against
    anyway -- there is no comparably long observed run to hold out).
    """
    gaps = np.asarray(gap_lengths, dtype=int)
    if gaps.size == 0:
        raise ValueError("No gaps available to build a gap-length distribution.")
    gaps = gaps[gaps < np.quantile(gaps, outlier_quant)]
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        raise ValueError("All gaps removed as outliers; lower outlier_quant.")

    hist = np.bincount(gaps)
    pmf = hist / hist.sum()

    idx = np.flatnonzero(np.cumsum(pmf) > smooth_tail_start)
    if idx.size == 0:
        pmf = pmf.copy()
        pmf[0] = 0.0
        return pmf / pmf.sum()
    smooth_start = int(idx[0])

    try:
        from sklearn.neighbors import KernelDensity
    except ImportError:
        out = pmf.copy()
        out[0] = 0.0
        return out / out.sum()

    kd = KernelDensity(bandwidth=bandwidth, kernel="epanechnikov").fit(gaps[:, None])
    smooth = np.exp(kd.score_samples(np.arange(len(pmf))[:, None]))

    true_tail = pmf[smooth_start:]
    smooth_tail = smooth[smooth_start:]
    if smooth_tail.sum() <= 0:
        out = pmf.copy()
    else:
        out = pmf.copy()
        out[smooth_start:] = smooth_tail * (true_tail.sum() / smooth_tail.sum())

    out[0] = 0.0  # never sample zero-length gaps
    return out / out.sum()


def sample_artificial_gaps(
    flux_data: Any,
    sampling_pmf: np.ndarray,
    *,
    eval_frac: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    overlap_retries: int = 20,
    max_iter: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Place non-overlapping artificial gaps into a series until `eval_frac` of
    the OBSERVED values are masked. Gap lengths are i.i.d. draws from
    `sampling_pmf`. Existing (real) gaps are ignored when placing, so an
    artificial window landing next to a real gap effectively extends it --
    this is realistic, and makes the held-out task slightly harder than the
    nominal distribution (i.e. conservative for uncertainty estimation).
    Returns (masked_series, artificially_masked_mask).
    """
    rng = np.random.default_rng() if rng is None else rng
    flux = np.asarray(flux_data, dtype=float)
    n = flux.size
    observed = np.isfinite(flux)
    n_obs = int(observed.sum())
    if n_obs == 0:
        raise ValueError("flux_data has no observed values.")

    pmf = np.clip(np.asarray(sampling_pmf, dtype=float), 0.0, None)
    if pmf.sum() <= 0:
        raise ValueError("sampling_pmf must have positive mass.")
    pmf = pmf / pmf.sum()
    lengths = np.arange(pmf.size)

    masked = np.zeros(n, dtype=bool)
    n_target = int(np.ceil(eval_frac * n_obs))
    n_masked_obs = 0
    it = 0

    while n_masked_obs < n_target and it < max_iter:
        it += 1
        L = int(rng.choice(lengths, p=pmf))
        if L <= 0:
            continue
        for _ in range(overlap_retries):
            start = int(rng.integers(0, n))
            end = min(start + L, n)
            if not masked[start:end].any():
                masked[start:end] = True
                n_masked_obs += int(observed[start:end].sum())
                break

    out = flux.copy()
    out[masked] = np.nan
    return out, masked


def _ecdf_distance(a: Any, b: Any, kind: str = "cvm") -> float:
    """Cramer-von-Mises ('cvm') or Kolmogorov-Smirnov ('ks') distance."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return float("inf")
    allv = np.concatenate([a, b])
    c1 = np.searchsorted(a, allv, side="right") / a.size
    c2 = np.searchsorted(b, allv, side="right") / b.size
    d = np.abs(c1 - c2)
    return float(d.max()) if kind == "ks" else float(np.sum(d ** 2))


def learn_gap_dist(
    flux_data: Any,
    *,
    n_grid: int = 6,
    n_mc: int = 20,
    p_add: float = 0.3,
    dist: str = "cvm",
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Fit the distribution to SAMPLE gaps from, such that the gap structure
    AFTER injection matches the site's real gap structure.

    The correction is needed because injecting gaps into a series that
    already has gaps produces a union biased toward longer runs (artificial
    windows merge with real ones). So the sampling distribution has to be
    biased short: a convex mixture with a geometric, grid-searched over the
    mixing weight `alpha` and geometric parameter `p`, scored by Monte Carlo.

    The raw empirical PMF is always evaluated as a baseline candidate, so
    the returned distribution is never worse than not fitting at all.
    """
    rng = np.random.default_rng(1000) if rng is None else rng
    flux = np.asarray(flux_data, dtype=float)

    gaps = get_gap_lengths(flux)
    if gaps.size == 0:
        raise ValueError("Series has no gaps; cannot learn a gap distribution.")

    base_pmf = compile_empirical_gap_dist(gaps)
    target = gaps[gaps < np.quantile(gaps, 0.95)]

    def _score(pmf: np.ndarray) -> float:
        scores = []
        for _ in range(n_mc):
            union, _ = sample_artificial_gaps(flux, pmf, eval_frac=p_add, rng=rng)
            ug = get_gap_lengths(union)
            if ug.size == 0:
                continue
            ug = ug[ug < np.quantile(ug, 0.95)]
            scores.append(_ecdf_distance(target, ug, kind=dist))
        return float(np.mean(scores)) if scores else float("inf")

    best_pmf = base_pmf
    best_score = _score(base_pmf)
    base_score = best_score

    alphas = np.linspace(0.01, 0.5, n_grid)
    p_lo = float(base_pmf[1]) if base_pmf.size > 1 and base_pmf[1] > 0 else 0.1
    ps = np.linspace(p_lo, min(0.9, 2.0 * p_lo), n_grid)

    for alpha in alphas:
        for p in ps:
            cand = convex_combine_geom(base_pmf, p=float(p), alpha=float(alpha))
            sc = _score(cand)
            if sc < best_score:
                best_score, best_pmf = sc, cand

    if verbose:
        tag = "raw empirical" if best_score == base_score else "fitted mixture"
        print(f"  Gap distribution: {tag} "
              f"({dist} distance {best_score:.4f}; raw baseline {base_score:.4f})")
    return best_pmf


def create_artificial_gap_splits(
    target_full: pd.Series,
    train_index: pd.Index,
    sampling_pmf: np.ndarray,
    *,
    n_splits: int = 20,
    eval_frac: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    min_test: int = 10,
    verbose: bool = True,
) -> Tuple[List[Split], np.ndarray]:
    """
    Build CV splits by repeatedly injecting artificial gaps into the FULL
    series (real time), then mapping the masked timestamps onto positions in
    the training matrix.

    Args:
        target_full: target column on the original index, real NaNs intact.
        train_index: index of the training matrix (subset of target_full).
        sampling_pmf: gap-length distribution (see learn_gap_dist).
        n_splits: number of independent masking iterations. Because masking
            is independent rather than a partition, a row is held out
            Binomial(n_splits, eval_frac) times -- with n_splits=10 and
            eval_frac=0.1 about 35% of rows are never validated, so 20+ is
            recommended to keep distinct coverage high.

    Returns (splits, coverage) where coverage[i] = how many times training
    row i was held out.
    """
    rng = np.random.default_rng() if rng is None else rng
    flux = target_full.to_numpy(dtype=float)

    pos = target_full.index.get_indexer(train_index)
    if (pos < 0).any():
        raise ValueError("train_index contains labels not present in target_full.index")

    splits: List[Split] = []
    coverage = np.zeros(len(train_index), dtype=int)

    for _ in range(n_splits):
        _, masked = sample_artificial_gaps(
            flux, sampling_pmf, eval_frac=eval_frac, rng=rng,
        )
        is_test = masked[pos]
        te = np.flatnonzero(is_test)
        tr = np.flatnonzero(~is_test)
        if te.size < min_test:
            continue
        splits.append((tr, te))
        coverage += is_test.astype(int)

    if not splits:
        raise RuntimeError("No usable artificial-gap splits were generated.")

    if verbose:
        fracs = [len(te) / len(train_index) for _, te in splits]
        never = int((coverage == 0).sum())
        print(f"CV Strategy: artificial gaps in real time; {len(splits)} splits; "
              f"test fractions {min(fracs):.3f}-{max(fracs):.3f}")
        print(f"  Coverage: {never}/{len(train_index)} rows never held out "
              f"({never / len(train_index):.1%}); mean times held out {coverage.mean():.2f}")

    return splits, coverage


# -----------------------------------------------------------------------------
# Parcel / dataset assembly helpers
# -----------------------------------------------------------------------------

def build_df_for_parcel(data_main, target_flux, letter, selected_features, add_trt):
    if "parcel" not in data_main.columns:
        raise KeyError("data_main must contain a 'parcel' column for masking.")

    # Collect data in a dictionary (Prevents fragmentation)
    data_to_build = {}
    # Check available columns once for speed
    cols_available = set(data_main.columns)
    for f in selected_features:
        if f == "trt":
            continue
        fp = f"{f}_parcel{letter}"
        # Select the correct series based on availability
        if fp in cols_available:
            data_to_build[f] = data_main[fp]
        elif f in cols_available:
            data_to_build[f] = data_main[f]
        else:
            raise KeyError(f"Missing feature '{f}' (neither '{fp}' nor '{f}' exists).")

    # Handle 'trt' feature
    if add_trt and "trt" in selected_features:
        # Create a constant series for the whole index
        val = 0 if letter == "A" else 1
        data_to_build["trt"] = pd.Series(val, index=data_main.index)
    # Create the DataFrame ONCE
    df = pd.DataFrame(data_to_build, index=data_main.index)

    # Handle Flux Columns (Bulk add)
    mask = data_main["parcel"].eq(letter)
    flux_cols = [c for c in data_main.columns if c.startswith(target_flux)]
    if flux_cols:
        # Create the masked flux data
        df_flux = data_main[flux_cols].where(mask)
        # Concatenate horizontally (safe against fragmentation)
        df = pd.concat([df, df_flux], axis=1)

    return df


# -----------------------------------------------------------------------------
# Time-series CV splits
# -----------------------------------------------------------------------------

def plot_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    *,
    ncols: int = 2
):
    import math
    import matplotlib.pyplot as plt

    n_splits = len(splits)
    nrows = math.ceil(n_splits / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 2), squeeze=False)
    ax_list = axes.flatten()

    for i, (train_idx, test_idx) in enumerate(splits):
        ax = ax_list[i]
        train_ix = X.iloc[train_idx].index
        test_ix = X.iloc[test_idx].index
        ax.plot(y.loc[train_ix].index, y.loc[train_ix], ".", label="Train")
        ax.plot(y.loc[test_ix].index, y.loc[test_ix], "x", label="Test")
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Split {i+1}")
        ax.legend()

    for j in range(n_splits, len(ax_list)):
        fig.delaxes(ax_list[j])
    fig.tight_layout()
    plt.show()
    return fig, axes


# -----------------------------------------------------------------------------
# RFE Feature Selection
# -----------------------------------------------------------------------------

def _maybe_inverse(inv_y: Optional[Callable[[Any], Any]], x: Any) -> Any:
    if inv_y is None:
        return x
    return inv_y(x)

def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Split],
    *,
    model_factory: Callable[[], object],
    inv_y: Optional[Callable[[Any], Any]] = None,
    step: int = 1,
    min_features: int = 1,
    verbose: bool = True,
    score_mode: str = "composite",  # "rmse" or "composite"
    w_rmse: float = 0.5,
    w_r2: float = 0.5,
    w_penalty: float = 0.001,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Recursive Feature Elimination using mean feature_importances_ across folds.

    Scoring uses mean CV performance across folds (RMSE/R²).
    Returns:
      best_features, feature_ranking, history_df
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if min_features < 1:
        raise ValueError("min_features must be >= 1")

    features = list(X.columns)
    history = []
    removal_order = []  # (feature, iteration)

    iteration = 0
    while len(features) > max(min_features, 1):
        iteration += 1

        importances = []
        rmse_vals = []
        r2_vals = []

        for tr, te in splits:
            X_tr = X[features].iloc[tr]
            y_tr = y.iloc[tr]
            X_te = X[features].iloc[te]
            y_te = y.iloc[te]

            m = model_factory()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)

            # Feature importance
            if not hasattr(m, "feature_importances_"):
                raise AttributeError(
                    "Model has no feature_importances_. "
                    "Use a model that provides it or change the RFE strategy."
                )

            imp = np.asarray(m.feature_importances_, dtype=float)
            if imp.shape[0] != len(features):
                raise ValueError(
                    "feature_importances_ length does not match current feature list."
                )
            importances.append(imp)

            # Score on original scale if inv_y is provided
            y_te_s = _as_1d_float(_maybe_inverse(inv_y, y_te))
            pred_s = _as_1d_float(_maybe_inverse(inv_y, pred))

            rmse_vals.append(_rmse(y_te_s, pred_s))
            r2_vals.append(_r2(y_te_s, pred_s))

        rmse_cv = float(np.nanmean(rmse_vals))
        r2_cv = float(np.nanmean(r2_vals))

        history.append(
            {
                "iteration": iteration,
                "n_features": len(features),
                "features": features.copy(),
                "rmse_cv": rmse_cv,
                "r2_cv": r2_cv,
            }
        )

        mean_imp = np.mean(np.vstack(importances), axis=0)
        n_remove = min(step, len(features) - min_features)
        remove_idx = np.argsort(mean_imp)[:n_remove]  # lowest importance first
        remove_feats = [features[i] for i in remove_idx.tolist()]

        for f in remove_feats:
            features.remove(f)
            removal_order.append((f, iteration))

        if verbose:
            print(
                f"Iter {iteration}: kept={len(features)} removed={remove_feats} "
                f"RMSE_cv={rmse_cv:.4f} R2_cv={r2_cv:.4f}"
            )

    # add survivors to ranking (last survivors rank highest)
    for f in features:
        removal_order.append((f, iteration + 1))
    feature_ranking = [f for f, _ in sorted(removal_order, key=lambda x: x[1], reverse=True)]

    hist_df = pd.DataFrame(history)

    # choose best subset
    if hist_df.empty:
        return features.copy(), feature_ranking, hist_df

    if score_mode == "rmse":
        best_row = hist_df.sort_values(["rmse_cv", "n_features"], ascending=[True, True]).iloc[0]
        best_features = list(best_row["features"])
        return best_features, feature_ranking, hist_df

    if score_mode == "composite":
        rmse_vals = hist_df["rmse_cv"].to_numpy(float)
        r2_vals = hist_df["r2_cv"].to_numpy(float)
        nfeat = hist_df["n_features"].to_numpy(float)

        def _minmax(a):
            a_min, a_max = float(np.min(a)), float(np.max(a))
            if np.isclose(a_min, a_max):
                return np.zeros_like(a, dtype=float)
            return (a - a_min) / (a_max - a_min)

        norm_rmse = _minmax(rmse_vals)
        norm_r2 = _minmax(r2_vals)

        comp = w_rmse * norm_rmse + w_r2 * (1.0 - norm_r2) + w_penalty * nfeat
        best_i = int(np.argmin(comp))
        best_features = list(hist_df.iloc[best_i]["features"])
        return best_features, feature_ranking, hist_df

    raise ValueError("score_mode must be 'rmse' or 'composite'")

# -----------------------------------------------------------------------------
# CV Model Fitting & OOF
# -----------------------------------------------------------------------------

def fit_cv_ensemble(
    model_factory: Callable[[], object],
    X: pd.DataFrame,
    y_train: pd.Series,
    splits: List[Split],
    *,
    inv_y: Optional[Callable[[Any], Any]] = None,
) -> Tuple[pd.Series, List[object], pd.DataFrame, np.ndarray]:
    """
    Trains one model per CV split and generates Out-Of-Fold predictions.

    Handles splits that are NOT a partition (artificial-gap splits mask
    eval_frac independently each iteration, so a row can be held out several
    times or never). Every OOF prediction is kept in `records` rather than
    overwritten; `oof` averages them per row for plotting/diagnostics.

    Returns
    -------
    oof : pd.Series
        Aligned with X.index. Mean OOF prediction per row, NaN where the row
        was never held out. With a true partition this is exactly the single
        OOF prediction, so behaviour is unchanged for block splits.
    models : List[object]
        One trained model per split -- the ensemble used for sigma_ens.
    records : pd.DataFrame
        Long format, one row per (fold, held-out row): [fold, pos, yhat].
        This is what residual calibration should use, since it preserves
        every residual instead of collapsing repeats.
    coverage : np.ndarray
        How many times each training row was held out.
    """
    n = len(X)
    sum_pred = np.zeros(n, dtype=float)
    coverage = np.zeros(n, dtype=int)
    models: List[object] = []
    recs: List[pd.DataFrame] = []

    for k, (tr, te) in enumerate(splits):
        m = model_factory()
        m.fit(X.iloc[tr], y_train.iloc[tr])
        models.append(m)

        pred = m.predict(X.iloc[te]).astype(float)
        pred = _as_1d_float(_maybe_inverse(inv_y, pred))

        te = np.asarray(te, dtype=int)
        sum_pred[te] += pred
        coverage[te] += 1
        recs.append(pd.DataFrame({"fold": k, "pos": te, "yhat": pred}))

    denom = np.where(coverage == 0, 1, coverage)
    oof = pd.Series(np.where(coverage > 0, sum_pred / denom, np.nan), index=X.index)
    records = (pd.concat(recs, ignore_index=True) if recs
               else pd.DataFrame(columns=["fold", "pos", "yhat"]))
    return oof, models, records, coverage


def _records_metrics(
    y_true_vals: np.ndarray,
    records: pd.DataFrame,
    row_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Per-fold RMSE/MAE/R2 from OOF records (correct even when folds overlap).

    row_mask, if given, is a boolean array indexed the same way as
    y_true_vals (i.e. by position in the training matrix) -- e.g. parcel=='A'
    -- and restricts the metric to that subset. Folds with zero matching
    rows are skipped rather than producing NaN/empty-mean warnings.

    "n" in the returned dict is the number of DISTINCT rows that actually
    contributed (post-mask, post-held-out), not the size of row_mask itself.
    """
    rmse_vals, mae_vals, r2_vals = [], [], []
    seen_pos = set()
    for _, g in records.groupby("fold"):
        pos = g["pos"].to_numpy(dtype=int)
        yp = g["yhat"].to_numpy(dtype=float)
        if row_mask is not None:
            keep = row_mask[pos]
            if not keep.any():
                continue
            pos, yp = pos[keep], yp[keep]
        yt = y_true_vals[pos]
        rmse_vals.append(_rmse(yt, yp))
        mae_vals.append(_mae(yt, yp))
        r2_vals.append(_r2(yt, yp))
        seen_pos.update(pos.tolist())
    if not rmse_vals:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "n": 0}
    return {
        "rmse": float(np.nanmean(rmse_vals)),
        "mae": float(np.nanmean(mae_vals)),
        "r2": float(np.nanmean(r2_vals)),
        "n": len(seen_pos),
    }


@dataclass
class ResidualScaleModel:
    """Maps a prediction yhat to a residual scale sigma_resid(yhat)."""
    method: str
    sigma_global: float
    bin_edges: Optional[np.ndarray] = None
    sigma_by_bin: Optional[np.ndarray] = None

    def sigma(self, yhat: Any) -> np.ndarray:
        yhat = _as_1d_float(yhat)
        if self.method == "global" or self.bin_edges is None or self.sigma_by_bin is None:
            return np.full_like(yhat, self.sigma_global, dtype=float)
        b = np.digitize(yhat, self.bin_edges, right=True) - 1
        b = np.clip(b, 0, len(self.sigma_by_bin) - 1)
        return self.sigma_by_bin[b].astype(float)


def fit_residual_scale(
    y_obs_raw: Any,
    yhat_oof_raw: Any,
    *,
    method: str = "by_pred_quantile",
    q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    min_per_bin: int = 200,
    row_ids: Optional[np.ndarray] = None,
) -> ResidualScaleModel:
    """
    Residual scale sigma_resid(yhat), binned by predicted-flux quantile.

    If `row_ids` is given, inputs are treated as flat arrays of pooled OOF
    records (one entry per fold-row pair) and `min_per_bin` is checked
    against the number of DISTINCT rows in a bin rather than the raw count.
    That matters with non-partition splits: a row held out twice contributes
    two near-identical residuals (same features, models sharing most of
    their training data), so the raw count overstates how much independent
    information the bin actually holds -- most acutely in the top quantile
    bin, which is both the smallest and the one that dominates the budget.
    """
    if row_ids is None:
        df = pd.DataFrame({"y": y_obs_raw, "yhat": yhat_oof_raw}).dropna()
        y = df["y"].to_numpy(dtype=float)
        yh = df["yhat"].to_numpy(dtype=float)
        rid = np.arange(len(df))
    else:
        y = _as_1d_float(y_obs_raw)
        yh = _as_1d_float(yhat_oof_raw)
        rid = np.asarray(row_ids)
        keep = np.isfinite(y) & np.isfinite(yh)
        y, yh, rid = y[keep], yh[keep], rid[keep]

    resid = y - yh
    sigma_global = float(np.nanstd(resid, ddof=1))

    if method == "global":
        return ResidualScaleModel(method="global", sigma_global=sigma_global)

    qs = np.unique(np.array(list(q), dtype=float))
    if qs[0] != 0.0: qs = np.insert(qs, 0, 0.0)
    if qs[-1] != 1.0: qs = np.append(qs, 1.0)

    edges = np.quantile(yh, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf

    bin_id = np.digitize(yh, edges, right=True) - 1
    n_bins = len(edges) - 1

    sigmas = np.full(n_bins, sigma_global, dtype=float)
    for b in range(n_bins):
        mask = bin_id == b
        if int(np.unique(rid[mask]).size) < min_per_bin:
            continue
        sigmas[b] = float(np.nanstd(resid[mask], ddof=1))

    return ResidualScaleModel(
        method="by_pred_quantile",
        sigma_global=sigma_global,
        bin_edges=edges,
        sigma_by_bin=sigmas,
    )


def combine_gapfill_sigma(
    sigma_ens: Any,
    sigma_resid: Any,
    *,
    min_sigma: float = 0.0,
) -> np.ndarray:
    """
    sigma_gf = sqrt(sigma_ens^2 + sigma_resid^2), with optional floor.

    This is the POINTWISE (per-half-hour) uncertainty, for per-row bands.
    For a cumulative/seasonal total do NOT quadrature-sum this: use
    sigma_ens_cumulative + cumulative_gapfill_uncertainty below, which
    accumulates the structural term correctly.
    """
    sigma_ens = _as_1d_float(sigma_ens)
    sigma_resid = _as_1d_float(sigma_resid)
    sgf = np.sqrt(np.square(sigma_ens) + np.square(sigma_resid))
    if min_sigma > 0:
        sgf = np.maximum(sgf, float(min_sigma))
    return sgf


# -----------------------------------------------------------------------------
# Cumulative uncertainty (correlation-aware structural term)
# -----------------------------------------------------------------------------

def predict_ensemble_matrix(
    ensemble_models: List[object],
    X_full: pd.DataFrame,
    inv_y: Optional[Callable[[Any], Any]] = None,
) -> np.ndarray:
    """Predictions from every ensemble member, stacked as (K models, T rows)."""
    preds = np.full((len(ensemble_models), len(X_full)), np.nan, dtype=float)
    for i, m in enumerate(ensemble_models):
        p = m.predict(X_full).astype(float)
        preds[i, :] = _as_1d_float(_maybe_inverse(inv_y, p))
    return preds


def sigma_ens_cumulative(
    fit: "GapfillFit",
    df_pred: pd.DataFrame,
    *,
    is_gap: Optional[pd.Series] = None,
    return_paths: bool = False,
) -> Any:
    """
    Structural uncertainty of the CUMULATIVE sum: keep each ensemble
    member's full predicted trajectory, cumsum each one separately (masked
    to gap-filled rows), and take the spread across the K cumulative sums.

    This is the whole point of carrying the ensemble through: collapsing to
    a pointwise std and quadrature-summing assumes the structural error
    re-randomizes every half hour, when in fact a member biased in some
    regime stays biased for the entire gap. Accumulating the member paths
    captures that correlation empirically, with no assumed correlation
    length -- and it makes the result sensitive to gap CONTIGUITY, not just
    to how many rows are filled.

    Set return_paths=True to also get the (K, T) matrix of cumulative paths,
    e.g. to build a parcel A - B contrast per member.
    """
    _check_columns(df_pred, fit.feature_cols, name="df_pred")
    preds = predict_ensemble_matrix(fit.ensemble_models, df_pred[fit.feature_cols], inv_y=fit.inv_y)

    if is_gap is not None:
        mask = is_gap.reindex(df_pred.index).fillna(False).to_numpy(dtype=bool)
        preds = preds * mask[None, :]

    cum_paths = np.cumsum(preds, axis=1)
    sigma_cum = pd.Series(np.std(cum_paths, axis=0, ddof=1),
                          index=df_pred.index, name="sigma_ens_cum")
    if return_paths:
        return sigma_cum, cum_paths
    return sigma_cum


def cumulative_gapfill_uncertainty(
    y_obs: pd.Series,
    y_hat: pd.Series,
    sigma_obs: pd.Series,
    sigma_resid: pd.Series,
    sigma_ens_cum: pd.Series,
    is_gap: pd.Series,
    *,
    z: float = 1.96,
) -> pd.DataFrame:
    """
    Cumulative flux and its uncertainty.

        sigma_err(T)^2 = sigma_ens_cum(T)^2
                       + sum_{gap-filled, t<=T} sigma_resid(t)^2
                       + sum_{measured,   t<=T} sigma_obs(t)^2

    sigma_obs and sigma_resid stay plain quadrature sums (their errors are
    short-correlated and genuinely do average down); only the structural
    term needs the trajectory treatment.

    Returns cum_flux / sigma_err / ci_low / ci_high; read the last row for
    the seasonal total, or plot the whole thing as a time-resolved band.
    """
    idx = y_obs.index
    is_gap = is_gap.reindex(idx).fillna(False)

    cum_flux = y_obs.where(~is_gap, y_hat).cumsum()
    sq_obs = (sigma_obs.reindex(idx).where(~is_gap, 0.0).fillna(0.0) ** 2).cumsum()
    sq_res = (sigma_resid.reindex(idx).where(is_gap, 0.0).fillna(0.0) ** 2).cumsum()
    sq_ens = sigma_ens_cum.reindex(idx).ffill().fillna(0.0) ** 2

    sigma_err = np.sqrt(sq_obs + sq_res + sq_ens)
    return pd.DataFrame({
        "cum_flux": cum_flux,
        "sigma_err": sigma_err,
        "ci_low": cum_flux - z * sigma_err,
        "ci_high": cum_flux + z * sigma_err,
    })


def sigma_turb(cum_totals: Iterable[float]) -> float:
    """u*-threshold term: half the range across the U16/U50/U84 totals."""
    vals = np.asarray(list(cum_totals), dtype=float)
    return float((np.nanmax(vals) - np.nanmin(vals)) / 2.0)


def combine_with_turb(sigma_err_final: float, turb: float) -> float:
    """Combine propagated flux uncertainty with sigma_turb in quadrature."""
    return float(np.sqrt(float(sigma_err_final) ** 2 + float(turb) ** 2))


# -----------------------------------------------------------------------------
# Convenience pipeline: fit/apply gapfiller with uncertainty
# -----------------------------------------------------------------------------

@dataclass
class GapfillFit:
    target_col: str
    feature_cols: List[str]
    model_factory: Callable[[], object]
    model_final: object
    ensemble_models: List[object]   # The K models -> feeds sigma_ens AND the
                                    # cumulative member-path spread
    X_tr: pd.DataFrame
    y_tr: pd.Series
    y_raw: pd.Series
    inv_y: Optional[Callable[[Any], Any]]
    splits: List[Split]
    yhat_oof_raw: pd.Series         # mean OOF per row (NaN if never held out)
    resid_model: ResidualScaleModel
    random_state: int
    oof_records: Optional[pd.DataFrame] = None   # long [fold, pos, yhat]
    oof_coverage: Optional[np.ndarray] = None    # times each row was held out
    sampling_pmf: Optional[np.ndarray] = None    # learned gap-length pmf

    def oof_frame(self) -> pd.DataFrame:
        """
        Pooled OOF residuals as a tidy frame: timestamp, fold, y, yhat, resid.
        Use this for residual diagnostics (ACF, bias by regime) rather than
        yhat_oof_raw, which averages away repeated held-out predictions.
        """
        if self.oof_records is None or self.oof_records.empty:
            raise ValueError("No OOF records stored on this fit.")
        pos = self.oof_records["pos"].to_numpy(dtype=int)
        y = self.y_raw.to_numpy(dtype=float)[pos]
        out = pd.DataFrame({
            "timestamp": self.X_tr.index[pos],
            "fold": self.oof_records["fold"].to_numpy(),
            "y": y,
            "yhat": self.oof_records["yhat"].to_numpy(dtype=float),
        })
        out["resid"] = out["y"] - out["yhat"]
        return out


# -----------------------------------------------------------------------------
# Internal Plotting Helper (Triggered inside fit_gapfill_ts)
# -----------------------------------------------------------------------------

_DEFAULT_GROUP_COLORS = {'A': "#D55E00", 'B': "#0072B2"}
_DEFAULT_GROUP_LABELS = {'A': "BAU", 'B': "VRA"}


def _panel_label(ax, letter: str) -> None:
    """Bold (a)/(b)/... panel label, placed where a title would go."""
    ax.text(-0.08, 1.05, f"({letter})", transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='bottom', ha='left')


def _plot_internal_diagnostics(
    y_raw: pd.Series,
    oof_records: pd.DataFrame,
    ensemble_models: List[object],
    model_final: object,
    X: pd.DataFrame,
    inv_y: Optional[Callable],
    feature_cols: List[str],
    target_name: str,
    group_vals: Optional[pd.Series] = None,
    group_colors: Optional[Dict[str, str]] = None,
    group_labels: Optional[Dict[str, str]] = None,
):
    """
    Plots 2x2 diagnostics. Panels are labeled (a)-(d) instead of titled:
    (a) Obs vs Pred (Scatter) -- raw per-fold OOF predictions, one point per
        (fold, held-out row) pair, matching the metrics printed by
        fit_gapfill_ts. NOT the row-averaged series: averaging repeated OOF
        predictions before scoring understates the error a single
        fold-model actually makes. Colored by `group_vals` (e.g. parcel
        identity) when given, with overall + per-group RMSE/MAE/R2
        annotated (per-group metrics use the same fold-then-average method
        as the overall number -- see _records_metrics).

        group_vals must be indexed like X (X.index), NOT a column of X --
        e.g. parcel identity ('A'/'B') typically isn't a model feature, so
        it has to be supplied separately from whatever full dataframe it
        actually lives in.
    (b) Feature Importance
    (c) Ensemble Time Series (Instantaneous)
    (d) Ensemble Time Series (Cumulative)
    """
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    import os

    group_colors = group_colors or _DEFAULT_GROUP_COLORS
    group_labels = group_labels or _DEFAULT_GROUP_LABELS

    # define the fontsite for the plots
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- (a) Obs vs Pred (raw per-fold OOF), colored by group ---
    ax1 = axes[0, 0]
    y_vals = y_raw.to_numpy(dtype=float)
    pos = oof_records["pos"].to_numpy(dtype=int)
    obs_raw = y_vals[pos]
    pred_raw = oof_records["yhat"].to_numpy(dtype=float)

    group_arr = group_vals.reindex(X.index).to_numpy() if group_vals is not None else None
    group_oof = group_arr[pos] if group_arr is not None else None

    df_plot = pd.DataFrame({"Obs": obs_raw, "Pred": pred_raw})
    if group_oof is not None:
        df_plot["group"] = group_oof
    df_plot = df_plot.dropna(subset=["Obs", "Pred"])

    overall = _records_metrics(y_vals, oof_records)
    metric_lines = [f"Overall: RMSE={overall['rmse']:.3f}  "
                     f"MAE={overall['mae']:.3f}  R\u00b2={overall['r2']:.3f}"]

    if len(df_plot) > 1:
        x, y = df_plot["Pred"], df_plot["Obs"]
        slope, intercept, r_val, p_val, std_err = linregress(x, y)

        if group_oof is not None:
            for key in sorted(group_colors):
                m_plot = df_plot["group"] == key
                if m_plot.any():
                    label = f"{group_labels.get(key, key)}"
                    ax1.scatter(df_plot.loc[m_plot, "Pred"], df_plot.loc[m_plot, "Obs"],
                                alpha=0.25, s=10, c=group_colors[key], label=label)
                row_mask = (group_arr == key)
                m = _records_metrics(y_vals, oof_records, row_mask=row_mask)
                if m["n"] > 0:
                    metric_lines.append(
                        f"{group_labels.get(key, key)}: "
                        f"RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R\u00b2={m['r2']:.3f}"
                    )
        else:
            ax1.scatter(x, y, alpha=0.15, s=8, c='k', label='Data (raw per-fold OOF)')

        min_v, max_v = min(x.min(), y.min()), max(x.max(), y.max())
        ax1.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1, label='1:1')
        ax1.plot(np.array([min_v, max_v]), slope * np.array([min_v, max_v]) + intercept,
                  'r-', lw=1.5, label=f"Fit (r={r_val:.2f})")
        ax1.set_xlabel(f"Predicted {target_name}")
        ax1.set_ylabel(f"Observed {target_name}")
        ax1.legend(loc='lower right', frameon=False)
        ax1.text(0.03, 0.97, "\n".join(metric_lines), transform=ax1.transAxes, va='top', ha='left')
    else:
        ax1.text(0.5, 0.5, "No Data", ha='center')
    _panel_label(ax1, 'a')

    # --- (b) Feature Importance ---
    ax2 = axes[0, 1]
    importances = []
    for m in ensemble_models:
        if hasattr(m, 'feature_importances_'):
            importances.append(m.feature_importances_)
        elif hasattr(m, 'get_score'): # Native XGBoost
            scores = m.get_score(importance_type='gain')
            imp = np.zeros(len(feature_cols))
            for f, score in scores.items():
                if f in feature_cols: imp[feature_cols.index(f)] = score
            importances.append(imp)

    if importances:
        avg_imp = np.mean(importances, axis=0)
        std_imp = np.std(importances, axis=0)
        indices = np.argsort(avg_imp)[-10:]
        ax2.barh(range(len(indices)), avg_imp[indices], xerr=std_imp[indices], align='center', capsize=3)
        ax2.set_yticks(range(len(indices)))
        ax2.set_yticklabels(np.array(feature_cols)[indices])
        ax2.set_xlabel(f"Mean importance (gain), {len(ensemble_models)} folds")
    else:
        ax2.text(0.5, 0.5, "Importance N/A", ha='center')
    _panel_label(ax2, 'b')

    # --- (c) & (d): Time Series Predictions ---
    X_sorted = X.sort_index()
    ens_preds = []
    for m in ensemble_models:
        p = m.predict(X_sorted)
        if inv_y: p = inv_y(p)
        ens_preds.append(_as_1d_float(p))

    p_final = model_final.predict(X_sorted)
    if inv_y: p_final = inv_y(p_final)
    p_final = _as_1d_float(p_final)
    x_axis = X_sorted.index

    ax3 = axes[1, 0]
    for p in ens_preds:
        # add line for legend only for the first member
        label = 'CV ensemble members' if p is ens_preds[0] else None
        ax3.plot(x_axis, p, color='gray', alpha=0.5, label=label)
    ax3.plot(x_axis, p_final, color='red', alpha=1, label='Final Model')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylabel(f"{target_name}")
    ax3.legend(frameon=False)
    _panel_label(ax3, 'c')

    ax4 = axes[1, 1]
    for p in ens_preds:
        # add line for legend only for the first member
        label = 'CV ensemble members' if p is ens_preds[0] else None
        ax4.plot(x_axis, np.cumsum(p), color='gray', alpha=0.5, label=label)
    ax4.plot(x_axis, np.cumsum(p_final), color='red', label='Final Model')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel(f"Cum. {target_name}")
    ax4.legend(frameon=False)
    _panel_label(ax4, 'd')

    plt.tight_layout()
    save_path = f'plots/gapfilling_diagnostics_{target_name}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()


def fit_gapfill_ts(
    df: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: List[str],
    model_factory: Callable[[], object],
    log_transform: bool = False,
    undersample: bool = False,
    undersample_quantile: float = 0.8,
    undersample_fraction: float = 0.5,
    n_splits: int = 20,
    eval_frac: float = 0.1,
    gap_dist_fit: bool = True,
    gap_dist_n_grid: int = 6,
    gap_dist_n_mc: int = 20,
    sampling_pmf: Optional[np.ndarray] = None,
    random_state: int = 42,
    resid_method: str = "by_pred_quantile",
    resid_q: Iterable[float] = (0.0, 0.5, 0.8, 0.95, 1.0),
    resid_min_per_bin: int = 100,
    plot_group_col: Optional[str] = "parcel",
    plot_group_colors: Optional[Dict[str, str]] = None,
    plot_group_labels: Optional[Dict[str, str]] = None,
    verbose: bool = True,
    plot: bool = True
) -> GapfillFit:
    """
    Fit a final model + uncertainty components using a CV ensemble.

    CV splits hold out windows positioned in REAL TIME, with lengths drawn
    from a distribution fitted to the site's own gap structure (Irvin et al.
    2021). Held-out windows then have realistic depth, so OOF residuals
    reflect mid-gap behaviour rather than always sitting one step from an
    observation. This is NOT a partition: with n_splits=20, eval_frac=0.1
    roughly 12% of rows are never held out and others are held out several
    times, which is why min_per_bin (in fit_residual_scale) counts distinct
    rows rather than raw residual count, and why OOF metrics are computed
    per (fold, row) pair rather than on a row-averaged series.

    Pass a precomputed `sampling_pmf` (or gap_dist_fit=False) to skip the
    grid search -- useful when refitting the same series repeatedly.
    """
    _check_columns(df, [target_col] + list(feature_cols), name="df")

    out = df.copy()

    train_mask = out[target_col].notna() & out[feature_cols].notna().all(axis=1)
    if train_mask.sum() < 10:
        raise ValueError(f"Not enough complete training rows for {target_col}.")

    df_train = out.loc[train_mask, feature_cols + [target_col]].copy()
    df_train = df_train.rename(columns={target_col: "_y_raw"})
    y_raw = df_train["_y_raw"].astype(float).copy()

    if undersample:
        df_train, cutoff = undersample_target(
            df_train, "_y_raw",
            quantile_cutoff=undersample_quantile,
            fraction=undersample_fraction,
            random_state=random_state,
            verbose=verbose,
        )
        y_raw = df_train["_y_raw"].astype(float).copy()

    inv_y = None
    if log_transform:
        log_fn, inv_fn, min_value, _ = setup_log_transform(df_train, "_y_raw")
        df_train["_y_train"] = log_fn(df_train["_y_raw"].to_numpy(float))
        inv_y = inv_fn
    else:
        df_train["_y_train"] = df_train["_y_raw"].to_numpy(float)

    X_tr = df_train[feature_cols].copy()
    y_tr = pd.Series(df_train["_y_train"].to_numpy(float), index=df_train.index)

    # CV Splits: windows positioned in real time, lengths from the site's
    # own gap-length distribution
    rng = np.random.default_rng(random_state)

    if sampling_pmf is None:
        if gap_dist_fit:
            sampling_pmf = learn_gap_dist(
                out[target_col].to_numpy(dtype=float),
                n_grid=gap_dist_n_grid, n_mc=gap_dist_n_mc,
                rng=rng, verbose=verbose,
            )
        else:
            sampling_pmf = compile_empirical_gap_dist(get_gap_lengths(out[target_col]))
            if verbose:
                print("  Gap distribution: raw empirical (grid search skipped)")

    splits, oof_coverage = create_artificial_gap_splits(
        target_full=out[target_col],
        train_index=X_tr.index,
        sampling_pmf=sampling_pmf,
        n_splits=n_splits,
        eval_frac=eval_frac,
        rng=rng,
        verbose=verbose,
    )

    # Plot CV Splits
    if plot:
        # Note: We pass y_raw so the plot shows the actual data structure
        plot_cv_splits(X_tr, y_raw, splits)

    # Fit CV Ensemble
    if verbose:
        print(f"Training CV Ensemble ({len(splits)} folds) & generating OOF predictions...")

    yhat_oof_raw, ensemble_models, oof_records, coverage = fit_cv_ensemble(
        model_factory=model_factory,
        X=X_tr,
        y_train=y_tr,
        splits=splits,
        inv_y=inv_y,
    )
    if oof_coverage is None:
        oof_coverage = coverage

    # Performance metrics, computed per fold from the pooled records so
    # overlapping folds are scored correctly
    cv_metrics = _records_metrics(y_raw.to_numpy(dtype=float), oof_records)
    rmse_score = cv_metrics["rmse"]
    mae_score = cv_metrics["mae"]
    r2_score = cv_metrics["r2"]
    if verbose:
        print("-" * 40)
        print(f"Gapfilling performance (mean across CV folds):")
        print(f"  Target: {target_col}")
        print(f"  RMSE:   {rmse_score:.4f}")
        print(f"  MAE:    {mae_score:.4f}")
        print(f"  R2:     {r2_score:.4f}")
        print("-" * 40)

    # Residual scale from ALL pooled OOF residuals, with min_per_bin gated on
    # distinct rows (see fit_residual_scale docstring)
    _pos = oof_records["pos"].to_numpy(dtype=int)
    resid_model = fit_residual_scale(
        y_obs_raw=y_raw.to_numpy(dtype=float)[_pos],
        yhat_oof_raw=oof_records["yhat"].to_numpy(dtype=float),
        method=resid_method,
        q=resid_q,
        min_per_bin=resid_min_per_bin,
        row_ids=_pos,
    )

    # Final Model (Best Estimate trained on all data)
    if verbose:
        print("Fitting final model on full training set...")
    m_final = model_factory()
    m_final.fit(X_tr, y_tr)

    # Plots for diagnostics
    if plot:
            group_vals = None
            if plot_group_col is not None and plot_group_col in out.columns:
                group_vals = out[plot_group_col].reindex(X_tr.index)
            _plot_internal_diagnostics(
                y_raw=y_raw, oof_records=oof_records, ensemble_models=ensemble_models,
                model_final=m_final, X=X_tr, inv_y=inv_y, feature_cols=list(feature_cols),
                target_name=target_col, group_vals=group_vals,
                group_colors=plot_group_colors, group_labels=plot_group_labels,
            )

    return GapfillFit(
        target_col=target_col,
        feature_cols=list(feature_cols),
        model_factory=model_factory,
        model_final=m_final,
        ensemble_models=ensemble_models, # CV models stored here
        X_tr=X_tr,
        y_tr=y_tr,
        y_raw=y_raw,
        inv_y=inv_y,
        splits=splits,
        yhat_oof_raw=yhat_oof_raw,
        resid_model=resid_model,
        random_state=int(random_state),
        oof_records=oof_records,
        oof_coverage=oof_coverage,
        sampling_pmf=sampling_pmf,
    )


def apply_gapfill_ts(
    df_pred: pd.DataFrame,
    fit: GapfillFit,
    *,
    prefix: str = "GF",
    min_sigma: float = 0.0,
    is_gap: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Apply the fitted model.
    - Prediction = fit.model_final
    - Sigma_Ens = StdDev of predictions from fit.ensemble_models (CV models)

    Also emits {prefix}_sigmaEnsCum: the cumulative structural uncertainty
    from the member paths, masked to gap-filled rows. Reuses the per-member
    predictions already computed for sigma_ens -- no extra model.predict()
    calls. Use sigmaEnsCum -- NOT a quadrature sum of sigmaGF -- with
    cumulative_gapfill_uncertainty for seasonal totals.

    Which rows count as "gap-filled" is is_gap = y_obs.isna() & y_hat.notna().
    y_hat is always computed here; y_obs is taken from df_pred[fit.target_col]
    if that column is present (it is whenever df_pred is a full site/parcel
    dataframe rather than a features-only slice), so is_gap is derived
    automatically in the common case -- no second call needed. Pass `is_gap`
    explicitly only to override this, e.g. a custom definition of "gap", or
    if df_pred genuinely doesn't carry the target column.
    """
    _check_columns(df_pred, fit.feature_cols, name="df_pred")

    out = df_pred.copy()
    X_full = out[fit.feature_cols].copy()

    # Main Prediction
    yhat = fit.model_final.predict(X_full).astype(float)
    yhat = _as_1d_float(_maybe_inverse(fit.inv_y, yhat))

    if is_gap is None and fit.target_col in out.columns:
        is_gap = out[fit.target_col].isna() & pd.Series(np.isfinite(yhat), index=out.index)

    # Ensemble Prediction (using CV models)
    preds_ens = None
    if fit.ensemble_models:
        preds_ens = predict_ensemble_matrix(fit.ensemble_models, X_full, inv_y=fit.inv_y)
        sigma_ens = _as_1d_float(np.nanstd(preds_ens, axis=0, ddof=1))
    else:
        sigma_ens = 0.0

    # Residual Uncertainty
    sigma_resid = _as_1d_float(fit.resid_model.sigma(yhat))
    
    # Combined
    sigma_gf = combine_gapfill_sigma(sigma_ens, sigma_resid, min_sigma=min_sigma)

    out[f"{prefix}_yhat"] = yhat
    out[f"{prefix}_sigmaEns"] = sigma_ens
    out[f"{prefix}_sigmaResid"] = sigma_resid
    out[f"{prefix}_sigmaGF"] = sigma_gf

    if preds_ens is not None and is_gap is not None:
        mask = is_gap.reindex(out.index).fillna(False).to_numpy(dtype=bool)
        cum_paths = np.cumsum(preds_ens * mask[None, :], axis=1)
        out[f"{prefix}_sigmaEnsCum"] = np.std(cum_paths, axis=0, ddof=1)

    return out


# -----------------------------------------------------------------------------
# High-Level Reusable Workflows
# -----------------------------------------------------------------------------

@dataclass
class CumulativeContext:
    """
    What's needed to rebuild a correlation-aware cumulative uncertainty band
    for one column, restarted fresh at any period's start -- matching how
    the plotted flux cumsum itself restarts per displayed period. A single
    precomputed whole-season sigma_ens_cum series can't be sliced to a
    sub-period by subtracting standard deviations (that's not a valid way to
    get a sub-range's variance); it has to be recomputed from the actual
    ensemble predictions over that period's rows.
    """
    fit: "GapfillFit"
    df_view: pd.DataFrame          # must contain fit.feature_cols, same index as y_obs
    sigma_obs: pd.Series
    is_gap: Optional[pd.Series]    # None => every row counts (e.g. a pure "Predicted" column)


def merge_gapfill_results(
    main_df: pd.DataFrame,
    views: List[Tuple[str, pd.DataFrame, pd.DataFrame, "GapfillFit"]],
    target_flux: str,
    target: str,
    model_type: str,
    ustar_cut: str,
    random_err_col: str,
    prefix: str = "GF",
    qc_levels: List[str] = ["QCF", "QCF0"],
    store_uncertainty_inputs: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, CumulativeContext]]:
    """
    Merges gap-filling predictions into main dataframe.
 
    `views` now carries the GapfillFit for each view -- (view_name, df_view,
    pred_df, fit) -- so a correlation-aware cumulative band can be rebuilt
    later for any period. Returns (df_final, contexts): contexts maps each
    filled/predicted column name to what plot_gapfill_dashboard needs to
    recompute its cumulative uncertainty band.
 
    store_uncertainty_inputs: if True, also stores the raw per-fold
    ensemble predictions ("{gf_base_name}_ens00".."_ens{K-1}") and the
    pointwise sigma_obs ("{gf_base_name}_sigmaObs") for each gap-filled
    column.
    """
    columns: Dict[str, pd.Series] = {}
    contexts: Dict[str, CumulativeContext] = {}
    target_base_root = f"{target_flux}_L3.3_{ustar_cut}"
    
    for view_name, df_view, pred_df, fit in views:
        for qc in qc_levels:
            obs_col = f"{target_base_root}_{qc}"
            y_obs = df_view[obs_col].astype(float)
            y_hat = pred_df[f"{prefix}_yhat"]
            sigma_obs = df_view[random_err_col]
            sigma_ens = pred_df[f"{prefix}_sigmaEns"]
            sigma_resid = pred_df[f"{prefix}_sigmaResid"]
            sigma_gf = pred_df[f"{prefix}_sigmaGF"]
            is_gap = y_obs.isna() & y_hat.notna()
            
            obs_col_out   = f"{target_base_root}_{qc}_{view_name}"
            gf_base_name  = f"{target_base_root}_{qc}_{view_name}_gf{model_type}"
            col_filled    = gf_base_name
            col_isfilled  = f"FLAG_{gf_base_name}_ISFILLED"
            col_total_unc = f"{gf_base_name}_total_unc"
            col_sigmaEns = f"{gf_base_name}_sigmaEns"
            col_sigmaResid = f"{gf_base_name}_sigmaResid"
            col_sigmaGF = f"{gf_base_name}_sigmaGF"
            col_pred_only = f"{gf_base_name}_yhat"
            
            columns[obs_col_out] = y_obs.reindex(main_df.index)
            columns[col_pred_only] = y_hat.reindex(main_df.index)
            columns[col_filled] = y_obs.where(~is_gap, y_hat).reindex(main_df.index)
            columns[col_isfilled] = is_gap.astype(int).reindex(main_df.index)
            columns[col_sigmaGF] = sigma_gf.reindex(main_df.index)          # POINTWISE only -- fine for
            columns[col_sigmaResid] = sigma_resid.reindex(main_df.index)    # per-row bands, NOT for a
            columns[col_sigmaEns] = sigma_ens.reindex(main_df.index)        # cumulative sum (see contexts)
            columns[col_total_unc] = sigma_obs.where(~is_gap, sigma_gf).reindex(main_df.index)
 
            if store_uncertainty_inputs:
                columns[f"{gf_base_name}_sigmaObs"] = sigma_obs.reindex(main_df.index)
                preds_ens = predict_ensemble_matrix(fit.ensemble_models, df_view[fit.feature_cols], inv_y=fit.inv_y)
                for k in range(preds_ens.shape[0]):
                    columns[f"{gf_base_name}_ens{k:02d}"] = \
                        pd.Series(preds_ens[k], index=df_view.index).reindex(main_df.index)
 
            # Everything needed to rebuild a correct cumulative band later,
            # for this column, restarted fresh at any period's start.
            ctx = CumulativeContext(fit=fit, df_view=df_view, sigma_obs=sigma_obs, is_gap=is_gap)
            contexts[col_filled] = ctx
            contexts[col_pred_only] = CumulativeContext(
                fit=fit, df_view=df_view, sigma_obs=sigma_obs, is_gap=None,
            )
 
    df_final = pd.DataFrame(columns, index=main_df.index)
    return df_final, contexts


def plot_gapfill_dashboard(
    df: pd.DataFrame,
    periods: List[Tuple[str, str, str]],
    target_flux: str,
    target,
    model_type: str,
    ustar_cut: str,
    contexts: Dict[str, CumulativeContext],
    qc_levels: List[str] = ["QCF", "QCF0"],
    parcels: List[str] = ["A", "B"],
    sigma_scale: float = 1.96,
):
    import matplotlib.pyplot as plt

    def _cumulative_band(col: str, period_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Correlation-aware cumulative uncertainty for `col`, recomputed fresh
        over period_df's own rows -- matching how the plotted flux cumsum
        itself restarts at each period's start. The structural term uses
        sigma_ens_cumulative (member-path accumulation); sigma_obs/sigma_resid
        are legitimately independent, so a plain quadrature sum restarted at
        the period boundary is fine for those.
        """
        ctx = contexts.get(col)
        if ctx is None:
            return None
        idx = period_df.index.intersection(ctx.df_view.index)
        if len(idx) == 0:
            return None

        sub_view = ctx.df_view.loc[idx]
        is_gap = ctx.is_gap.reindex(idx).fillna(False) if ctx.is_gap is not None \
            else pd.Series(True, index=idx)
        sigma_obs = ctx.sigma_obs.reindex(idx)

        sens_cum = sigma_ens_cumulative(ctx.fit, sub_view, is_gap=is_gap)

        yhat_sub = ctx.fit.model_final.predict(sub_view[ctx.fit.feature_cols]).astype(float)
        yhat_sub = _as_1d_float(_maybe_inverse(ctx.fit.inv_y, yhat_sub))
        sigma_resid_sub = pd.Series(ctx.fit.resid_model.sigma(yhat_sub), index=idx)

        sq_obs = (sigma_obs.where(~is_gap, 0.0).fillna(0.0) ** 2).cumsum()
        sq_res = (sigma_resid_sub.where(is_gap, 0.0).fillna(0.0) ** 2).cumsum()
        sigma_err = np.sqrt(sq_obs + sq_res + sens_cum ** 2)
        return sigma_err.reindex(period_df.index)

    def _plot_flux_with_uncertainty(ax, data, col_val, col_unc=None, label=None, cumulative=False):
        if col_val not in data.columns: return
        y = data[col_val]
        if label is None: label = col_val
        y_plot = y.cumsum() if cumulative else y
        line, = ax.plot(data.index, y_plot, label=label, alpha=0.8)

        if cumulative:
            sigma_plot = _cumulative_band(col_val, data)
        else:
            sigma_plot = data[col_unc].fillna(0.0) if (col_unc and col_unc in data.columns) else None

        if sigma_plot is not None:
            ax.fill_between(data.index, y_plot - sigma_scale * sigma_plot, y_plot + sigma_scale * sigma_plot,
                             color=line.get_color(), alpha=0.2, linewidth=0)

    def _tgt(qc): return f'{target_flux}_L3.3_{ustar_cut}_{qc}'
    def _get_sigma(c):
        if c.endswith("_yhat"): return c.replace("_yhat", "_sigmaGF")
        if "_gf" in c: return c + "_total_unc"
        return None

    for start, end, label in periods:
        period_df = df.loc[pd.to_datetime(start):pd.to_datetime(end)]
        if period_df.empty: continue
        
        rows = []
        cols = [f"{target}_parcel{p}_gf{model_type}_yhat" for p in parcels if f"{target}_parcel{p}_gf{model_type}_yhat" in period_df.columns]
        rows.append((cols, "Predicted (Model Only)"))

        for qc in qc_levels:
            _t = _tgt(qc)
            cols = [f"{_t}_parcel{p}" for p in parcels if f"{_t}_parcel{p}" in period_df.columns]
            rows.append((cols, f"Observed [{qc}]"))
            
        for qc in qc_levels:
            _t = _tgt(qc)
            cols = [f"{_t}_parcel{p}_gf{model_type}" for p in parcels if f"{_t}_parcel{p}_gf{model_type}" in period_df.columns]
            rows.append((cols, f"Gap-filled [{qc}]"))

        cols = [c for c in [f"{_tgt(qc)}_footprint_gf{model_type}" for qc in qc_levels] if c in period_df.columns]
        rows.append((cols, f"Gap-filled (Full-Footprint)"))

        nrows = len(rows)
        if nrows == 0: continue
        fig, axes = plt.subplots(nrows, 2, figsize=(15, 3*nrows), sharex='col')
        if nrows == 1: axes = axes.reshape(1, -1)

        for r, (cols, title) in enumerate(rows):
            for col in cols:
                _plot_flux_with_uncertainty(axes[r, 0], period_df, col, _get_sigma(col), cumulative=False)
            axes[r, 0].set_title(title, fontsize=10, fontweight='bold')
            axes[r, 0].legend(fontsize=8, loc='upper right')
            axes[r, 0].grid(True, alpha=0.3)
            
            for col in cols:
                _plot_flux_with_uncertainty(axes[r, 1], period_df, col, _get_sigma(col), cumulative=True)
            axes[r, 1].set_title(f"{title} — Cumulative", fontsize=10, fontweight='bold')
            axes[r, 1].grid(True, alpha=0.3)

        fig.suptitle(f"{label} ({start} → {end})", y=0.995, fontsize=14)
        plt.tight_layout()
        plt.show()
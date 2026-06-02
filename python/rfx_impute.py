"""
RFX Imputation: Random Forest-based Missing Value Imputation
=============================================================

Implements the imputation methods from:

    Joshua Young (2017). "New Imputation Methods for Random Forests"
    Master's Project, Utah State University
    Major Professor: Dr. Adele Cutler
    Department: Mathematics and Statistics

This project introduces two new methods for imputation of missing data in 
random forests, developed under the guidance of Dr. Adele Cutler (co-creator
of the original Random Forest algorithm with Leo Breiman).

Three methods implemented:
1. rfx_impute_rough (New/Rough): Initial median/mode imputation, then RF refinement
2. rfx_impute_rand (New/Rand): Initial random sampling imputation, then RF refinement  
3. rfx_impute_proximity (Prox): Breiman-Cutler proximity-based imputation

Algorithm (New/Rough and New/Rand):
1. Mark which features contain missing values
2. Initially impute all missing values (rough=median/mode or rand=random sampling)
3. For each feature with missing values:
   a. Set it as the response variable
   b. Train RF regressor (numeric) or classifier (categorical) to predict missing
   c. Replace missing values with RF predictions
4. Optionally iterate until convergence

GPU-accelerated via RFX for large datasets.

References:
- Young, J. (2017). New Imputation Methods for Random Forests. Utah State University.
- Breiman, L. & Cutler, A. Random Forests. https://www.stat.berkeley.edu/~breiman/RandomForests/
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import warnings

try:
    from RFXFuse import RandomForestRegressor, RandomForestClassifier
    HAS_RFX = True
except ImportError:
    HAS_RFX = False
    warnings.warn("RFX-Fuse not available. Install rfx-fuse package.")


# ---------------------------------------------------------------------------
# Deprecation helpers
# ---------------------------------------------------------------------------

def _resolve_deprecated(new_val, old_val, new_name, old_name):
    """Return the correct value, emitting a deprecation warning if the old name was used."""
    if old_val is not None:
        warnings.warn(
            f"'{old_name}' is deprecated and will be removed in a future release. "
            f"Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return old_val
    return new_val


def _normalize_categorical_features(categorical_features, n_features):
    """Accept bool array, int list, or None and return a bool array of length n_features."""
    if categorical_features is None:
        return None
    cat = np.asarray(categorical_features)
    if cat.dtype == bool and cat.shape == (n_features,):
        return cat
    if np.issubdtype(cat.dtype, np.integer):
        mask = np.zeros(n_features, dtype=bool)
        mask[cat] = True
        return mask
    if cat.dtype == bool:
        if len(cat) != n_features:
            raise ValueError(
                f"categorical_features bool array length {len(cat)} != n_features {n_features}"
            )
        return cat
    raise TypeError(
        "categorical_features must be a bool array or integer index array, "
        f"got dtype={cat.dtype}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_categorical(X: np.ndarray, max_unique: int = 10) -> np.ndarray:
    """Detect which columns are categorical based on unique value count."""
    n_features = X.shape[1]
    is_categorical = np.zeros(n_features, dtype=bool)
    for j in range(n_features):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            n_unique = len(np.unique(valid))
            is_integers = np.allclose(valid, np.round(valid))
            is_categorical[j] = (n_unique <= max_unique) and is_integers
    return is_categorical


def _rough_impute(X: np.ndarray, is_categorical: np.ndarray) -> np.ndarray:
    """na.roughfix equivalent: Impute with median (numeric) or mode (categorical)."""
    X_imputed = X.copy()
    for j in range(X.shape[1]):
        col = X_imputed[:, j]
        mask = np.isnan(col)
        if mask.sum() == 0:
            continue
        valid = col[~mask]
        if len(valid) == 0:
            X_imputed[mask, j] = 0
        elif is_categorical[j]:
            values, counts = np.unique(valid, return_counts=True)
            X_imputed[mask, j] = values[np.argmax(counts)]
        else:
            X_imputed[mask, j] = np.median(valid)
    return X_imputed


def _random_impute(X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Random imputation: sample from observed values in each column."""
    X_imputed = X.copy()
    for j in range(X.shape[1]):
        col = X_imputed[:, j]
        mask = np.isnan(col)
        if mask.sum() == 0:
            continue
        valid = col[~mask]
        if len(valid) == 0:
            X_imputed[mask, j] = 0
        else:
            X_imputed[mask, j] = rng.choice(valid, size=int(mask.sum()), replace=True)
    return X_imputed


def _auto_encode(X):
    """Auto-detect and encode string/object/category columns to integer codes.

    Accepts pd.DataFrame, numpy object array, or already-numeric array.

    Returns
    -------
    X_numeric : np.ndarray (float32)
        All columns numeric, strings replaced by 0-based integer codes, NaN preserved.
    auto_cat_mask : np.ndarray (bool)
        True for columns that were string-encoded.
    encoders : dict[int, np.ndarray]
        Maps column index -> array of original category labels.
        e.g. {3: array(['blue', 'green', 'red'])} means blue=0, green=1, red=2.
    """
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False

    # --- pd.DataFrame path ---
    if has_pandas and isinstance(X, pd.DataFrame):
        n_cols = X.shape[1]
        encoders = {}
        auto_cat = np.zeros(n_cols, dtype=bool)
        X_out = np.empty(X.shape, dtype=np.float32)

        for j, col_name in enumerate(X.columns):
            series = X.iloc[:, j]
            if series.dtype == 'category' or series.dtype == object:
                cat = pd.Categorical(series)
                codes = cat.codes.astype(np.float32)
                codes[codes == -1] = np.nan  # pandas uses -1 for NaN
                X_out[:, j] = codes
                encoders[j] = np.asarray(cat.categories)
                auto_cat[j] = True
            elif series.dtype == bool:
                X_out[:, j] = series.astype(np.float32).values
                encoders[j] = np.array([False, True])
                auto_cat[j] = True
            else:
                vals = series.values
                if hasattr(vals, 'astype'):
                    X_out[:, j] = vals.astype(np.float32)
                else:
                    X_out[:, j] = np.asarray(vals, dtype=np.float32)
        return X_out, auto_cat, encoders

    # --- numpy object array path ---
    X_arr = np.asarray(X)
    if X_arr.dtype == object:
        n_rows, n_cols = X_arr.shape
        encoders = {}
        auto_cat = np.zeros(n_cols, dtype=bool)
        X_out = np.empty((n_rows, n_cols), dtype=np.float32)

        for j in range(n_cols):
            col = X_arr[:, j]
            # Check if any non-None/non-NaN value is a string
            has_str = False
            for v in col:
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    if isinstance(v, str):
                        has_str = True
                        break

            if has_str:
                # Collect unique non-missing string labels in sorted order
                labels = sorted(set(
                    v for v in col
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                ))
                label_map = {lab: i for i, lab in enumerate(labels)}
                codes = np.empty(n_rows, dtype=np.float32)
                for k in range(n_rows):
                    v = col[k]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        codes[k] = np.nan
                    else:
                        codes[k] = label_map[v]
                X_out[:, j] = codes
                encoders[j] = np.array(labels)
                auto_cat[j] = True
            else:
                # Numeric -- just cast
                for k in range(n_rows):
                    v = col[k]
                    if v is None:
                        X_out[k, j] = np.nan
                    else:
                        X_out[k, j] = float(v)
        return X_out, auto_cat, encoders

    # --- Already numeric ---
    return np.asarray(X, dtype=np.float32), np.zeros(X_arr.shape[1], dtype=bool), {}


def _apply_encoders(X, encoders):
    """Re-encode new data using stored label maps from a prior _auto_encode() call.

    Unknown categories are mapped to NaN.
    """
    try:
        import pandas as pd
        has_pandas = True
    except ImportError:
        has_pandas = False

    if has_pandas and isinstance(X, pd.DataFrame):
        n_cols = X.shape[1]
        X_out = np.empty(X.shape, dtype=np.float32)
        for j, col_name in enumerate(X.columns):
            series = X.iloc[:, j]
            if j in encoders:
                labels = encoders[j]
                label_map = {lab: i for i, lab in enumerate(labels)}
                codes = np.empty(len(series), dtype=np.float32)
                for k, v in enumerate(series):
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        codes[k] = np.nan
                    elif hasattr(v, '__hash__') and v in label_map:
                        codes[k] = label_map[v]
                    else:
                        codes[k] = np.nan  # unknown category
                X_out[:, j] = codes
            else:
                vals = series.values
                if hasattr(vals, 'astype'):
                    X_out[:, j] = vals.astype(np.float32)
                else:
                    X_out[:, j] = np.asarray(vals, dtype=np.float32)
        return X_out

    X_arr = np.asarray(X)
    if X_arr.dtype == object:
        n_rows, n_cols = X_arr.shape
        X_out = np.empty((n_rows, n_cols), dtype=np.float32)
        for j in range(n_cols):
            col = X_arr[:, j]
            if j in encoders:
                labels = encoders[j]
                label_map = {lab: i for i, lab in enumerate(labels)}
                for k in range(n_rows):
                    v = col[k]
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        X_out[k, j] = np.nan
                    elif v in label_map:
                        X_out[k, j] = label_map[v]
                    else:
                        X_out[k, j] = np.nan
            else:
                for k in range(n_rows):
                    v = col[k]
                    X_out[k, j] = np.nan if v is None else float(v)
        return X_out

    return np.asarray(X, dtype=np.float32)


def decode_imputed(X_imputed, encoders):
    """Map integer-coded columns back to original string labels.

    Args:
        X_imputed: float32 ndarray from rfx_impute or Imputer.transform.
        encoders: dict from info['encoders'] or Imputer.encoders_.

    Returns:
        np.ndarray with dtype=object where encoded columns have string values restored.
    """
    if not encoders:
        return X_imputed

    X_out = np.empty(X_imputed.shape, dtype=object)
    for j in range(X_imputed.shape[1]):
        if j in encoders:
            labels = encoders[j]
            col = X_imputed[:, j]
            decoded = np.empty(len(col), dtype=object)
            for k in range(len(col)):
                if np.isnan(col[k]):
                    decoded[k] = None
                else:
                    idx = int(round(col[k]))
                    decoded[k] = labels[idx] if 0 <= idx < len(labels) else None
            X_out[:, j] = decoded
        else:
            X_out[:, j] = X_imputed[:, j]
    return X_out


def _build_rf_kwargs(ntree, mtry, nodesize, iseed, use_gpu, categorical_features_for_rf,
                     extra_kwargs):
    """Build the keyword dict passed to every internal RF constructor."""
    kw = dict(
        ntree=ntree,
        mtry=mtry,
        nodesize=nodesize,
        iseed=iseed,
        use_gpu=use_gpu,
        compute_importance=False,
        compute_proximity=False,
        show_progress=False,
    )
    if categorical_features_for_rf is not None:
        kw['categorical_features'] = categorical_features_for_rf
    kw.update(extra_kwargs)
    return kw


# ---------------------------------------------------------------------------
# Main imputation function
# ---------------------------------------------------------------------------

def rfx_impute(
    X: np.ndarray,
    method: str = 'rough',
    n_iterations: int = 1,
    ntree: int = 100,
    mtry: int = 0,
    nodesize: int = 5,
    iseed: int = 42,
    use_gpu: bool = True,
    show_progress: bool = False,
    categorical_features=None,
    maxcat: int = 10,
    auto_gpu: bool = True,
    # Deprecated names (kept for one release cycle)
    n_trees: int = None,
    seed: int = None,
    verbose: bool = None,
    max_categorical_unique: int = None,
    **rf_kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Impute missing values using Random Forest.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with missing values (np.nan).
    method : str
        'rough' - Initial imputation with median/mode (recommended).
        'rand' - Initial imputation with random sampling.
    n_iterations : int
        Number of refinement iterations (1 = single pass, higher = convergence).
    ntree : int
        Number of trees for each internal RF model.
    mtry : int
        Features per split. 0 = auto (sqrt for classification, p/3 for regression).
    nodesize : int
        Minimum terminal node size for internal RF models.
    iseed : int
        Random seed (passed to all internal RF constructors).
    use_gpu : bool
        Use GPU acceleration (may be overridden by auto_gpu).
    show_progress : bool
        Print progress messages.
    categorical_features : array-like, optional
        Boolean mask or integer index array of categorical columns.
        Auto-detected if None.
    maxcat : int
        Max unique values to consider a feature categorical (auto-detection).
    auto_gpu : bool
        If True, automatically fall back to CPU for small datasets
        (n_samples * n_features < 500,000).
    **rf_kwargs
        Additional keyword arguments passed through to internal RF constructors
        (e.g. use_histogram, n_bins).
        
    Returns
    -------
    X_imputed : np.ndarray
        Imputed data matrix.
    info : dict
        Imputation statistics (n_missing per feature, iterations, etc.).
    """
    if not HAS_RFX:
        raise ImportError("RFX package required for rfx_impute")

    # Handle deprecated parameter names
    ntree = _resolve_deprecated(ntree, n_trees, 'ntree', 'n_trees')
    iseed = _resolve_deprecated(iseed, seed, 'iseed', 'seed')
    show_progress = _resolve_deprecated(show_progress, verbose, 'show_progress', 'verbose')
    maxcat = _resolve_deprecated(maxcat, max_categorical_unique, 'maxcat', 'max_categorical_unique')

    # Auto-encode string/object/category columns before float32 cast
    X, auto_cat_mask, encoders = _auto_encode(X)
    n_samples, n_features = X.shape
    rng = np.random.RandomState(iseed)

    # Auto GPU selection
    if auto_gpu:
        data_size = n_samples * n_features
        gpu_threshold = 500_000
        use_gpu_actual = use_gpu and (data_size > gpu_threshold)
        if show_progress and use_gpu and not use_gpu_actual:
            print(f"  Auto GPU: Using CPU (data size {data_size:,} < {gpu_threshold:,} threshold)")
    else:
        use_gpu_actual = use_gpu

    # Find missing values
    missing_mask = np.isnan(X)
    n_missing_total = int(missing_mask.sum())
    n_missing_per_feature = missing_mask.sum(axis=0)
    features_with_missing = np.where(n_missing_per_feature > 0)[0]

    if show_progress:
        print(f"RFX Imputation ({method})")
        print(f"  Samples: {n_samples:,}, Features: {n_features}")
        print(f"  Total missing: {n_missing_total:,} "
              f"({100 * n_missing_total / (n_samples * n_features):.1f}%)")
        print(f"  Features with missing: {len(features_with_missing)}")
        if encoders:
            print(f"  Auto-encoded {len(encoders)} string column(s): "
                  f"{list(encoders.keys())}")

    if n_missing_total == 0:
        if show_progress:
            print("  No missing values - returning original data")
        return X.copy(), {'n_missing': 0, 'features_imputed': [], 'encoders': encoders}

    # Merge user-supplied + auto-detected categorical masks
    cat_mask = _normalize_categorical_features(categorical_features, n_features)
    if cat_mask is None:
        cat_mask = _detect_categorical(X, maxcat)
    cat_mask = cat_mask | auto_cat_mask

    n_categorical = int(cat_mask.sum())
    if show_progress:
        print(f"  Categorical features: {n_categorical}, Numeric: {n_features - n_categorical}")

    # Step 1: Initial imputation
    if method == 'rough':
        X_imputed = _rough_impute(X, cat_mask)
        if show_progress:
            print("  Initial imputation: median/mode (rough)")
    elif method == 'rand':
        X_imputed = _random_impute(X, rng)
        if show_progress:
            print("  Initial imputation: random sampling")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rough' or 'rand'")

    # Step 2: RF refinement (features sorted by ascending missingness)
    sorted_features = features_with_missing[
        np.argsort(n_missing_per_feature[features_with_missing])
    ]

    for iteration in range(n_iterations):
        if show_progress:
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")
        changes = 0.0

        for j in sorted_features:
            feat_mask = missing_mask[:, j]
            n_miss_j = int(feat_mask.sum())
            if n_miss_j == 0:
                continue

            train_mask = ~feat_mask
            other = [f for f in range(n_features) if f != j]
            X_train = X_imputed[train_mask][:, other]
            y_train = X_imputed[train_mask, j]
            X_test = X_imputed[feat_mask][:, other]

            if len(y_train) < 10:
                if show_progress:
                    print(f"    Feature {j}: Skipped (only {len(y_train)} training samples)")
                continue

            auto_mtry = mtry if mtry > 0 else max(1, int(np.sqrt(len(other))))

            # Build categorical mask for the *other* features
            cat_for_rf = cat_mask[other] if cat_mask is not None else None

            kw = _build_rf_kwargs(
                ntree, auto_mtry, nodesize, iseed, use_gpu_actual, cat_for_rf, rf_kwargs
            )

            try:
                if cat_mask[j]:
                    classes = np.unique(y_train)
                    if len(classes) < 2:
                        y_pred = np.full(n_miss_j, classes[0])
                    else:
                        y_train_int = np.searchsorted(classes, y_train).astype(np.int32)
                        model = RandomForestClassifier(**kw)
                        model.fit(X_train, y_train_int)
                        y_pred_int = model.predict(X_test)
                        y_pred = classes[y_pred_int.astype(int)]
                else:
                    if mtry == 0:
                        kw['mtry'] = max(1, len(other) // 3)
                    model = RandomForestRegressor(**kw)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                old_values = X_imputed[feat_mask, j]
                change = float(np.mean(np.abs(y_pred - old_values)))
                changes += change
                X_imputed[feat_mask, j] = y_pred

                if show_progress:
                    ftype = "cat" if cat_mask[j] else "num"
                    print(f"    Feature {j} ({ftype}): imputed {n_miss_j} values, "
                          f"mean change={change:.4f}")
            except Exception as e:
                if show_progress:
                    print(f"    Feature {j}: Error - {e}")
                continue

        if show_progress:
            print(f"  Total change this iteration: {changes:.4f}")
        if changes < 1e-6 and iteration > 0:
            if show_progress:
                print("  Converged!")
            break

    info = {
        'n_missing_total': n_missing_total,
        'n_missing_per_feature': n_missing_per_feature,
        'features_imputed': list(features_with_missing),
        'n_iterations': iteration + 1 if n_iterations > 0 else 0,
        'method': method,
        'is_categorical': cat_mask,
        'use_gpu': use_gpu_actual,
        'auto_gpu': auto_gpu,
        'encoders': encoders,
    }

    if show_progress:
        print(f"\n  Imputation complete!")
    return X_imputed, info


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def rfx_impute_rough(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
    """Convenience wrapper for rough imputation method."""
    return rfx_impute(X, method='rough', **kwargs)


def rfx_impute_rand(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
    """Convenience wrapper for random imputation method."""
    return rfx_impute(X, method='rand', **kwargs)


# =============================================================================
# Proximity-based imputation (from original Breiman-Cutler)
# =============================================================================

def rfx_impute_proximity(
    X: np.ndarray,
    ntree: int = 100,
    n_iterations: int = 5,
    use_gpu: bool = True,
    categorical_features=None,
    maxcat: int = 10,
    show_progress: bool = False,
    iseed: int = 42,
    top_k: Optional[int] = None,
    aggregation: str = 'mean',
    nodesize: int = 5,
    mtry: int = 0,
    # Deprecated names
    n_trees: int = None,
    seed: int = None,
    verbose: bool = None,
    max_categorical_unique: int = None,
    **rf_kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Proximity-based imputation (Breiman-Cutler method).
    
    Uses RF proximity matrix to weight observed values for imputation.
    Missing values are imputed using similar samples from the proximity matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with missing values (np.nan).
    ntree : int
        Number of trees.
    n_iterations : int
        Number of iterations (RF is retrained each iteration with updated imputations).
    use_gpu : bool
        Use GPU acceleration.
    categorical_features : array-like, optional
        Boolean mask or integer index array of categorical columns.
    maxcat : int
        Max unique values for categorical auto-detection.
    show_progress : bool
        Print progress.
    iseed : int
        Random seed.
    top_k : int, optional
        If specified, only use top-K most similar samples (by proximity).
        If None, use all samples weighted by proximity.
        Recommended: 5-20 for robustness.
    aggregation : str
        How to aggregate values from similar samples:
        - 'mean': Weighted average (classic Breiman-Cutler).
        - 'median': Median of top-K similar (more robust to outliers).
        - 'weighted_mean': Mean weighted by proximity (same as 'mean' for top_k=None).
    nodesize : int
        Minimum terminal node size for the unsupervised RF.
    mtry : int
        Features per split. 0 = auto (sqrt(n_features)).
    **rf_kwargs
        Additional keyword arguments passed through to the unsupervised RF.
        
    Returns
    -------
    X_imputed : np.ndarray
        Imputed data matrix.
    info : dict
        Imputation statistics.
    """
    from RFXFuse import RandomForestUnsupervised

    ntree = _resolve_deprecated(ntree, n_trees, 'ntree', 'n_trees')
    iseed = _resolve_deprecated(iseed, seed, 'iseed', 'seed')
    show_progress = _resolve_deprecated(show_progress, verbose, 'show_progress', 'verbose')
    maxcat = _resolve_deprecated(maxcat, max_categorical_unique, 'maxcat', 'max_categorical_unique')

    X, auto_cat_mask, encoders = _auto_encode(X)
    n_samples, n_features = X.shape

    missing_mask = np.isnan(X)
    n_missing_total = int(missing_mask.sum())

    if show_progress:
        agg_str = f"top-{top_k} {aggregation}" if top_k else f"weighted {aggregation}"
        print(f"RFX Proximity Imputation ({agg_str})")
        print(f"  Samples: {n_samples:,}, Features: {n_features}")
        print(f"  Total missing: {n_missing_total:,}")
        if top_k:
            print(f"  Using top-{top_k} most similar samples")
        if encoders:
            print(f"  Auto-encoded {len(encoders)} string column(s): "
                  f"{list(encoders.keys())}")

    if n_missing_total == 0:
        return X.copy(), {'n_missing': 0, 'encoders': encoders}

    cat_mask = _normalize_categorical_features(categorical_features, n_features)
    if cat_mask is None:
        cat_mask = _detect_categorical(X, maxcat)
    cat_mask = cat_mask | auto_cat_mask

    X_imputed = _rough_impute(X, cat_mask)

    auto_mtry = mtry if mtry > 0 else max(1, int(np.sqrt(n_features)))

    for iteration in range(n_iterations):
        if show_progress:
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")

        kw = dict(
            ntree=ntree,
            mtry=auto_mtry,
            nodesize=nodesize,
            iseed=iseed,
            use_gpu=use_gpu,
            compute_proximity=True,
            compute_importance=False,
            show_progress=False,
        )
        kw.update(rf_kwargs)

        model = RandomForestUnsupervised(**kw)
        model.fit(X_imputed)

        try:
            prox = model.get_proximity_matrix()
            if prox is None:
                raise ValueError("Proximity matrix not available")
            if prox.shape[0] > n_samples:
                prox = prox[:n_samples, :n_samples]
        except Exception as e:
            if show_progress:
                print(f"    Error getting proximity: {e}")
            break

        changes = 0.0
        for j in range(n_features):
            feature_missing = missing_mask[:, j]
            if feature_missing.sum() == 0:
                continue

            for i in np.where(feature_missing)[0]:
                prox_scores = prox[i, :].copy()
                prox_scores[i] = 0
                observed = ~missing_mask[:, j]
                prox_scores[~observed] = 0
                observed_indices = np.where(observed & (prox_scores > 0))[0]

                if len(observed_indices) == 0:
                    continue

                if top_k is not None and len(observed_indices) > top_k:
                    top_k_idx = np.argsort(prox_scores[observed_indices])[-top_k:]
                    selected = observed_indices[top_k_idx]
                else:
                    selected = observed_indices

                vals = X_imputed[selected, j]
                wts = prox_scores[selected]

                if cat_mask[j]:
                    uvals = np.unique(vals)
                    if aggregation == 'median':
                        counts = np.array([(vals == v).sum() for v in uvals])
                    else:
                        counts = np.array([wts[vals == v].sum() for v in uvals])
                    new_val = uvals[np.argmax(counts)]
                else:
                    if aggregation == 'median':
                        new_val = np.median(vals)
                    elif aggregation in ('mean', 'weighted_mean'):
                        wts = wts / wts.sum()
                        new_val = float(np.sum(wts * vals))
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")

                old_val = X_imputed[i, j]
                changes += abs(new_val - old_val)
                X_imputed[i, j] = new_val

        if show_progress:
            print(f"    Total change: {changes:.4f}")
        if changes < 1e-6:
            if show_progress:
                print("  Converged!")
            break

    info = {
        'n_missing_total': n_missing_total,
        'n_iterations': iteration + 1 if n_iterations > 0 else 0,
        'method': 'proximity',
        'top_k': top_k,
        'aggregation': aggregation,
        'encoders': encoders,
    }
    if show_progress:
        print(f"\n  Proximity imputation complete!")
    return X_imputed, info


# ---------------------------------------------------------------------------
# Convenience proximity wrappers
# ---------------------------------------------------------------------------

def rfx_impute_topk_mean(X: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, dict]:
    """Impute using mean of top-K most similar samples (RF proximity-based)."""
    return rfx_impute_proximity(X, top_k=k, aggregation='mean', **kwargs)


def rfx_impute_topk_median(X: np.ndarray, k: int = 10, **kwargs) -> Tuple[np.ndarray, dict]:
    """Impute using median of top-K most similar samples (RF proximity-based)."""
    return rfx_impute_proximity(X, top_k=k, aggregation='median', **kwargs)


rfx_impute_knn_mean = rfx_impute_topk_mean
rfx_impute_knn_median = rfx_impute_topk_median


# =============================================================================
# Class-based imputer with fit/transform API
# =============================================================================

class Imputer:
    """Random Forest imputer with fit/transform API.

    Usage::

        imputer = Imputer(ntree=100, use_gpu=True)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

    Parameters
    ----------
    method : str
        'rough' or 'rand'.
    ntree : int
        Number of trees.
    n_iterations : int
        Refinement iterations.
    mtry : int
        Features per split (0 = auto).
    nodesize : int
        Minimum terminal node size.
    iseed : int
        Random seed.
    use_gpu : bool
        Use GPU acceleration.
    auto_gpu : bool
        Auto-fall-back to CPU for small data.
    categorical_features : array-like, optional
        Boolean mask or integer index array.
    maxcat : int
        Max unique values for categorical auto-detection.
    show_progress : bool
        Print progress messages.
    **rf_kwargs
        Pass-through to internal RF constructors.
    """

    def __init__(
        self,
        method='rough',
        ntree=100,
        n_iterations=1,
        mtry=0,
        nodesize=5,
        iseed=42,
        use_gpu=True,
        auto_gpu=True,
        categorical_features=None,
        maxcat=10,
        show_progress=False,
        # Deprecated names
        n_trees=None,
        seed=None,
        verbose=None,
        max_categorical_unique=None,
        **rf_kwargs,
    ):
        self.method = method
        self.ntree = _resolve_deprecated(ntree, n_trees, 'ntree', 'n_trees')
        self.n_iterations = n_iterations
        self.mtry = mtry
        self.nodesize = nodesize
        self.iseed = _resolve_deprecated(iseed, seed, 'iseed', 'seed')
        self.use_gpu = use_gpu
        self.auto_gpu = auto_gpu
        self.categorical_features = categorical_features
        self.maxcat = _resolve_deprecated(maxcat, max_categorical_unique, 'maxcat', 'max_categorical_unique')
        self.show_progress = _resolve_deprecated(show_progress, verbose, 'show_progress', 'verbose')
        self.rf_kwargs = rf_kwargs

        # Populated during fit()
        self.models_ = {}
        self.classes_ = {}
        self.initial_fill_ = None
        self.is_categorical_ = None
        self.feature_order_ = None
        self.n_features_ = None
        self.encoders_ = {}
        self.is_fitted_ = False

    def fit(self, X):
        """Fit imputer: learn initial fill values and train per-feature RF models.

        Args:
            X: np.ndarray, pd.DataFrame, or object array. String/category columns
               are auto-encoded to integers. Use decode() to reverse.
        """
        if not HAS_RFX:
            raise ImportError("RFX package required for Imputer")

        X, auto_cat_mask, self.encoders_ = _auto_encode(X)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.show_progress and self.encoders_:
            print(f"  Auto-encoded {len(self.encoders_)} string column(s): "
                  f"{list(self.encoders_.keys())}")

        # Detect / normalize categorical, merge with auto-detected
        cat_mask = _normalize_categorical_features(self.categorical_features, n_features)
        if cat_mask is None:
            cat_mask = _detect_categorical(X, self.maxcat)
        cat_mask = cat_mask | auto_cat_mask
        self.is_categorical_ = cat_mask

        # Store initial fill values
        self.initial_fill_ = np.zeros(n_features, dtype=np.float32)
        for j in range(n_features):
            valid = X[~np.isnan(X[:, j]), j]
            if len(valid) == 0:
                self.initial_fill_[j] = 0.0
            elif cat_mask[j]:
                values, counts = np.unique(valid, return_counts=True)
                self.initial_fill_[j] = values[np.argmax(counts)]
            else:
                self.initial_fill_[j] = np.median(valid)

        # Features with missing, sorted ascending by missingness
        missing_mask = np.isnan(X)
        n_miss_per = missing_mask.sum(axis=0)
        features_with = np.where(n_miss_per > 0)[0]
        self.feature_order_ = features_with[np.argsort(n_miss_per[features_with])]

        # Initial imputation
        X_imputed = X.copy()
        for j in range(n_features):
            m = np.isnan(X_imputed[:, j])
            if m.any():
                X_imputed[m, j] = self.initial_fill_[j]

        # GPU auto
        use_gpu_actual = self.use_gpu
        if self.auto_gpu:
            data_size = n_samples * n_features
            gpu_threshold = 500_000
            use_gpu_actual = self.use_gpu and (data_size > gpu_threshold)
            if self.show_progress and self.use_gpu and not use_gpu_actual:
                print(f"  Auto GPU: Using CPU (data size {data_size:,} < {gpu_threshold:,})")

        if self.show_progress:
            print(f"Imputer fit")
            print(f"  Samples: {n_samples:,}, Features: {n_features}")
            print(f"  Features with missing: {len(self.feature_order_)}")
            print(f"  Categorical: {int(cat_mask.sum())}, Numeric: {n_features - int(cat_mask.sum())}")

        self.models_ = {}
        self.classes_ = {}

        for iteration in range(self.n_iterations):
            if self.show_progress:
                print(f"\n  Iteration {iteration + 1}/{self.n_iterations}")

            for j in self.feature_order_:
                feat_mask = missing_mask[:, j]
                n_miss_j = int(feat_mask.sum())
                if n_miss_j == 0:
                    continue

                other = [f for f in range(n_features) if f != j]
                X_train = X_imputed[~feat_mask][:, other]
                y_train = X_imputed[~feat_mask, j]
                X_test = X_imputed[feat_mask][:, other]

                if len(y_train) < 10:
                    continue

                auto_mtry = self.mtry if self.mtry > 0 else max(1, int(np.sqrt(len(other))))
                cat_for_rf = cat_mask[other]

                kw = _build_rf_kwargs(
                    self.ntree, auto_mtry, self.nodesize, self.iseed,
                    use_gpu_actual, cat_for_rf, self.rf_kwargs,
                )

                try:
                    if cat_mask[j]:
                        classes = np.unique(y_train)
                        self.classes_[j] = classes
                        if len(classes) < 2:
                            y_pred = np.full(n_miss_j, classes[0])
                        else:
                            y_int = np.searchsorted(classes, y_train).astype(np.int32)
                            model = RandomForestClassifier(**kw)
                            model.fit(X_train, y_int)
                            self.models_[j] = model
                            y_pred = classes[model.predict(X_test).astype(int)]
                    else:
                        if self.mtry == 0:
                            kw['mtry'] = max(1, len(other) // 3)
                        model = RandomForestRegressor(**kw)
                        model.fit(X_train, y_train)
                        self.models_[j] = model
                        y_pred = model.predict(X_test)

                    X_imputed[feat_mask, j] = y_pred
                    if self.show_progress:
                        ftype = "cat" if cat_mask[j] else "num"
                        print(f"    Feature {j} ({ftype}): trained on {len(y_train)} samples")
                except Exception as e:
                    if self.show_progress:
                        print(f"    Feature {j}: Error - {e}")

        self.is_fitted_ = True
        if self.show_progress:
            print(f"\n  Fit complete! {len(self.models_)} models stored.")
        return self

    def transform(self, X) -> np.ndarray:
        """Impute missing values in new data using stored models.

        Args:
            X: np.ndarray, pd.DataFrame, or object array. String columns
               are encoded using the same label map from fit().
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform()")

        if self.encoders_:
            X = _apply_encoders(X, self.encoders_)
        else:
            X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        X_imputed = X.copy()

        # Initial fill with stored values
        for j in range(self.n_features_):
            m = np.isnan(X_imputed[:, j])
            if m.any():
                X_imputed[m, j] = self.initial_fill_[j]

        # Apply stored models
        missing_mask = np.isnan(X)
        for j in self.feature_order_:
            feat_mask = missing_mask[:, j]
            n_miss_j = int(feat_mask.sum())
            if n_miss_j == 0:
                continue

            other = [f for f in range(self.n_features_) if f != j]
            X_test = X_imputed[feat_mask][:, other]

            if j in self.models_:
                try:
                    if self.is_categorical_[j]:
                        classes = self.classes_[j]
                        y_pred = classes[self.models_[j].predict(X_test).astype(int)]
                    else:
                        y_pred = self.models_[j].predict(X_test)
                    X_imputed[feat_mask, j] = y_pred
                except Exception as e:
                    if self.show_progress:
                        print(f"    Feature {j}: transform error - {e}")
            elif j in self.classes_ and len(self.classes_[j]) < 2:
                X_imputed[feat_mask, j] = self.classes_[j][0]

        if self.show_progress:
            remaining = int(np.isnan(X_imputed).sum())
            print(f"  Transform complete! NaN remaining: {remaining}")
        return X_imputed

    def fit_transform(self, X) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def decode(self, X_imputed):
        """Map integer-coded columns back to original string labels.

        Args:
            X_imputed: float32 ndarray from transform() or fit_transform().

        Returns:
            np.ndarray with dtype=object where encoded columns have strings restored.
            If no string columns were auto-encoded, returns X_imputed unchanged.
        """
        return decode_imputed(X_imputed, self.encoders_)

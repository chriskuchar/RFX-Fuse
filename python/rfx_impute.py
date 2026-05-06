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

# Import RFX-Fuse
try:
    from RFXFuse import RandomForestRegressor, RandomForestClassifier
    HAS_RFX = True
except ImportError:
    HAS_RFX = False
    warnings.warn("RFX-Fuse not available. Install rfx-fuse package.")


def _detect_categorical(X: np.ndarray, max_unique: int = 10) -> np.ndarray:
    """Detect which columns are categorical based on unique value count."""
    n_samples, n_features = X.shape
    is_categorical = np.zeros(n_features, dtype=bool)
    
    for j in range(n_features):
        col = X[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            n_unique = len(np.unique(valid))
            # Categorical if few unique values AND all are integers
            is_integers = np.allclose(valid, np.round(valid))
            is_categorical[j] = (n_unique <= max_unique) and is_integers
    
    return is_categorical


def _rough_impute(X: np.ndarray, is_categorical: np.ndarray) -> np.ndarray:
    """
    na.roughfix equivalent: Impute with median (numeric) or mode (categorical).
    """
    X_imputed = X.copy()
    n_features = X.shape[1]
    
    for j in range(n_features):
        col = X_imputed[:, j]
        missing_mask = np.isnan(col)
        
        if missing_mask.sum() == 0:
            continue
            
        valid = col[~missing_mask]
        
        if len(valid) == 0:
            # All missing - use 0
            X_imputed[missing_mask, j] = 0
        elif is_categorical[j]:
            # Mode for categorical
            values, counts = np.unique(valid, return_counts=True)
            mode = values[np.argmax(counts)]
            X_imputed[missing_mask, j] = mode
        else:
            # Median for numeric
            median = np.median(valid)
            X_imputed[missing_mask, j] = median
    
    return X_imputed


def _random_impute(X: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Random imputation: Sample from observed values in each column.
    """
    np.random.seed(seed)
    X_imputed = X.copy()
    n_features = X.shape[1]
    
    for j in range(n_features):
        col = X_imputed[:, j]
        missing_mask = np.isnan(col)
        
        if missing_mask.sum() == 0:
            continue
            
        valid = col[~missing_mask]
        
        if len(valid) == 0:
            # All missing - use 0
            X_imputed[missing_mask, j] = 0
        else:
            # Sample from observed values
            n_missing = missing_mask.sum()
            samples = np.random.choice(valid, size=n_missing, replace=True)
            X_imputed[missing_mask, j] = samples
    
    return X_imputed


def rfx_impute(
    X: np.ndarray,
    method: str = 'rough',
    n_iterations: int = 1,
    n_trees: int = 50,  # Reduced default - good speed/accuracy balance
    use_gpu: bool = True,
    categorical_features: Optional[np.ndarray] = None,
    max_categorical_unique: int = 10,
    verbose: bool = True,
    seed: int = 42,
    auto_gpu: bool = True,  # Auto-select GPU vs CPU based on data size
) -> Tuple[np.ndarray, dict]:
    """
    Impute missing values using Random Forest.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with missing values (np.nan)
    method : str
        'rough' - Initial imputation with median/mode (recommended)
        'rand' - Initial imputation with random sampling
    n_iterations : int
        Number of refinement iterations (1 = single pass, higher = convergence)
    n_trees : int
        Number of trees for each RF model (default 50)
    use_gpu : bool
        Use GPU acceleration (may be overridden by auto_gpu)
    categorical_features : np.ndarray, optional
        Boolean array indicating categorical features. Auto-detected if None.
    max_categorical_unique : int
        Max unique values to consider a feature categorical (for auto-detection)
    verbose : bool
        Print progress
    seed : int
        Random seed
    auto_gpu : bool
        If True, automatically select GPU vs CPU based on data size.
        GPU is used when n_samples * n_features > 500,000.
        Set to False to force the use_gpu setting.
        
    Returns
    -------
    X_imputed : np.ndarray
        Imputed data matrix
    info : dict
        Imputation statistics (n_missing per feature, iterations, etc.)
    """
    if not HAS_RFX:
        raise ImportError("RFX package required for rfx_impute")
    
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    
    # Auto GPU selection: GPU overhead not worth it for small datasets
    # Each feature with missing values trains a separate RF, so GPU overhead multiplies
    if auto_gpu:
        data_size = n_samples * n_features
        gpu_threshold = 500_000  # ~500K elements before GPU is beneficial
        use_gpu_actual = use_gpu and (data_size > gpu_threshold)
        if verbose and use_gpu and not use_gpu_actual:
            print(f"  Auto GPU: Using CPU (data size {data_size:,} < {gpu_threshold:,} threshold)")
    else:
        use_gpu_actual = use_gpu
    
    # Find missing values
    missing_mask = np.isnan(X)
    n_missing_total = missing_mask.sum()
    n_missing_per_feature = missing_mask.sum(axis=0)
    features_with_missing = np.where(n_missing_per_feature > 0)[0]
    
    if verbose:
        print(f"RFX Imputation ({method})")
        print(f"  Samples: {n_samples:,}, Features: {n_features}")
        print(f"  Total missing: {n_missing_total:,} ({100*n_missing_total/(n_samples*n_features):.1f}%)")
        print(f"  Features with missing: {len(features_with_missing)}")
    
    if n_missing_total == 0:
        if verbose:
            print("  No missing values - returning original data")
        return X.copy(), {'n_missing': 0, 'features_imputed': []}
    
    # Detect categorical features
    if categorical_features is None:
        is_categorical = _detect_categorical(X, max_categorical_unique)
    else:
        is_categorical = np.asarray(categorical_features, dtype=bool)
    
    n_categorical = is_categorical.sum()
    if verbose:
        print(f"  Categorical features: {n_categorical}, Numeric: {n_features - n_categorical}")
    
    # Step 1: Initial imputation
    if method == 'rough':
        X_imputed = _rough_impute(X, is_categorical)
        if verbose:
            print("  Initial imputation: median/mode (rough)")
    elif method == 'rand':
        X_imputed = _random_impute(X, seed)
        if verbose:
            print("  Initial imputation: random sampling")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rough' or 'rand'")
    
    # Step 2: RF refinement for each feature with missing values
    # Sort features by number of missing values (ascending) for better stability
    sorted_features = features_with_missing[np.argsort(n_missing_per_feature[features_with_missing])]
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")
        
        changes = 0
        
        for j in sorted_features:
            feature_missing_mask = missing_mask[:, j]
            n_missing_j = feature_missing_mask.sum()
            
            if n_missing_j == 0:
                continue
            
            # Prepare training data: rows without missing in feature j
            train_mask = ~feature_missing_mask
            
            # Features for prediction: all other features
            other_features = [f for f in range(n_features) if f != j]
            X_train = X_imputed[train_mask][:, other_features]
            y_train = X_imputed[train_mask, j]
            
            # Test data: rows with missing in feature j
            X_test = X_imputed[feature_missing_mask][:, other_features]
            
            # Skip if not enough training data
            if len(y_train) < 10:
                if verbose:
                    print(f"    Feature {j}: Skipped (only {len(y_train)} training samples)")
                continue
            
            # Train RF
            mtry = max(1, int(np.sqrt(len(other_features))))
            
            try:
                if is_categorical[j]:
                    # Classification for categorical
                    # Convert to integer classes
                    classes = np.unique(y_train)

                    if len(classes) < 2:
                        # Only 1 unique class — just fill with that value
                        y_pred = np.full(n_missing_j, classes[0])
                    else:
                        y_train_int = np.searchsorted(classes, y_train).astype(np.int32)

                        model = RandomForestClassifier(
                            ntree=n_trees,
                            mtry=mtry,
                            nodesize=5,
                            use_gpu=use_gpu_actual,
                            compute_importance=False,
                            compute_proximity=False,
                            show_progress=False,
                        )
                        model.fit(X_train, y_train_int)
                        y_pred_int = model.predict(X_test)
                        y_pred = classes[y_pred_int.astype(int)]
                else:
                    # Regression for numeric
                    model = RandomForestRegressor(
                        ntree=n_trees,
                        mtry=mtry,
                        nodesize=5,
                        use_gpu=use_gpu_actual,
                        compute_importance=False,
                        compute_proximity=False,
                        show_progress=False,
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate change
                old_values = X_imputed[feature_missing_mask, j]
                change = np.mean(np.abs(y_pred - old_values))
                changes += change
                
                # Update imputed values
                X_imputed[feature_missing_mask, j] = y_pred
                
                if verbose:
                    feature_type = "cat" if is_categorical[j] else "num"
                    print(f"    Feature {j} ({feature_type}): imputed {n_missing_j} values, "
                          f"mean change={change:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f"    Feature {j}: Error - {e}")
                continue
        
        if verbose:
            print(f"  Total change this iteration: {changes:.4f}")
        
        # Early stopping if converged
        if changes < 1e-6 and iteration > 0:
            if verbose:
                print("  Converged!")
            break
    
    # Compile info
    info = {
        'n_missing_total': n_missing_total,
        'n_missing_per_feature': n_missing_per_feature,
        'features_imputed': list(features_with_missing),
        'n_iterations': iteration + 1,
        'method': method,
        'is_categorical': is_categorical,
        'use_gpu': use_gpu_actual,  # Actual GPU/CPU used (may differ from request if auto_gpu)
        'auto_gpu': auto_gpu,
    }
    
    if verbose:
        print(f"\n  ✓ Imputation complete!")
    
    return X_imputed, info


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
    n_trees: int = 100,
    n_iterations: int = 5,
    use_gpu: bool = True,
    categorical_features: Optional[np.ndarray] = None,
    max_categorical_unique: int = 10,
    verbose: bool = True,
    seed: int = 42,
    top_k: Optional[int] = None,
    aggregation: str = 'mean',
) -> Tuple[np.ndarray, dict]:
    """
    Proximity-based imputation (Breiman-Cutler method).
    
    Uses RF proximity matrix to weight observed values for imputation.
    Missing values are imputed using similar samples from the proximity matrix.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix with missing values (np.nan)
    n_trees : int
        Number of trees
    n_iterations : int
        Number of iterations (RF is retrained each iteration with updated imputations)
    use_gpu : bool
        Use GPU acceleration
    categorical_features : np.ndarray, optional
        Boolean array indicating categorical features
    max_categorical_unique : int
        Max unique values for categorical auto-detection
    verbose : bool
        Print progress
    seed : int
        Random seed
    top_k : int, optional
        If specified, only use top-K most similar samples (by proximity).
        If None, use all samples weighted by proximity.
        Recommended: 5-20 for robustness.
    aggregation : str
        How to aggregate values from similar samples:
        - 'mean': Weighted average (classic Breiman-Cutler)
        - 'median': Median of top-K similar (more robust to outliers)
        - 'weighted_mean': Mean weighted by proximity (same as 'mean' for top_k=None)
        
    Returns
    -------
    X_imputed : np.ndarray
        Imputed data matrix
    info : dict
        Imputation statistics
    """
    from RFXFuse import RandomForestUnsupervised
    
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape
    
    # Find missing values
    missing_mask = np.isnan(X)
    n_missing_total = missing_mask.sum()
    
    if verbose:
        agg_str = f"top-{top_k} {aggregation}" if top_k else f"weighted {aggregation}"
        print(f"RFX Proximity Imputation ({agg_str})")
        print(f"  Samples: {n_samples:,}, Features: {n_features}")
        print(f"  Total missing: {n_missing_total:,}")
        if top_k:
            print(f"  Using top-{top_k} most similar samples")
    
    if n_missing_total == 0:
        return X.copy(), {'n_missing': 0}
    
    # Detect categorical features
    if categorical_features is None:
        is_categorical = _detect_categorical(X, max_categorical_unique)
    else:
        is_categorical = np.asarray(categorical_features, dtype=bool)
    
    # Initial rough imputation
    X_imputed = _rough_impute(X, is_categorical)
    
    for iteration in range(n_iterations):
        if verbose:
            print(f"\n  Iteration {iteration + 1}/{n_iterations}")
        
        # Train unsupervised RF to get proximity matrix
        model = RandomForestUnsupervised(
            ntree=n_trees,
            mtry=max(1, int(np.sqrt(n_features))),
            nodesize=5,
            use_gpu=use_gpu,
            compute_proximity=True,
            compute_importance=False,
            show_progress=False,
        )
        
        model.fit(X_imputed)
        
        # Get proximity matrix
        try:
            prox = model.proximity_matrix()
            if prox is None:
                raise ValueError("Proximity matrix not available")
            
            # Handle unsupervised doubling (real + synthetic)
            if prox.shape[0] > n_samples:
                prox = prox[:n_samples, :n_samples]
            
        except Exception as e:
            if verbose:
                print(f"    Error getting proximity: {e}")
            break
        
        # Impute using proximity-based values
        changes = 0
        for j in range(n_features):
            feature_missing = missing_mask[:, j]
            if feature_missing.sum() == 0:
                continue
            
            # For each missing value, use similar samples
            for i in np.where(feature_missing)[0]:
                # Get proximity scores (exclude self)
                prox_scores = prox[i, :].copy()
                prox_scores[i] = 0
                
                # Only use samples with observed values for this feature
                observed = ~missing_mask[:, j]
                prox_scores[~observed] = 0
                
                # Get indices of samples with observed values
                observed_indices = np.where(observed & (prox_scores > 0))[0]
                
                if len(observed_indices) == 0:
                    continue
                
                # Apply top_k filtering if specified
                if top_k is not None and len(observed_indices) > top_k:
                    # Get top-K most similar samples
                    top_k_indices = np.argsort(prox_scores[observed_indices])[-top_k:]
                    selected_indices = observed_indices[top_k_indices]
                else:
                    selected_indices = observed_indices
                
                selected_values = X_imputed[selected_indices, j]
                selected_weights = prox_scores[selected_indices]
                
                if is_categorical[j]:
                    # Mode for categorical (use top-K voting)
                    unique_vals = np.unique(selected_values)
                    if aggregation == 'median':
                        # For categorical median = mode
                        counts = np.array([(selected_values == v).sum() for v in unique_vals])
                    else:
                        # Weighted mode
                        counts = np.array([selected_weights[selected_values == v].sum() for v in unique_vals])
                    new_val = unique_vals[np.argmax(counts)]
                else:
                    # Numeric aggregation
                    if aggregation == 'median':
                        # Simple median of top-K similar values
                        new_val = np.median(selected_values)
                    elif aggregation == 'mean' or aggregation == 'weighted_mean':
                        # Weighted mean
                        selected_weights = selected_weights / selected_weights.sum()
                        new_val = np.sum(selected_weights * selected_values)
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")
                
                old_val = X_imputed[i, j]
                changes += abs(new_val - old_val)
                X_imputed[i, j] = new_val
        
        if verbose:
            print(f"    Total change: {changes:.4f}")
        
        if changes < 1e-6:
            if verbose:
                print("  Converged!")
            break
    
    info = {
        'n_missing_total': n_missing_total,
        'n_iterations': iteration + 1,
        'method': 'proximity',
        'top_k': top_k,
        'aggregation': aggregation,
    }
    
    if verbose:
        print(f"\n  ✓ Proximity imputation complete!")
    
    return X_imputed, info


def rfx_impute_topk_mean(
    X: np.ndarray,
    k: int = 10,
    n_trees: int = 100,
    n_iterations: int = 3,
    use_gpu: bool = True,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Impute using mean of top-K most similar samples (RF proximity-based).
    
    This uses Random Forest proximity to define similarity, then takes
    the mean of the K most similar samples with observed values.
    
    Parameters
    ----------
    X : np.ndarray
        Data with missing values (np.nan)
    k : int
        Number of most similar samples to use (default: 10)
    n_trees : int
        Number of trees for RF proximity (default: 100)
    n_iterations : int
        Number of refinement iterations (default: 3)
    use_gpu : bool
        Use GPU acceleration
    verbose : bool
        Print progress
        
    Returns
    -------
    X_imputed, info
    """
    return rfx_impute_proximity(
        X, n_trees=n_trees, n_iterations=n_iterations,
        use_gpu=use_gpu, verbose=verbose,
        top_k=k, aggregation='mean', **kwargs
    )


def rfx_impute_topk_median(
    X: np.ndarray,
    k: int = 10,
    n_trees: int = 100,
    n_iterations: int = 3,
    use_gpu: bool = True,
    verbose: bool = True,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Impute using median of top-K most similar samples (RF proximity-based).
    
    This uses Random Forest proximity to define similarity, then takes
    the median of the K most similar samples with observed values.
    More robust to outliers than mean.
    
    Parameters
    ----------
    X : np.ndarray
        Data with missing values (np.nan)
    k : int
        Number of most similar samples to use (default: 10)
    n_trees : int
        Number of trees for RF proximity (default: 100)
    n_iterations : int
        Number of refinement iterations (default: 3)
    use_gpu : bool
        Use GPU acceleration
    verbose : bool
        Print progress
        
    Returns
    -------
    X_imputed, info
    """
    return rfx_impute_proximity(
        X, n_trees=n_trees, n_iterations=n_iterations,
        use_gpu=use_gpu, verbose=verbose,
        top_k=k, aggregation='median', **kwargs
    )


# Aliases for backward compatibility
rfx_impute_knn_mean = rfx_impute_topk_mean
rfx_impute_knn_median = rfx_impute_topk_median


# =============================================================================
# Class-based imputer with fit/transform API
# =============================================================================

class Imputer:
    """Random Forest imputer with fit/transform API.

    Usage:
        imputer = Imputer(n_trees=50, use_gpu=True)
        X_train_imp = imputer.fit_transform(X_train)   # fit + impute training data
        X_test_imp = imputer.transform(X_test)          # impute new data using stored models
    """

    def __init__(self, method='rough', n_trees=50, n_iterations=1,
                 use_gpu=True, auto_gpu=True, categorical_features=None,
                 max_categorical_unique=10, verbose=True, seed=42):
        self.method = method
        self.n_trees = n_trees
        self.n_iterations = n_iterations
        self.use_gpu = use_gpu
        self.auto_gpu = auto_gpu
        self.categorical_features = categorical_features
        self.max_categorical_unique = max_categorical_unique
        self.verbose = verbose
        self.seed = seed

        # Populated during fit()
        self.models_ = {}           # feature index → trained RF model
        self.classes_ = {}          # feature index → unique class array (categorical only)
        self.initial_fill_ = None   # per-feature fill values (median/mode)
        self.is_categorical_ = None
        self.feature_order_ = None  # order to apply models
        self.n_features_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray):
        """Fit imputer: learn initial fill values and train per-feature RF models."""
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        # Detect categorical
        if self.categorical_features is not None:
            self.is_categorical_ = np.asarray(self.categorical_features, dtype=bool)
        else:
            self.is_categorical_ = _detect_categorical(X, self.max_categorical_unique)

        # Compute and store initial fill values
        self.initial_fill_ = np.zeros(n_features, dtype=np.float32)
        for j in range(n_features):
            valid = X[~np.isnan(X[:, j]), j]
            if len(valid) == 0:
                self.initial_fill_[j] = 0.0
            elif self.is_categorical_[j]:
                values, counts = np.unique(valid, return_counts=True)
                self.initial_fill_[j] = values[np.argmax(counts)]
            else:
                self.initial_fill_[j] = np.median(valid)

        # Find features with missing
        missing_mask = np.isnan(X)
        n_missing_per_feature = missing_mask.sum(axis=0)
        features_with_missing = np.where(n_missing_per_feature > 0)[0]
        self.feature_order_ = features_with_missing[np.argsort(n_missing_per_feature[features_with_missing])]

        # Initial imputation
        X_imputed = X.copy()
        for j in range(n_features):
            mask = np.isnan(X_imputed[:, j])
            if mask.any():
                X_imputed[mask, j] = self.initial_fill_[j]

        # GPU auto-select
        use_gpu_actual = self.use_gpu
        if self.auto_gpu:
            data_size = n_samples * n_features
            gpu_threshold = 500_000
            use_gpu_actual = self.use_gpu and (data_size > gpu_threshold)
            if self.verbose and self.use_gpu and not use_gpu_actual:
                print(f"  Auto GPU: Using CPU (data size {data_size:,} < {gpu_threshold:,} threshold)")

        if self.verbose:
            n_cat = self.is_categorical_.sum()
            print(f"Imputer fit")
            print(f"  Samples: {n_samples:,}, Features: {n_features}")
            print(f"  Features with missing: {len(self.feature_order_)}")
            print(f"  Categorical: {n_cat}, Numeric: {n_features - n_cat}")

        # Train per-feature RF models
        self.models_ = {}
        self.classes_ = {}

        for iteration in range(self.n_iterations):
            if self.verbose:
                print(f"\n  Iteration {iteration + 1}/{self.n_iterations}")

            for j in self.feature_order_:
                feat_missing = missing_mask[:, j]
                n_missing_j = feat_missing.sum()
                if n_missing_j == 0:
                    continue

                train_mask = ~feat_missing
                other_features = [f for f in range(n_features) if f != j]
                X_train = X_imputed[train_mask][:, other_features]
                y_train = X_imputed[train_mask, j]
                X_test = X_imputed[feat_missing][:, other_features]

                if len(y_train) < 10:
                    continue

                mtry = max(1, int(np.sqrt(len(other_features))))

                try:
                    if self.is_categorical_[j]:
                        classes = np.unique(y_train)
                        self.classes_[j] = classes

                        if len(classes) < 2:
                            y_pred = np.full(n_missing_j, classes[0])
                        else:
                            y_train_int = np.searchsorted(classes, y_train).astype(np.int32)
                            model = RandomForestClassifier(
                                ntree=self.n_trees, mtry=mtry, nodesize=5,
                                use_gpu=use_gpu_actual, compute_importance=False,
                                compute_proximity=False, show_progress=False,
                            )
                            model.fit(X_train, y_train_int)
                            self.models_[j] = model
                            y_pred_int = model.predict(X_test)
                            y_pred = classes[y_pred_int.astype(int)]
                    else:
                        model = RandomForestRegressor(
                            ntree=self.n_trees, mtry=mtry, nodesize=5,
                            use_gpu=use_gpu_actual, compute_importance=False,
                            compute_proximity=False, show_progress=False,
                        )
                        model.fit(X_train, y_train)
                        self.models_[j] = model
                        y_pred = model.predict(X_test)

                    X_imputed[feat_missing, j] = y_pred

                    if self.verbose:
                        ftype = "cat" if self.is_categorical_[j] else "num"
                        print(f"    Feature {j} ({ftype}): trained on {len(y_train)} samples")

                except Exception as e:
                    if self.verbose:
                        print(f"    Feature {j}: Error - {e}")

        self.is_fitted_ = True
        if self.verbose:
            print(f"\n  ✓ Fit complete! {len(self.models_)} models stored.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values in new data using stored models."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before transform()")

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")

        X_imputed = X.copy()

        # Initial fill with stored values
        for j in range(self.n_features_):
            mask = np.isnan(X_imputed[:, j])
            if mask.any():
                X_imputed[mask, j] = self.initial_fill_[j]

        # Apply stored models in same feature order
        missing_mask = np.isnan(X)
        for j in self.feature_order_:
            feat_missing = missing_mask[:, j]
            n_missing_j = feat_missing.sum()
            if n_missing_j == 0:
                continue

            other_features = [f for f in range(self.n_features_) if f != j]
            X_test = X_imputed[feat_missing][:, other_features]

            if j in self.models_:
                try:
                    if self.is_categorical_[j]:
                        classes = self.classes_[j]
                        y_pred_int = self.models_[j].predict(X_test)
                        y_pred = classes[y_pred_int.astype(int)]
                    else:
                        y_pred = self.models_[j].predict(X_test)

                    X_imputed[feat_missing, j] = y_pred
                except Exception as e:
                    if self.verbose:
                        print(f"    Feature {j}: transform error - {e}")
            elif j in self.classes_ and len(self.classes_[j]) < 2:
                # Single-class feature — fill with that value
                X_imputed[feat_missing, j] = self.classes_[j][0]

        if self.verbose:
            remaining_nan = np.isnan(X_imputed).sum()
            print(f"  ✓ Transform complete! NaN remaining: {remaining_nan}")

        return X_imputed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

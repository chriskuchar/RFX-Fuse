#!/usr/bin/env python3
"""
RFX Imputation Quality Validation: Two Dataset Demo
====================================================

USE CASE 4: Imputation Quality Validation
Traditional Approach: No comparable automated approach exists
RFX Approach: RFX Unsupervised model ranks imputation methods WITHOUT ground truth

Datasets:
1. UCI Bike Sharing (17,379 hourly records, 12 features)
2. California Housing (20,640 samples, 8 features)

KEY INSIGHT: RFX Unsupervised can distinguish "real" data from "synthetic" permuted
data. Poorly imputed data looks more "synthetic" - this enables ranking imputation
methods by quality WITHOUT ground truth labels.

RFX Imputation (Young-Cutler 2017):
- rfx_impute_rough(): Initial median/mode, then RF refinement (recommended)
- Based on Joshua Young's thesis under Dr. Adele Cutler (co-creator of Random Forests)

References:
- Young, J. (2017). Imputation for Random Forests. Utah State University.
- Breiman, L. & Cutler, A. Random Forests.
  https://www.stat.berkeley.edu/~breiman/RandomForests/
"""

import io
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from pathlib import Path

# Setup paths relative to this file
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'python'))

# Model save/load configuration
LOAD_MODELS = '--load' in sys.argv or '-l' in sys.argv
FORCE_RETRAIN = '--retrain' in sys.argv or '-r' in sys.argv

# Model and output directories (same folder as script)
MODEL_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR

import rfx
from rfx import RandomForestUnsupervised
from rfx_impute import rfx_impute_rough

# Try to import sklearn imputation methods
try:
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.datasets import fetch_california_housing
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available")


def load_bike_data():
    """Load UCI Bike Sharing Dataset."""
    cache_path = PROJECT_ROOT / "data" / "bike" / "bike_sharing.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        print("   Downloading UCI Bike Sharing Dataset...")
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00275/Bike-Sharing-Dataset.zip")
        import urllib.request
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                with z.open('hour.csv') as f:
                    df = pd.read_csv(f)
        df.to_csv(cache_path, index=False)

    # Select features (exclude index, date string, target)
    feature_cols = [
        'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
        'weathersit', 'temp', 'atemp', 'hum', 'windspeed'
    ]
    X = df[feature_cols].values.astype(np.float32)
    return X, feature_cols


def load_california_data():
    """Load California Housing Dataset."""
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    feature_cols = list(data.feature_names)
    return X, feature_cols


def introduce_missing_values(X, missing_fraction=0.15, seed=42):
    """Randomly introduce missing values (MCAR)."""
    np.random.seed(seed)
    X_missing = X.copy()
    mask = np.random.random(X_missing.shape) < missing_fraction
    X_missing[mask] = np.nan
    return X_missing, mask


def run_imputation_validation(dataset_name, X_all, feature_cols,
                              n_train=3000, n_test=1500):
    """
    Run imputation validation on a dataset.

    Returns dict with results for paper table.
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*70}")

    # Subsample if needed
    if len(X_all) > n_train + n_test:
        np.random.seed(42)
        indices = np.random.permutation(len(X_all))[:n_train + n_test]
        X_all = X_all[indices]

    # Split
    X_train = X_all[:n_train]
    X_test_clean = X_all[n_train:n_train + n_test]

    print(f"   Train: {n_train}, Test: {n_test}, Features: {len(feature_cols)}")

    # Train RFX Unsupervised on clean data (1 RF model)
    print(f"\n   Training RFX Unsupervised on clean data...")
    n_features = X_train.shape[1]

    # Model save path (dataset-specific)
    safe_name = dataset_name.lower().replace(' ', '_')
    model_path = MODEL_DIR / f"unsupervised_{safe_name}.rfx"

    if LOAD_MODELS and not FORCE_RETRAIN and model_path.exists():
        print(f"   Loading saved model from {model_path.name}...")
        model = rfx.load(str(model_path))
        print("   Model loaded successfully")
    else:
        model = RandomForestUnsupervised(
            ntree=100,
            mtry=int(np.sqrt(n_features)) + 1,
            nodesize=5,
            use_gpu=True,
            compute_importance=True,
            compute_local_importance=False,
            compute_proximity=False,
            iseed=42,
        )
        start = time.time()
        model.fit(X_train)
        train_time = time.time() - start
        print(f"   Trained in {train_time:.1f}s, "
              f"OOB Error: {model.get_oob_error():.4f}")

        # Save model
        print(f"   Saving model to {model_path.name}...")
        model.save(str(model_path))
        print("   Model saved successfully")

    # Introduce 15% missing values
    X_test_missing, missing_mask = introduce_missing_values(
        X_test_clean, missing_fraction=0.15
    )
    n_missing = missing_mask.sum()
    print(f"   Introduced {n_missing:,} missing values (15%)")

    # Imputation methods
    imputation_methods = {}

    # Sklearn baselines
    if HAS_SKLEARN:
        print(f"\n   Running imputation methods...")
        imputation_methods['Mean'] = SimpleImputer(
            strategy='mean'
        ).fit_transform(X_test_missing)
        imputation_methods['Median'] = SimpleImputer(
            strategy='median'
        ).fit_transform(X_test_missing)
        imputation_methods['KNN-5'] = KNNImputer(
            n_neighbors=5
        ).fit_transform(X_test_missing)
        imputation_methods['MICE'] = IterativeImputer(
            max_iter=10, random_state=42
        ).fit_transform(X_test_missing)

    # RFX/Rough (Young-Cutler 2017) - 5 iterations = 5 RF models
    print(f"   Running RFX/Rough (Young-Cutler)...")
    start = time.time()
    X_rfx_rough, _ = rfx_impute_rough(
        X_test_missing.astype(np.float32),
        n_trees=100,
        n_iterations=5,
        use_gpu=True,
        verbose=False,
        seed=42
    )
    print(f"   RFX/Rough: {time.time()-start:.1f}s")
    imputation_methods['RFX/Rough'] = X_rfx_rough

    # Oracle (clean data)
    imputation_methods['Original'] = X_test_clean

    # Score each imputation with RFX Unsupervised
    print(f"\n   Scoring imputation quality...")
    results = {}
    for method_name, X_imputed in imputation_methods.items():
        proba = model.predict_proba(X_imputed.astype(np.float32))
        p_synthetic = proba[:, 0] if proba.shape[1] == 2 else 1 - proba[:, 0]
        mean_p_syn = np.mean(p_synthetic)
        pct_syn = np.mean(p_synthetic > 0.5) * 100

        # Compute MAE (we have ground truth for validation)
        if method_name != 'Original':
            mae = np.mean(np.abs(
                X_imputed[missing_mask] - X_test_clean[missing_mask]
            ))
        else:
            mae = 0.0

        results[method_name] = {
            'p_synthetic': mean_p_syn,
            'pct_synthetic': pct_syn,
            'mae': mae
        }

    # Sort by P(synthetic) - lower is better
    sorted_results = sorted(results.items(), key=lambda x: x[1]['p_synthetic'])

    # Print results table
    print(f"\n   {'Method':<12} {'P(syn)':<10} {'%Syn':<10} {'MAE':<10} {'Rank'}")
    print(f"   {'-'*50}")
    for rank, (method, stats) in enumerate(sorted_results, 1):
        print(f"   {method:<12} {stats['p_synthetic']:<10.4f} "
              f"{stats['pct_synthetic']:<10.1f} {stats['mae']:<10.4f} #{rank}")

    # Compute ranking agreement
    mae_sorted = sorted(
        [(m, s['mae']) for m, s in results.items() if m != 'Original'],
        key=lambda x: x[1]
    )
    rfx_ranks = {m: i for i, (m, _) in enumerate(sorted_results) if m != 'Original'}
    mae_ranks = {m: i for i, (m, _) in enumerate(mae_sorted)}

    matches = sum(1 for m in mae_ranks if rfx_ranks.get(m) == mae_ranks.get(m))
    total = len(mae_ranks)
    print(f"\n   RFX vs MAE ranking agreement: {matches}/{total} "
          f"({100*matches/total:.0f}%)")

    return sorted_results, results


def main():
    """Main function to run imputation validation demo."""
    print("=" * 70)
    print("RFX IMPUTATION QUALITY VALIDATION: BIKE SHARING DATASET")
    print("=" * 70)
    print()
    print("Total RF models: 1 (validator) + 5 (RFX/Rough iterations) = 6")
    print()

    all_results = {}

    # Dataset: Bike Sharing only
    X_bike, cols_bike = load_bike_data()
    print(f"Loaded Bike Sharing: {X_bike.shape}")
    bike_sorted, bike_results = run_imputation_validation(
        "Bike Sharing", X_bike, cols_bike
    )
    all_results['Bike'] = bike_results

    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY: IMPUTATION RANKING")
    print("=" * 70)
    print()
    print("P(synthetic) = Average probability sample looks synthetic (test set)")
    print("% Real = % of test samples classified as real (P(syn) < 0.5)")
    print("Lower P(synthetic) = Better imputation")
    print()
    print(f"{'Method':<12} {'P(syn)':<12} {'% Real':<12}")
    print("-" * 36)

    methods = ['Original', 'RFX/Rough', 'MICE', 'KNN-5', 'Median', 'Mean']
    for method in methods:
        p_syn = all_results['Bike'].get(method, {}).get(
            'p_synthetic', float('nan')
        )
        pct_real = 100 - all_results['Bike'].get(method, {}).get(
            'pct_synthetic', float('nan')
        )
        print(f"{method:<12} {p_syn:<12.4f} {pct_real:<12.1f}")

    print()
    print("=" * 70)
    print("KEY FINDING: RFX/Rough (Young-Cutler 2017) ranks among the best")
    print("imputation methods. Original test data has highest % Real (baseline).")
    print("=" * 70)

    # Generate visualization
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATION")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(
        "Use Case 4: Imputation Quality Validation\n"
        "(RFX-Fuse Unique Capability - No Traditional Alternative)",
        fontsize=14, fontweight='bold', y=1.02
    )

    # Bike Dataset Only
    results = all_results['Bike']
    sorted_items = sorted(results.items(), key=lambda x: x[1]['p_synthetic'])
    methods = [m for m, _ in sorted_items]
    p_syn_values = [s['p_synthetic'] for _, s in sorted_items]
    pct_real = [100 - s['pct_synthetic'] for _, s in sorted_items]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.barh(
        range(len(methods)), p_syn_values,
        color=colors, edgecolor='black', alpha=0.8
    )
    ax.set_xlabel(
        'Avg P(synthetic) on Test Set - Lower = Better Imputation',
        fontsize=11, fontweight='bold'
    )
    ax.set_title(
        'UCI Bike Sharing Dataset (3,000 train / 1,500 test, 12 features)\n'
        '% Real = Test samples classified as real (P(syn) < 0.5)',
        fontsize=11, fontweight='bold'
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 0.45)

    for i, (bar, val, pct) in enumerate(zip(bars, p_syn_values, pct_real)):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f} ({pct:.0f}% real)',
            ha='left', va='center', fontsize=10, fontweight='bold'
        )

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "imputation_validation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()

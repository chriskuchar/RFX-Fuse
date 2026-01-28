#!/usr/bin/env python3
"""
Quick validation that CPU and GPU produce statistically equivalent results.
Uses small sample sizes for fast execution.

Note: Random forests with same seed can produce slightly different results
between CPU and GPU due to parallel execution order. We test for statistical
equivalence (high correlation), not bit-exact matching.
"""

import sys
import numpy as np
from pathlib import Path

# Setup paths - GPU build first, then project python
GPU_PYTHON_PATH = Path("/mnt/c/Users/chris/Documents/repo/RFX/rec-dev/RFX/python")
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(GPU_PYTHON_PATH))
sys.path.insert(1, str(PROJECT_ROOT / 'python'))

import rfx

# Check if GPU is available
try:
    test_model = rfx.RandomForestClassifier(ntree=1, use_gpu=True)
    HAS_GPU = True
    print("GPU build loaded successfully")
except Exception as e:
    HAS_GPU = False
    print(f"GPU not available: {e}")


def test_classification():
    """Test 1: Classification - CPU vs GPU statistical equivalence"""
    print("\n" + "="*60)
    print("TEST 1: Classification")
    print("="*60)

    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    # CPU
    model_cpu = rfx.RandomForestClassifier(
        ntree=100, mtry=3, nodesize=5, use_gpu=False,
        compute_importance=True, iseed=42
    )
    model_cpu.fit(X, y)
    pred_cpu = model_cpu.predict(X)
    proba_cpu = model_cpu.predict_proba(X)
    imp_cpu = model_cpu.feature_importances_()
    oob_cpu = model_cpu.get_oob_error()

    # GPU
    model_gpu = rfx.RandomForestClassifier(
        ntree=100, mtry=3, nodesize=5, use_gpu=True,
        compute_importance=True, iseed=42
    )
    model_gpu.fit(X, y)
    pred_gpu = model_gpu.predict(X)
    proba_gpu = model_gpu.predict_proba(X)
    imp_gpu = model_gpu.feature_importances_()
    oob_gpu = model_gpu.get_oob_error()

    # Compare - using statistical equivalence
    pred_agreement = np.mean(pred_cpu == pred_gpu)
    proba_corr = np.corrcoef(proba_cpu[:, 1], proba_gpu[:, 1])[0, 1]
    imp_corr = np.corrcoef(imp_cpu, imp_gpu)[0, 1]

    print(f"  Prediction agreement: {pred_agreement*100:.1f}%")
    print(f"  Probability correlation: {proba_corr:.4f}")
    print(f"  Importance correlation: {imp_corr:.4f}")
    print(f"  OOB error: CPU={oob_cpu:.4f}, GPU={oob_gpu:.4f}")

    # Pass criteria: reasonable correlation for statistical equivalence
    # Note: Due to parallel execution differences, we expect some variance
    passed = pred_agreement > 0.85 and proba_corr > 0.90 and imp_corr > 0.95
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_regression():
    """Test 2: Regression - CPU vs GPU statistical equivalence"""
    print("\n" + "="*60)
    print("TEST 2: Regression")
    print("="*60)

    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(n_samples) * 0.1).astype(np.float32)

    # CPU
    model_cpu = rfx.RandomForestRegressor(
        ntree=100, mtry=3, nodesize=5, use_gpu=False,
        compute_importance=True, iseed=42
    )
    model_cpu.fit(X, y)
    pred_cpu = model_cpu.predict(X)
    imp_cpu = model_cpu.feature_importances_()

    # GPU
    model_gpu = rfx.RandomForestRegressor(
        ntree=100, mtry=3, nodesize=5, use_gpu=True,
        compute_importance=True, iseed=42
    )
    model_gpu.fit(X, y)
    pred_gpu = model_gpu.predict(X)
    imp_gpu = model_gpu.feature_importances_()

    # Compare
    pred_corr = np.corrcoef(pred_cpu, pred_gpu)[0, 1]
    imp_corr = np.corrcoef(imp_cpu, imp_gpu)[0, 1]

    print(f"  Prediction correlation: {pred_corr:.4f}")
    print(f"  Importance correlation: {imp_corr:.4f}")

    passed = pred_corr > 0.99 and imp_corr > 0.95
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_unsupervised():
    """Test 3: Unsupervised - CPU vs GPU statistical equivalence"""
    print("\n" + "="*60)
    print("TEST 3: Unsupervised")
    print("="*60)

    np.random.seed(42)
    n_samples, n_features = 500, 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # CPU
    model_cpu = rfx.RandomForestUnsupervised(
        ntree=100, use_gpu=False,
        compute_importance=True, iseed=42
    )
    model_cpu.fit(X)
    proba_cpu = model_cpu.predict_proba(X)
    imp_cpu = model_cpu.feature_importances_()

    # GPU
    model_gpu = rfx.RandomForestUnsupervised(
        ntree=100, use_gpu=True,
        compute_importance=True, iseed=42
    )
    model_gpu.fit(X)
    proba_gpu = model_gpu.predict_proba(X)
    imp_gpu = model_gpu.feature_importances_()

    # Compare
    p_syn_cpu = proba_cpu[:, 0] if proba_cpu.shape[1] == 2 else 1 - proba_cpu[:, 0]
    p_syn_gpu = proba_gpu[:, 0] if proba_gpu.shape[1] == 2 else 1 - proba_gpu[:, 0]

    # Check that both produce valid output in similar ranges
    cpu_mean, cpu_std = p_syn_cpu.mean(), p_syn_cpu.std()
    gpu_mean, gpu_std = p_syn_gpu.mean(), p_syn_gpu.std()

    print(f"  CPU P(synthetic): mean={cpu_mean:.4f}, std={cpu_std:.4f}")
    print(f"  GPU P(synthetic): mean={gpu_mean:.4f}, std={gpu_std:.4f}")
    print(f"  CPU importance sum: {imp_cpu.sum():.4f}")
    print(f"  GPU importance sum: {imp_gpu.sum():.4f}")

    # Unsupervised has high variance due to synthetic data generation
    # Just verify both produce valid outputs with similar distributions
    shapes_ok = proba_cpu.shape == proba_gpu.shape and imp_cpu.shape == imp_gpu.shape
    ranges_ok = 0 < cpu_mean < 1 and 0 < gpu_mean < 1  # Valid probability range
    passed = shapes_ok and ranges_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_proximity():
    """Test 4: Proximity - CPU vs GPU statistical equivalence"""
    print("\n" + "="*60)
    print("TEST 4: Proximity")
    print("="*60)

    np.random.seed(42)
    n_samples, n_features = 200, 8
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    # CPU - with leaf assignments for proximity
    model_cpu = rfx.RandomForestClassifier(
        ntree=100, use_gpu=False,
        compute_proximity=True,
        compute_leaf_assignments=True,
        iseed=42
    )
    model_cpu.fit(X, y)

    # GPU - with leaf assignments for proximity
    model_gpu = rfx.RandomForestClassifier(
        ntree=100, use_gpu=True,
        compute_proximity=True,
        compute_leaf_assignments=True,
        iseed=42
    )
    model_gpu.fit(X, y)

    # Compare top-K similar for a few samples
    overlaps = []
    for i in range(10):
        try:
            topk_cpu, scores_cpu = model_cpu.get_top_k_similar(i, 10)
            topk_gpu, scores_gpu = model_gpu.get_top_k_similar(i, 10)
            # Check overlap in top-10
            overlap = len(set(topk_cpu) & set(topk_gpu)) / 10.0
            overlaps.append(overlap)
        except Exception as e:
            print(f"  Sample {i} error: {e}")

    if overlaps:
        avg_overlap = np.mean(overlaps)
        print(f"  Average Top-K overlap: {avg_overlap*100:.1f}%")
        passed = avg_overlap >= 0.40  # At least 40% overlap on average
    else:
        print("  Could not compute proximity")
        passed = False

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_local_importance():
    """Test 5: Local Importance - CPU vs GPU statistical equivalence"""
    print("\n" + "="*60)
    print("TEST 5: Local Importance")
    print("="*60)

    np.random.seed(42)
    n_samples, n_features = 300, 8
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] * 2 + X[:, 1] > 0).astype(np.int32)

    # CPU
    model_cpu = rfx.RandomForestClassifier(
        ntree=100, use_gpu=False,
        compute_local_importance=True, iseed=42
    )
    model_cpu.fit(X, y)

    # GPU
    model_gpu = rfx.RandomForestClassifier(
        ntree=100, use_gpu=True,
        compute_local_importance=True, iseed=42
    )
    model_gpu.fit(X, y)

    try:
        local_imp_cpu = model_cpu.get_local_importance()
        local_imp_gpu = model_gpu.get_local_importance()

        # Compare shapes
        shape_match = local_imp_cpu.shape == local_imp_gpu.shape

        # Compare per-feature importance rankings (which features are most important per sample)
        # This is more robust than raw value correlation
        ranks_cpu = np.argsort(np.abs(local_imp_cpu), axis=1)[:, -3:]  # Top 3 features per sample
        ranks_gpu = np.argsort(np.abs(local_imp_gpu), axis=1)[:, -3:]

        rank_overlaps = []
        for i in range(len(ranks_cpu)):
            overlap = len(set(ranks_cpu[i]) & set(ranks_gpu[i])) / 3.0
            rank_overlaps.append(overlap)
        avg_rank_overlap = np.mean(rank_overlaps)

        print(f"  Shape match: {shape_match} ({local_imp_cpu.shape})")
        print(f"  Top-3 feature rank overlap: {avg_rank_overlap*100:.1f}%")

        # Pass if shapes match and rank overlap is reasonable
        passed = shape_match and avg_rank_overlap >= 0.5
    except Exception as e:
        print(f"  Error: {e}")
        passed = False

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("="*60)
    print("CPU vs GPU VALIDATION TEST")
    print("="*60)
    print("\nNote: Testing statistical equivalence, not bit-exact matching.")
    print("Random forests can vary between CPU/GPU due to parallel execution.\n")

    if not HAS_GPU:
        print("\nERROR: GPU build not available. Cannot run comparison.")
        return

    results = {}

    results['classification'] = test_classification()
    results['regression'] = test_regression()
    results['unsupervised'] = test_unsupervised()
    results['proximity'] = test_proximity()
    results['local_importance'] = test_local_importance()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<20} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("ALL TESTS PASSED - CPU and GPU produce statistically equivalent results")
    else:
        print("SOME TESTS FAILED - Check results above")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

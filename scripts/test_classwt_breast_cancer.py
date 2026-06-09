#!/usr/bin/env python3
"""
Test Breiman class prior weighting on breast cancer dataset.
Tests: CPU dense, CPU sparse, GPU dense, GPU sparse
With: no weighting, balanced, explicit classwt
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / 'python'))
import RFXFuse as rfx

data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int32)

n0, n1 = (y == 0).sum(), (y == 1).sum()
print(f"Breast cancer: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Class 0 (malignant): {n0}  Class 1 (benign): {n1}  ratio: {n1/n0:.2f}:1")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Check GPU availability
try:
    has_gpu = rfx.cuda_is_available()
except:
    has_gpu = False
print(f"GPU available: {has_gpu}")
print()

# Make sparse version
try:
    from scipy.sparse import csr_matrix
    X_train_sparse = csr_matrix(X_train)
    X_test_sparse = csr_matrix(X_test)
    has_scipy = True
except ImportError:
    has_scipy = False
print(f"Scipy available (for sparse): {has_scipy}")
print()

classwt_configs = [
    ("no_wt",       {}),
    ("balanced",    {"class_weight": "balanced"}),
    ("boost_cls0",  {"classwt": [2.0, 1.0]}),
]

backends = [
    ("cpu_dense",  {"use_gpu": False, "use_sparse": False}),
    ("cpu_sparse", {"use_gpu": False, "use_sparse": True}),
]
if has_gpu:
    backends += [
        ("gpu_dense",  {"use_gpu": True, "use_sparse": False}),
        ("gpu_sparse", {"use_gpu": True, "use_sparse": True}),
    ]

results = []

for backend_name, backend_params in backends:
    if "sparse" in backend_name and not has_scipy:
        print(f"SKIP {backend_name} (no scipy)")
        continue

    for wt_name, wt_params in classwt_configs:
        label = f"{backend_name}/{wt_name}"
        print(f"--- {label} ---")

        params = dict(
            ntree=100, mtry=6, nodesize=1,
            compute_importance=False, compute_proximity=False,
            show_progress=False, iseed=42,
        )
        params.update(backend_params)
        params.update(wt_params)

        try:
            clf = rfx.RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

            oob_err = clf.get_oob_error()
            preds = clf.predict(X_test)
            proba = clf.predict_proba(X_test)

            # Sanity checks
            assert proba.shape == (len(y_test), 2), f"Bad shape: {proba.shape}"
            row_sums = proba.sum(axis=1)
            bad_rows = np.abs(row_sums - 1.0) > 0.01
            if bad_rows.any():
                print(f"  WARNING: {bad_rows.sum()} rows don't sum to 1.0")
                print(f"    example sums: {row_sums[bad_rows][:5]}")

            proba_1 = proba[:, 1]
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, proba_1)

            tp0 = np.sum((preds == 0) & (y_test == 0))
            fn0 = np.sum((preds == 1) & (y_test == 0))
            tp1 = np.sum((preds == 1) & (y_test == 1))
            fn1 = np.sum((preds == 0) & (y_test == 1))
            recall_0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
            recall_1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0

            mismatches = (preds != np.argmax(proba, axis=1)).sum()

            print(f"  OOB={oob_err:.4f}  Acc={acc:.4f}  AUC={auc:.4f}  "
                  f"R0={recall_0:.4f}  R1={recall_1:.4f}  pred_mismatch={mismatches}")
            status = "PASS"

            results.append({
                "label": label, "oob": oob_err, "acc": acc, "auc": auc,
                "r0": recall_0, "r1": recall_1, "status": status
            })
        except Exception as e:
            print(f"  FAIL: {e}")
            results.append({"label": label, "status": "FAIL", "oob": 0, "acc": 0, "auc": 0, "r0": 0, "r1": 0})

print()
print("=" * 85)
print("SUMMARY")
print("=" * 85)
print(f"{'Config':<30s} {'Status':>6s} {'OOB':>6s} {'Acc':>6s} {'AUC':>6s} {'R(0)':>6s} {'R(1)':>6s}")
print("-" * 85)
for r in results:
    if r["status"] == "PASS":
        print(f"{r['label']:<30s} {r['status']:>6s} {r['oob']:6.4f} {r['acc']:6.4f} {r['auc']:6.4f} {r['r0']:6.4f} {r['r1']:6.4f}")
    else:
        print(f"{r['label']:<30s} {r['status']:>6s}     --     --     --     --     --")

n_pass = sum(1 for r in results if r["status"] == "PASS")
n_total = len(results)
print(f"\n{n_pass}/{n_total} passed")
print("Done.")

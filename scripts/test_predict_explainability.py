"""
Test predict-time explainability methods for all three model types:
  - RandomForestClassifier
  - RandomForestRegressor
  - RandomForestUnsupervised

Methods tested:
  1. predict_proximity_importance(X)
  2. predict_local_importance(X, method=0)  (path attribution)
  3. predict_local_importance(X, method=1)  (permutation)
  4. predict_outlier_scores(X)
  5. predict_top_k_similar_with_explanations(X, k, n_explanations)
"""

import numpy as np
import sys

try:
    import RFXFuse as rfx
except ImportError:
    sys.exit("RFXFuse not importable. Run: python3 setup.py build_ext --inplace")

from sklearn.datasets import load_breast_cancer, load_diabetes

np.set_printoptions(precision=4, suppress=True)
CLASS_NAMES = {0: "malignant", 1: "benign"}

def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# =====================================================================
# 1. CLASSIFIER  (breast cancer, binary classification)
# =====================================================================
separator("CLASSIFIER: Breast Cancer")

data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int32)
n_samples, n_features = X.shape
feat_names = data.feature_names

X_train, X_test = X[:450], X[450:]
y_train, y_test = y[:450], y[450:]

clf = rfx.RandomForestClassifier(
    nsample=X_train.shape[0], ntree=200, mdim=n_features,
    mtry=int(np.sqrt(n_features)),
    compute_importance=True,
    compute_leaf_assignments=True,
    compute_proximity_importance=True,
)
clf.fit(X_train, y_train)
print(f"OOB error:  {clf.get_oob_error():.4f}")

preds = clf.predict(X_test)
acc = np.mean(preds == y_test)
print(f"Test acc:   {acc:.4f}")

# --- predict_proximity_importance ---
print("\n--- predict_proximity_importance ---")
prox_imp = clf.predict_proximity_importance(X_test[:5], n_repeats=5)
print(f"  shape: {prox_imp.shape}  (expected: (5, {n_features}))")
top3 = np.argsort(-prox_imp[0])[:3]
print(f"  sample 0 top-3: {[feat_names[i] for i in top3]}  values: {prox_imp[0, top3]}")
assert prox_imp.shape == (5, n_features)
assert np.all(prox_imp >= 0)

# --- predict_local_importance (method=0, path) ---
print("\n--- predict_local_importance (path, method=0) ---")
local_imp_path = clf.predict_local_importance(X_test[:5], method=0)
print(f"  shape: {local_imp_path.shape}  (expected: (5, {n_features}))")
top3 = np.argsort(-local_imp_path[0])[:3]
print(f"  sample 0 top-3: {[feat_names[i] for i in top3]}")
assert local_imp_path.shape == (5, n_features)
assert np.all(local_imp_path >= 0)

# --- predict_local_importance (method=1, permutation) ---
print("\n--- predict_local_importance (permutation, method=1) ---")
local_imp_perm = clf.predict_local_importance(X_test[:5], method=1, n_repeats=5)
print(f"  shape: {local_imp_perm.shape}  (expected: (5, {n_features}))")
top3 = np.argsort(-local_imp_perm[0])[:3]
print(f"  sample 0 top-3: {[feat_names[i] for i in top3]}  values: {local_imp_perm[0, top3]}")
assert local_imp_perm.shape == (5, n_features)

# --- predict_outlier_scores ---
print("\n--- predict_outlier_scores ---")
train_outlier_clf = clf.compute_outlier_scores()
print(f"  training outlier scores (first 5): {train_outlier_clf[:5]}")
outlier_scores = clf.predict_outlier_scores(X_test[:10])
print(f"  shape: {outlier_scores.shape}  (expected: (10,))")
print(f"  scores: {outlier_scores}")
print(f"  (normalized: >10 = outlier)")
assert outlier_scores.shape == (10,)

# --- predict_top_k_similar_with_explanations ---
print("\n--- predict_top_k_similar_with_explanations ---")
k, n_exp = 5, 3
indices, raw_sim, norm_sim, feat_idx, feat_val = clf.predict_top_k_similar_with_explanations(
    X_test[:3], k=k, n_explanations=n_exp)
print(f"  shapes: indices={indices.shape}, raw_sim={raw_sim.shape}, feat_idx={feat_idx.shape}")

for s in range(3):
    test_cls = CLASS_NAMES[y_test[s]]
    pred_cls = CLASS_NAMES[preds[s]]
    neighbor_classes = [CLASS_NAMES[y_train[i]] for i in indices[s]]
    same_pct = np.mean(y_train[indices[s]] == y_test[s]) * 100
    print(f"  Sample {s}: true={test_cls}, predicted={pred_cls}")
    print(f"    neighbors: {indices[s]}  classes: {neighbor_classes}  ({same_pct:.0f}% same class)")
    print(f"    raw_sim: {raw_sim[s]}  norm_sim: {norm_sim[s]}")
    print(f"    explaining features: {[feat_names[i] for i in feat_idx[s]]}  scores: {feat_val[s]}")

assert indices.shape == (3, k)
assert feat_idx.shape == (3, n_exp)
assert np.allclose(norm_sim[0, 0], 1.0)

print("\n  CLASSIFIER: ALL PASSED")


# =====================================================================
# 2. REGRESSOR  (diabetes dataset)
# =====================================================================
separator("REGRESSOR: Diabetes")

data_reg = load_diabetes()
Xr, yr = data_reg.data.astype(np.float32), data_reg.target.astype(np.float32)
n_reg, mdim_reg = Xr.shape
feat_names_r = data_reg.feature_names

Xr_train, Xr_test = Xr[:350], Xr[350:]
yr_train, yr_test = yr[:350], yr[350:]

reg = rfx.RandomForestRegressor(
    nsample=Xr_train.shape[0], ntree=200, mdim=mdim_reg,
    mtry=max(1, mdim_reg // 3),
    nodesize=5,
    compute_importance=True,
    compute_leaf_assignments=True,
    compute_proximity_importance=True,
)
reg.fit(Xr_train, yr_train)
print(f"OOB MSE:    {reg.get_oob_error():.4f}")

preds_r = reg.predict(Xr_test)
mse = np.mean((preds_r - yr_test) ** 2)
print(f"Test MSE:   {mse:.2f}")

# --- predict_proximity_importance ---
print("\n--- predict_proximity_importance ---")
prox_imp_r = reg.predict_proximity_importance(Xr_test[:5], n_repeats=5)
print(f"  shape: {prox_imp_r.shape}  (expected: (5, {mdim_reg}))")
top3 = np.argsort(-prox_imp_r[0])[:3]
print(f"  sample 0 top-3: {[feat_names_r[i] for i in top3]}  values: {prox_imp_r[0, top3]}")
assert prox_imp_r.shape == (5, mdim_reg)

# --- predict_local_importance (method=0, path) ---
print("\n--- predict_local_importance (path, method=0) ---")
local_imp_r_path = reg.predict_local_importance(Xr_test[:5], method=0)
print(f"  shape: {local_imp_r_path.shape}  (expected: (5, {mdim_reg}))")
assert local_imp_r_path.shape == (5, mdim_reg)

# --- predict_local_importance (method=1, MSE-based permutation) ---
print("\n--- predict_local_importance (permutation, method=1) ---")
local_imp_r_perm = reg.predict_local_importance(Xr_test[:5], method=1, n_repeats=5)
print(f"  shape: {local_imp_r_perm.shape}  (expected: (5, {mdim_reg}))")
top3 = np.argsort(-local_imp_r_perm[0])[:3]
print(f"  sample 0 top-3: {[feat_names_r[i] for i in top3]}  values: {local_imp_r_perm[0, top3]}")
assert local_imp_r_perm.shape == (5, mdim_reg)

# --- predict_outlier_scores ---
print("\n--- predict_outlier_scores ---")
train_outlier_reg = reg.compute_outlier_scores()
print(f"  training outlier scores (first 5): {train_outlier_reg[:5]}")
outlier_r = reg.predict_outlier_scores(Xr_test[:10])
print(f"  shape: {outlier_r.shape}  (expected: (10,))")
print(f"  scores: {outlier_r}")
print(f"  (normalized: >10 = outlier)")
assert outlier_r.shape == (10,)

# --- predict_top_k_similar_with_explanations ---
print("\n--- predict_top_k_similar_with_explanations ---")
k, n_exp = 5, 3
idx_r, raw_r, norm_r, fi_r, fv_r = reg.predict_top_k_similar_with_explanations(
    Xr_test[:3], k=k, n_explanations=n_exp)
print(f"  shapes: indices={idx_r.shape}, feat_idx={fi_r.shape}")

for s in range(3):
    pred_y = preds_r[s]
    true_y = yr_test[s]
    neighbor_ys = yr_train[idx_r[s]]
    avg_neighbor_y = np.mean(neighbor_ys)
    print(f"  Sample {s}: true_y={true_y:.1f}, predicted={pred_y:.1f}")
    print(f"    neighbor y values: {neighbor_ys}  (avg={avg_neighbor_y:.1f})")
    print(f"    |true - neighbor_avg| = {abs(true_y - avg_neighbor_y):.1f}")
    print(f"    raw_sim: {raw_r[s]}")
    print(f"    explaining features: {[feat_names_r[i] for i in fi_r[s]]}  scores: {fv_r[s]}")

assert idx_r.shape == (3, k)
assert fi_r.shape == (3, n_exp)

print("\n  REGRESSOR: ALL PASSED")


# =====================================================================
# 3. UNSUPERVISED  (breast cancer, no labels)
# =====================================================================
separator("UNSUPERVISED: Breast Cancer (no labels)")

unsup = rfx.RandomForestUnsupervised(
    nsample=X_train.shape[0], ntree=200, mdim=n_features,
    mtry=int(np.sqrt(n_features)),
    compute_leaf_assignments=True,
    compute_proximity_importance=True,
)
unsup.fit(X_train)
print(f"OOB error: {unsup.get_oob_error():.4f}")

# --- predict_proximity_importance ---
print("\n--- predict_proximity_importance ---")
prox_imp_u = unsup.predict_proximity_importance(X_test[:5], n_repeats=5)
print(f"  shape: {prox_imp_u.shape}  (expected: (5, {n_features}))")
assert prox_imp_u.shape == (5, n_features)

# --- predict_local_importance (method=0 only for unsupervised) ---
print("\n--- predict_local_importance (path, method=0) ---")
local_imp_u = unsup.predict_local_importance(X_test[:5], method=0)
print(f"  shape: {local_imp_u.shape}  (expected: (5, {n_features}))")
assert local_imp_u.shape == (5, n_features)

# method=1 should raise error for unsupervised
print("\n--- predict_local_importance (method=1, expect error) ---")
try:
    unsup.predict_local_importance(X_test[:5], method=1)
    print("  ERROR: should have raised RuntimeError!")
    sys.exit(1)
except RuntimeError as e:
    print(f"  Correctly raised: {e}")

# --- predict_outlier_scores ---
print("\n--- predict_outlier_scores ---")
train_outlier_unsup = unsup.compute_outlier_scores()
print(f"  training outlier scores (first 5): {train_outlier_unsup[:5]}")
outlier_u = unsup.predict_outlier_scores(X_test[:10])
print(f"  shape: {outlier_u.shape}  (expected: (10,))")
print(f"  scores: {outlier_u}")
print(f"  (normalized: >10 = outlier)")
assert outlier_u.shape == (10,)

# --- predict_top_k_similar ---
print("\n--- predict_top_k_similar ---")
idx_u, sc_u = unsup.predict_top_k_similar(X_test[:3], k=5)
print(f"  indices shape: {idx_u.shape}  scores shape: {sc_u.shape}")
assert idx_u.shape == (3, 5)

# --- predict_top_k_similar_with_explanations ---
print("\n--- predict_top_k_similar_with_explanations ---")
k, n_exp = 5, 3
idx_ue, raw_ue, norm_ue, fi_ue, fv_ue = unsup.predict_top_k_similar_with_explanations(
    X_test[:3], k=k, n_explanations=n_exp)
print(f"  shapes: indices={idx_ue.shape}, feat_idx={fi_ue.shape}")

for s in range(3):
    neighbor_true_classes = [CLASS_NAMES[y_train[i]] for i in idx_ue[s]]
    test_true_class = CLASS_NAMES[y_test[s]]
    same_pct = np.mean(y_train[idx_ue[s]] == y_test[s]) * 100
    print(f"  Sample {s}: true class (hidden from model)={test_true_class}")
    print(f"    neighbor true classes: {neighbor_true_classes}  ({same_pct:.0f}% same)")
    print(f"    raw_sim: {raw_ue[s]}")
    print(f"    explaining features: {[feat_names[i] for i in fi_ue[s]]}  scores: {fv_ue[s]}")

assert fi_ue.shape == (3, n_exp)

print("\n  UNSUPERVISED: ALL PASSED")


# =====================================================================
# SUMMARY
# =====================================================================
separator("ALL TESTS PASSED")
print("  Classifier:   predict_proximity_importance, predict_local_importance (path + perm),")
print("                predict_outlier_scores, predict_top_k_similar_with_explanations")
print("  Regressor:    predict_proximity_importance, predict_local_importance (path + MSE perm),")
print("                predict_outlier_scores, predict_top_k_similar_with_explanations")
print("  Unsupervised: predict_proximity_importance, predict_local_importance (path only),")
print("                predict_outlier_scores, predict_top_k_similar, predict_top_k_similar_with_explanations")
print()

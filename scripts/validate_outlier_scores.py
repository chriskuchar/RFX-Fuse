"""Validate that predict_outlier_scores produces intuitive results."""
import numpy as np
import RFXFuse as rfx
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data.astype(np.float32), data.target.astype(np.int32)
X_train, y_train = X[:450], y[:450]
X_test, y_test = X[450:], y[450:]

clf = rfx.RandomForestClassifier(
    nsample=450, ntree=200, mdim=30, mtry=5,
    compute_leaf_assignments=True)
clf.fit(X_train, y_train)

scores = clf.predict_outlier_scores(X_test)
preds = clf.predict(X_test)
correct = (preds == y_test)

print("=== Outlier Score vs Prediction Correctness ===")
if correct.sum() > 0:
    print(f"  Mean score (correct predictions):   {scores[correct].mean():.2f}  (n={correct.sum()})")
if (~correct).sum() > 0:
    print(f"  Mean score (incorrect predictions): {scores[~correct].mean():.2f}  (n={(~correct).sum()})")
else:
    print("  No incorrect predictions in test set")

print("\nTop 5 highest outlier scores:")
top5 = np.argsort(-scores)[:5]
for i in top5:
    tag = "WRONG" if preds[i] != y_test[i] else "ok"
    print(f"  test[{i:3d}]: score={scores[i]:8.2f}, true={y_test[i]}, pred={preds[i]}  [{tag}]")

print("\nBottom 5 lowest outlier scores (most normal):")
bot5 = np.argsort(scores)[:5]
for i in bot5:
    tag = "WRONG" if preds[i] != y_test[i] else "ok"
    print(f"  test[{i:3d}]: score={scores[i]:8.2f}, true={y_test[i]}, pred={preds[i]}  [{tag}]")

# Inject synthetic outlier: random noise far from training distribution
print("\n=== Synthetic Outlier Test ===")
np.random.seed(42)
outlier = np.random.randn(1, 30).astype(np.float32) * 100
outlier_score = clf.predict_outlier_scores(outlier)[0]
normal_score = scores.mean()
print(f"  Synthetic random outlier score: {outlier_score:.2f}")
print(f"  Mean normal test sample score:  {normal_score:.2f}")
print(f"  Ratio: {outlier_score / normal_score:.1f}x higher")

# Inject near-duplicate of training sample (should be very normal)
near_dup = X_train[0:1] + np.random.randn(1, 30).astype(np.float32) * 0.01
near_dup_score = clf.predict_outlier_scores(near_dup)[0]
print(f"  Near-duplicate of train[0] score: {near_dup_score:.2f}")
print(f"  Ratio vs mean: {near_dup_score / normal_score:.2f}x")

if outlier_score > normal_score and near_dup_score < normal_score:
    print("\n  PASS: Random outlier scores higher, near-duplicate scores lower than average")
else:
    print("\n  WARNING: Outlier ranking may not be as expected")

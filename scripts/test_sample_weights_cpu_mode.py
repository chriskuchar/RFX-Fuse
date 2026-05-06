import numpy as np
import sys

try:
    import RFXFuse
    print("RFXFuse imported successfully")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

np.random.seed(42)
X = np.random.rand(200, 5).astype(np.float32)
y = np.random.randint(0, 2, 200).astype(np.int32)
w = np.random.rand(200).astype(np.float32) + 0.1

# Test with use_gpu=False explicitly
clf = RFXFuse.RandomForestClassifier(ntree=50, use_gpu=False)
clf.fit(X, y, sample_weights=w)
print(f"Classification with sample_weights (CPU mode): OK, OOB error = {clf.get_oob_error():.4f}")

clf2 = RFXFuse.RandomForestClassifier(ntree=50, use_gpu=False)
clf2.fit(X, y)
print(f"Classification without sample_weights (CPU mode): OK, OOB error = {clf2.get_oob_error():.4f}")

y_reg = np.random.rand(200).astype(np.float32)
reg = RFXFuse.RandomForestRegressor(ntree=50, use_gpu=False)
reg.fit(X, y_reg, sample_weights=w)
print(f"Regression with sample_weights (CPU mode): OK, OOB MSE = {reg.get_oob_error():.4f}")

reg2 = RFXFuse.RandomForestRegressor(ntree=50, use_gpu=False)
reg2.fit(X, y_reg)
print(f"Regression without sample_weights (CPU mode): OK, OOB MSE = {reg2.get_oob_error():.4f}")

unsup = RFXFuse.RandomForestUnsupervised(ntree=20, use_gpu=False)
unsup.fit(X)
print(f"Unsupervised without sample_weights (CPU mode): OK")

print("\nAll CPU-mode tests passed!")

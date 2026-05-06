import numpy as np
import RFXFuse

np.random.seed(42)
X = np.random.rand(200, 5).astype(np.float32)
y = np.random.randint(0, 2, 200).astype(np.int32)
w = np.random.rand(200).astype(np.float32) + 0.1  # positive weights

# Test 1: Classification with sample_weights
clf = RFXFuse.RandomForestClassifier(ntree=50)
clf.fit(X, y, sample_weights=w)
print(f"Classification with sample_weights: OK, OOB error = {clf.get_oob_error():.4f}")

# Test 2: Classification without sample_weights (should still work)
clf2 = RFXFuse.RandomForestClassifier(ntree=50)
clf2.fit(X, y)
print(f"Classification without sample_weights: OK, OOB error = {clf2.get_oob_error():.4f}")

# Test 3: Regression with sample_weights
y_reg = np.random.rand(200).astype(np.float32)
reg = RFXFuse.RandomForestRegressor(ntree=50)
reg.fit(X, y_reg, sample_weights=w)
print(f"Regression with sample_weights: OK, OOB MSE = {reg.get_oob_error():.4f}")

# Test 4: Regression without sample_weights
reg2 = RFXFuse.RandomForestRegressor(ntree=50)
reg2.fit(X, y_reg)
print(f"Regression without sample_weights: OK, OOB MSE = {reg2.get_oob_error():.4f}")

# Test 5: Unsupervised (sample_weights should be ignored/deferred)
unsup = RFXFuse.RandomForestUnsupervised(ntree=20)
unsup.fit(X)
print(f"Unsupervised without sample_weights: OK")

print("\nAll tests passed!")

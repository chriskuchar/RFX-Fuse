import numpy as np
import RFXFuse

np.random.seed(42)
X = np.random.rand(200, 5).astype(np.float32)
y = np.random.randint(0, 2, 200).astype(np.int32)

clf = RFXFuse.RandomForestClassifier(ntree=50)
clf.fit(X, y)
print(f"Classification: OK, OOB error = {clf.get_oob_error():.4f}")

y_reg = np.random.rand(200).astype(np.float32)
reg = RFXFuse.RandomForestRegressor(ntree=50)
reg.fit(X, y_reg)
print(f"Regression: OK, OOB MSE = {reg.get_oob_error():.4f}")

unsup = RFXFuse.RandomForestUnsupervised(ntree=20)
unsup.fit(X)
print(f"Unsupervised: OK")

print("\nAll tests passed!")

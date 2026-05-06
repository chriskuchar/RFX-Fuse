import RFXFuse
print("RFXFuse imported OK")
clf = RFXFuse.RandomForestClassifier(ntree=10, use_gpu=False)
print("Classifier created OK")
import numpy as np
X = np.random.rand(50, 4).astype(np.float32)
y = (X[:, 0] > 0.5).astype(np.int32)
clf.fit(X, y)
print(f"OOB error: {clf.get_oob_error():.4f}")
print("CPU wheel test PASSED")

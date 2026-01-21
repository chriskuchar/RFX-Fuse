# RFX-Fuse API Reference

Complete API documentation for RFX-Fuse: Breiman and Cutler's Unified ML Engine.

---

## Table of Contents

- [RandomForestClassifier](#randomforestclassifier)
- [RandomForestRegressor](#randomforestregressor)
- [RandomForestUnsupervised](#randomforestunsupervised)
- [Model Persistence](#model-persistence)
- [Imputation](#imputation)
- [Visualization](#visualization)
- [Utility Functions](#utility-functions)
- [Data Loading](#data-loading)

---

## RandomForestClassifier

```python
rfx.RandomForestClassifier(
    ntree=100,
    mtry=0,
    maxcat=10,
    maxnode=0,
    minndsize=1,
    nodesize=5,
    iseed=12345,
    compute_proximity=False,
    compute_importance=True,
    compute_local_importance=False,
    use_gpu=False,
    use_qlora=False,
    quant_mode="nf4",
    rank=32,
    batch_size=0,
    use_casewise=False,
    use_rfgap=False,
    n_threads_cpu=0,
    show_progress=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `ntree` | int | 100 | Number of trees in the forest |
| `mtry` | int | 0 | Features to consider at each split. 0 = auto (sqrt(n_features)) |
| `maxcat` | int | 10 | Maximum categories for categorical variables |
| `maxnode` | int | 0 | Maximum nodes per tree. 0 = unlimited |
| `minndsize` | int | 1 | Minimum node size for splitting |
| `nodesize` | int | 5 | Minimum terminal node size |
| `iseed` | int | 12345 | Random seed for reproducibility |
| `compute_proximity` | bool | False | Compute sample proximity matrix |
| `compute_importance` | bool | True | Compute overall feature importance |
| `compute_local_importance` | bool | False | Compute per-sample feature importance |
| `use_gpu` | bool | False | Enable CUDA GPU acceleration |
| `use_qlora` | bool | False | Enable QLoRA low-rank proximity compression |
| `quant_mode` | str | "nf4" | Quantization mode: "int8", "nf4", "fp16", "fp32" |
| `rank` | int | 32 | Low-rank dimension for QLoRA compression |
| `batch_size` | int | 0 | GPU batch size. 0 = auto |
| `use_casewise` | bool | False | Use case-wise (bootstrap frequency) weighting |
| `use_rfgap` | bool | False | Use RF-GAP proximity normalization |
| `n_threads_cpu` | int | 0 | CPU threads. 0 = auto |
| `show_progress` | bool | True | Show training progress bar |

### Core Methods

#### fit(X, y)
Train the random forest classifier.
```python
model.fit(X, y)
```
**Returns:** self

#### predict(X)
Predict class labels for samples.
```python
predictions = model.predict(X)
```
**Returns:** ndarray of shape (n_samples,) with predicted class labels

#### predict_proba(X)
Predict class probabilities for samples.
```python
probabilities = model.predict_proba(X)
```
**Returns:** ndarray of shape (n_samples, n_classes) with class probabilities

#### get_oob_error()
Get out-of-bag error rate.
```python
error = model.get_oob_error()
```
**Returns:** float, OOB error rate (0.0 to 1.0)

### Variable Importance Methods

#### feature_importances_()
Get overall feature importance scores (permutation importance).
```python
importance = model.feature_importances_()
```
**Returns:** ndarray of shape (n_features,)

#### get_local_importance()
Get per-sample feature importance matrix.
```python
local_imp = model.get_local_importance()
# local_imp[i, j] = importance of feature j for sample i
```
**Returns:** ndarray of shape (n_samples, n_features)

**Note:** Requires `compute_local_importance=True` during training.

### Proximity Importance Methods (Novel)

#### get_proximity_importance()
Get local proximity importance matrix.
```python
prox_imp = model.get_proximity_importance()
# prox_imp[i, k] = feature k's contribution to sample i's similarity
```
**Returns:** ndarray of shape (n_samples, n_features)

**Note:** Requires `compute_proximity=True` during training.

#### get_local_proximity_importance()
Alias for `get_proximity_importance()`.

#### get_overall_proximity_importance()
Get overall proximity importance vector (mean across all samples).
```python
overall_prox_imp = model.get_overall_proximity_importance()
```
**Returns:** ndarray of shape (n_features,)

### Similarity Search Methods

#### get_top_k_similar(query_idx, k=10, exclude_self=True)
Get top-K most similar training samples.
```python
indices, scores = model.get_top_k_similar(query_idx, k=10)
```
**Returns:** Tuple of (indices, similarity_scores) arrays

#### get_top_k_similar_with_explanations(sample_idx, k=5, n_explanations=3)
Get top-K similar samples with feature explanations.
```python
result = model.get_top_k_similar_with_explanations(sample_idx, k=5, n_explanations=3)
indices, scores, per_sample_scores, feat_idx, feat_imp = result
```
**Returns:** Tuple of:
- `indices`: Top-K similar sample indices
- `scores`: Similarity scores
- `per_sample_scores`: Per-sample proximity importance for query
- `feat_idx`: Top feature indices explaining similarity
- `feat_imp`: Feature importance values for explanations

### Outlier Detection Methods

#### compute_outlier_scores(mode="full", n_anchors=100)
Compute Breiman-Cutler outlier scores for all training samples.
```python
scores = model.compute_outlier_scores()
# Scores > 10 typically indicate outliers
```
| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `mode` | str | "full" | "full" (exact) or "greedy" (approximate) |
| `n_anchors` | int | 100 | Anchor points for greedy mode |

**Returns:** ndarray of normalized outlier scores

#### compute_outliers(k=10, mode="full", n_anchors=100)
Get top-K outliers.
```python
top_indices, top_scores = model.compute_outliers(k=10)
```
**Returns:** Tuple of (indices, scores) for top-K outliers

### Proximity Matrix Methods

#### get_proximity_matrix()
Get full proximity matrix (CPU only, not for QLoRA).
```python
prox = model.get_proximity_matrix()
```
**Returns:** ndarray of shape (n_samples, n_samples)

#### get_lowrank_factors()
Get low-rank proximity factors A and B where P ≈ A @ B.T.
```python
A, B, rank = model.get_lowrank_factors()
```
**Returns:** tuple (A, B, rank)

#### compute_mds_from_factors(k=3)
Compute MDS coordinates from low-rank factors.
```python
mds = model.compute_mds_from_factors(k=3)
```
**Returns:** ndarray of shape (n_samples, k)

### Other Methods

#### get_prototypes(n_prototypes=3)
Get most representative samples in proximity space.
```python
prototypes = model.get_prototypes(n_prototypes=5)
# Returns: [(sample_idx, prototype_score), ...]
```

#### get_leaf_assignments()
Get leaf node assignments for all samples in all trees.
```python
leaves = model.get_leaf_assignments()
# Shape: (ntree, nsample)
```

#### save(filepath)
Save trained model to file.
```python
model.save("model.rfx")
```

---

## RandomForestRegressor

```python
rfx.RandomForestRegressor(
    ntree=100,
    mtry=0,
    maxnode=0,
    minndsize=5,
    nodesize=5,
    iseed=12345,
    compute_proximity=False,
    compute_importance=True,
    compute_local_importance=False,
    use_gpu=False,
    use_qlora=False,
    quant_mode="nf4",
    rank=32,
    batch_size=0,
    n_threads_cpu=0,
    show_progress=True
)
```

### Parameters

Same as RandomForestClassifier, except:
- No `nclass` parameter
- `minndsize` default is 5 (regression typically needs larger nodes)

### Methods

All methods from RandomForestClassifier are available, except:
- `predict_proba()` - not applicable for regression

#### get_oob_error()
For regression, returns OOB Mean Squared Error.
```python
mse = model.get_oob_error()
```

---

## RandomForestUnsupervised

Breiman and Cutler's unsupervised learning: classify real vs. synthetic permuted data.

```python
rfx.RandomForestUnsupervised(
    ntree=100,
    mtry=0,
    maxcat=10,
    maxnode=0,
    minndsize=1,
    nodesize=5,
    iseed=12345,
    compute_proximity=False,
    compute_importance=True,
    compute_local_importance=False,
    use_gpu=False,
    use_qlora=False,
    quant_mode="nf4",
    rank=32,
    batch_size=0,
    n_threads_cpu=0,
    show_progress=True
)
```

### Core Methods

#### fit(X)
Train on unlabeled data. Internally creates synthetic class by permuting features.
```python
model.fit(X)  # No labels needed
```

#### predict_proba(X)
Predict probability of being "real" vs "synthetic".
```python
proba = model.predict_proba(X)
# proba[:, 0] = P(synthetic), proba[:, 1] = P(real)
```

#### get_oob_error()
OOB error indicates ability to distinguish real from synthetic data.
- Low error (< 0.3): Strong feature dependencies detected
- High error (~ 0.5): Features are nearly independent

### All Other Methods

Same as RandomForestClassifier:
- `feature_importances_()`, `get_local_importance()`
- `get_proximity_importance()`, `get_overall_proximity_importance()`
- `get_top_k_similar()`, `get_top_k_similar_with_explanations()`
- `compute_outlier_scores()`, `compute_outliers()`
- `get_proximity_matrix()`, `get_lowrank_factors()`, `compute_mds_from_factors()`
- `save()`

---

## Model Persistence

### save(filepath)
Save a trained model.
```python
model.save("model.rfx")
```

### rfx.load(filepath)
Load a saved model.
```python
model = rfx.load("model.rfx")
# Works for Classifier, Regressor, or Unsupervised
```

---

## Imputation

### rfx_impute_rough(X, n_trees=100, n_iterations=5, use_gpu=True, verbose=False, seed=42)

Young-Cutler (2017) RF-based imputation method.

```python
from rfx_impute import rfx_impute_rough

X_imputed, n_iterations = rfx_impute_rough(
    X_missing,
    n_trees=100,
    n_iterations=5,
    use_gpu=True
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `X` | ndarray | required | Data with NaN values |
| `n_trees` | int | 100 | Trees per iteration |
| `n_iterations` | int | 5 | Refinement iterations |
| `use_gpu` | bool | True | GPU acceleration |
| `verbose` | bool | False | Print progress |
| `seed` | int | 42 | Random seed |

**Returns:** Tuple of (X_imputed, n_iterations)

---

## Visualization

### rfx.rfviz()

Generate interactive RFViz visualization with linked brushing.

```python
rfx.rfviz(
    rf_model,
    X,
    y,
    feature_names=None,
    class_names=None,
    n_clusters=3,
    title="RFViz",
    output_file="rfviz.html",
    show_in_browser=True,
    save_html=True,
    mds_k=3
)
```

**Features:**
- 2x2 dashboard layout
- Input features parallel coordinates
- Local importance parallel coordinates
- 3D MDS proximity plot
- Class votes heatmap
- Linked brushing across all plots

---

## Utility Functions

### rfx.cuda_is_available()
Check if CUDA GPU is available.
```python
if rfx.cuda_is_available():
    print("GPU acceleration available")
```

### rfx.clear_gpu_cache()
Clear GPU memory cache.
```python
rfx.clear_gpu_cache()
```

### rfx.get_gpu_memory_info()
Get GPU memory information.
```python
info = rfx.get_gpu_memory_info()
print(f"Free: {info['free'] / 1e9:.1f} GB")
```

---

## Data Loading

### rfx.load_wine()
Load the UCI Wine dataset.
```python
X, y = rfx.load_wine()
# X: (178, 13), y: (178,) with labels 0, 1, 2
```

### rfx.load_iris()
Load the UCI Iris dataset.
```python
X, y = rfx.load_iris()
# X: (150, 4), y: (150,) with labels 0, 1, 2
```

---

## Quantization Modes (QLoRA)

| Mode | Bits | Memory | Use Case |
|:-----|:-----|:-------|:---------|
| `fp32` | 32 | 1x | Debugging |
| `fp16` | 16 | 2x reduction | Default |
| `int8` | 8 | 4x reduction | Large datasets |
| `nf4` | 4 | 8x reduction | Very large datasets |

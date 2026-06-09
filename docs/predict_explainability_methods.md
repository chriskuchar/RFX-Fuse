# Predict-Time Explainability Methods

Five predict-time methods were implemented for new (unseen) data across all three model types: `RandomForestClassifier`, `RandomForestRegressor`, and `RandomForestUnsupervised`. These methods complement the existing training-time explainability (which operates on OOB samples) by providing the same types of insights for held-out or production data.

## Methods Overview

| Method | Classifier | Regressor | Unsupervised |
|--------|:---------:|:---------:|:------------:|
| `predict_proximity_importance(X, n_repeats)` | Yes | Yes | Yes |
| `predict_local_importance(X, method, n_repeats)` | path + perm | path + perm | path only |
| `predict_outlier_scores(X, mode, n_anchors)` | Yes | Yes | Yes |
| `predict_top_k_similar(X, k)` | Yes | Yes | Yes |
| `predict_top_k_similar_with_explanations(X, k, n_explanations)` | Yes | Yes | Yes |

---

## 1. `predict_proximity_importance(X, n_repeats=5)`

**What it measures:** For each feature, the fraction of trees whose terminal node changes when that feature is permuted with a donor value.

**Formula:**

For each new sample `x`, feature `k`, and `R` permutation repeats across `T` trees:

```
importance(x, k) = (1/R) * sum_r [ (1/T) * sum_t  I[ leaf_t(x) != leaf_t(x_permuted_r_k) ] ]
```

where `x_permuted_r_k` is sample `x` with feature `k` replaced by the value from donor row `r`.

**Output:** Array of shape `(n_samples, n_features)` with values in `[0, 1]`.

**Relationship to training:** The training-time `compute_proximity_importance` follows the same permutation approach but operates on OOB samples with a correctness filter (only accumulates for correctly-predicted OOB samples). At predict time, no ground truth is available, so all trees contribute equally without filtering.

---

## 2. `predict_local_importance(X, method=0, n_repeats=5)`

Two methods are available, selected via the `method` parameter.

### Method 0: Path Attribution

**What it measures:** How frequently each feature appears as a split variable on the root-to-leaf decision path, weighted inversely by path depth.

**Formula:**

For each sample `x` and tree `t`, let `P_t(x) = {v_1, v_2, ..., v_d}` be the split variables on the decision path of depth `d`:

```
importance(x, k) = (1/T) * sum_t [ count(v_j == k for v_j in P_t(x)) / |P_t(x)| ]
```

Features that appear on deeper paths receive lower weight per occurrence. This method requires no permutation and is available for all three model types including unsupervised.

### Method 1: Permutation (Classification)

**What it measures:** The fraction of permutation repeats where permuting feature `k` changes the ensemble majority vote.

**Formula:**

```
importance(x, k) = (1/R) * sum_r  I[ majority_vote(x) != majority_vote(x_permuted_r_k) ]
```

Values in `[0, 1]`; higher means the feature is more critical to the prediction.

### Method 1: Permutation (Regression)

**What it measures:** The mean absolute change in ensemble prediction when feature `k` is permuted.

**Formula:**

```
importance(x, k) = (1/R) * sum_r  | f_bar(x) - f_bar(x_permuted_r_k) |
```

where `f_bar(x) = (1/T) * sum_t f_t(x)` is the ensemble mean prediction.

**Unsupervised:** Method 1 is not available for unsupervised mode (no target to measure prediction change against). A `RuntimeError` is raised if attempted.

**Output:** Array of shape `(n_samples, n_features)`.

---

## 3. `predict_outlier_scores(X, mode='full', n_anchors=100)`

**What it measures:** How isolated a new sample is relative to the training data, based on leaf co-occurrence proximity.

**Formula (Breiman-Cutler):**

For new sample `x`, compute proximity to each training sample `i`:

```
prox(x, i) = (1/T) * sum_t  I[ leaf_t(x) == leaf_t(x_i_train) ]
```

Then the outlier score is:

```
score(x) = N / sum_i( prox(x, i)^2 )
```

where `N` is the count of training samples with positive proximity. Higher scores indicate greater isolation from training data.

**Modes:**
- `'full'`: Uses all training samples with positive proximity (exact).
- `'greedy'`: Uses only the top `n_anchors` most-proximate training samples (approximate, faster for large training sets).

**Output:** 1D array of shape `(n_samples,)`.

**Reference:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32. Section on outlier detection.

---

## 4. `predict_top_k_similar(X, k=10)`

**What it measures:** The `k` most similar training samples for each new sample, ranked by leaf co-occurrence proximity.

**Formula:**

Same proximity computation as outlier scores. Training samples are ranked by descending proximity and the top `k` are returned.

**Output:** Tuple of `(indices, similarity_scores)`, both of shape `(n_samples, k)`.

---

## 5. `predict_top_k_similar_with_explanations(X, k=5, n_explanations=3)`

**What it measures:** Same as `predict_top_k_similar`, plus per-feature explanations of *why* each neighbor is similar.

**Explanation method:**

For each tree where query `x` and training neighbor `i` land in the same leaf, the split variables on the query's decision path are credited. Each split variable on a path of depth `d` receives credit `1/d`. Credits are accumulated across co-occurring trees and aggregated across all `k` neighbors. The top `n_explanations` features by aggregated credit are returned.

**Output:** Tuple of `(indices, raw_similarities, normalized_similarities, feature_indices, feature_scores)`:
- `indices`: shape `(n_samples, k)` -- training sample indices
- `raw_similarities`: shape `(n_samples, k)` -- proximity scores in `[0, 1]`
- `normalized_similarities`: shape `(n_samples, k)` -- max-normalized to `[0, 1]`
- `feature_indices`: shape `(n_samples, n_explanations)` -- top explaining feature indices
- `feature_scores`: shape `(n_samples, n_explanations)` -- aggregated explanation scores

This output format matches the training-time `get_top_k_similar_with_explanations` API.

---

## Permutation Donor Sampling

At training time, Breiman's original implementation uses a Fisher-Yates shuffle of OOB sample indices to generate permuted feature values. At predict time, there is no OOB structure. Instead, donor rows are sampled uniformly with replacement from the input `X` itself, using `n_repeats` independent donor rows. This provides stable importance estimates without requiring access to the training data at predict time.

---

## Test Results

Tested on breast cancer (classification/unsupervised) and diabetes (regression) datasets.

### Classifier (Breast Cancer, 200 trees, 450 train / 119 test)

| Method | Result |
|--------|--------|
| Test accuracy | 97.5% |
| Proximity importance top features | worst perimeter, mean texture, worst area |
| Local importance (path) top features | worst concave points, worst area, worst radius |
| Top-K neighbors same-class rate | 100% for all 3 test samples |
| Explaining features | worst concave points, worst area, worst radius, worst perimeter |

### Regressor (Diabetes, 200 trees, 350 train / 92 test)

| Method | Result |
|--------|--------|
| Test MSE | 3412.5 |
| Proximity importance top features | bp, bmi, s1 |
| Local importance (perm) top features | s5, bp, bmi |
| Neighbor y-value accuracy | \|true - neighbor_avg\| = 4.8, 9.2, 6.0 |
| Explaining features | bp, bmi, s5, s3, s6 |

### Unsupervised (Breast Cancer, 200 trees, no labels)

| Method | Result |
|--------|--------|
| Neighbor same-class rate (hidden labels) | 100%, 100%, 80% |
| Explaining features | mean concavity, mean concave points, worst area, worst perimeter |

All results align with known domain knowledge: the "worst" morphological features are the primary drivers in breast cancer classification, and bmi/bp/s5 are the key diabetes progression predictors.

---

## Files Modified

- `include/rf_random_forest.hpp` -- Method declarations added to `RandomForest` class
- `src/rf_random_forest.cpp` -- C++ implementations of all five methods
- `python/randomforest_py.cpp` -- Python bindings for `RandomForestClassifier`, `RandomForestRegressor`, and `RandomForestUnsupervised`
- `scripts/test_predict_explainability.py` -- Test script covering all methods across all three model types

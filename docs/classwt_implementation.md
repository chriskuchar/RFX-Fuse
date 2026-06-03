# Breiman Class Prior Weighting -- Implementation Notes

## Overview

This implements the voting-based class weighting scheme from Breiman's Fortran V4+ random forest code, where **class weights modify the voting rule at prediction time** rather than the Gini splitting criterion.

## Background

Breiman's original 2001 paper noted that random forests underperformed AdaBoost on imbalanced data (Breiman, "Random Forests", *Machine Learning* 45(1), 2001, Section 8). In response, Breiman & Cutler introduced `classwt` in their Fortran V4.0 manual ("Using Random Forests V4.0", UC Berkeley, 2003), which weights votes at prediction time to adjust error rates between classes.

Andy Liaw (co-author of R's `randomForest`) described the motivation:

> "Prof. Breiman came up with the newer class weighting scheme implemented in the newer version of his Fortran code after we found that simply using the weights in the Gini index didn't seem to help much in extremely unbalanced data (say 1:100 or worse)."
> -- [R-help, 2011-09-13](https://stat.ethz.ch/pipermail/r-help/2011-September/289769.html)

> "Breiman uses the class weights to cast weighted votes."
> -- [R-help, 2004-01-12](https://stat.ethz.ch/pipermail/r-help/2004-January/044363.html)

## How It Works

Given a weight vector `classwt = [w0, w1, ..., w_{K-1}]`:

- **Prediction**: `predicted_class = argmax_c (votes[c] * classwt[c])`
- **Probability**: `P(c) = (votes[c] * classwt[c]) / sum_j (votes[j] * classwt[j])`

With `classwt = [1.0, 2.0]`, class 1 votes count double. A sample needs only ~33% of trees voting class 1 (instead of 50%) to be predicted as class 1.

Trees are grown identically regardless of class weights. The weighting only affects how votes are tallied.

## What Was Changed

### Config
- **`include/rf_random_forest.hpp`**: Added `std::vector<real_t> classwt` to `RandomForestConfig`
- **`include/rf_config.hpp`**: Added `std::vector<real_t> classwt` to `RFConfig` (global config used by GPU code)

### CPU -- Dense

**`src/rf_random_forest.cpp`**:

1. **OOB vote accumulation** -- Each tree's OOB vote for sample `n` into class `c` is multiplied by `classwt[c]`. Applies only when `task_type == CLASSIFICATION` and `classwt` is non-empty. Unsupervised mode is explicitly excluded.

2. **`predict_classification()` for new data** -- Changed the `votes` array from `integer_t` to `real_t` to support fractional weighted votes. Each tree vote is weighted by `classwt[predicted_class]`. The argmax finds the class with the highest weighted vote.

3. **`predict_proba()` for new data** -- Same weighted vote accumulation. Probabilities are normalized by the sum of weighted votes (not by `ntree`), so they sum to 1.0 even with non-uniform weights.

4. **`g_config.classwt` propagation** -- `fit()` and `fit_unsupervised()` copy `config_.classwt` to `g_config.classwt` so GPU kernels can access it.

### GPU -- Dense

**`cuda/rf_growtree.cu`**:

1. **`gpu_oob_vote_kernel_casewise`** -- Added `const real_t* classwt` parameter. When non-null, each vote is multiplied by `classwt[prediction]`. Works with both casewise (tnodewt-weighted) and non-casewise modes.

2. **`gpu_oob_vote_kernel`** (legacy) -- Same `classwt` parameter added. Multiplies `atomicAdd` value by `classwt[prediction]` when non-null.

3. **Device memory** -- `d_classwt` is allocated and populated from `g_config.classwt` when classification + non-empty classwt. Freed in cleanup.

### GPU -- Sparse

**`cuda/rf_testreebag_sparse.cu`** and **`cuda/rf_testreebag_sparse.cuh`**:

1. **`gpu_accumulate_oob_votes_kernel`** -- Added `classwt` parameter. Votes are `classwt[pred_class]` instead of `1.0f` when classwt is non-null.

2. **`gpu_adjust_casewise_votes_kernel`** -- For casewise mode, correctly undoes the initial weighted vote (`-classwt[c]`) and replaces with `tnodewt * classwt[c]`.

3. **Host functions** `gpu_testreebag_sparse()` and `gpu_adjust_casewise_votes()` -- Updated signatures to accept `d_classwt`, with default `nullptr`.

**`cuda/rf_sparse_forest.cu`**:

1. Added `d_classwt` allocation/free alongside `d_oob_votes`.
2. Passes `d_classwt` to both `gpu_testreebag_sparse()` and `gpu_adjust_casewise_votes()` call sites.

### Python Bindings

**`python/randomforest_py.cpp`**:

1. **Constructor parameters** -- Added `class_weight` (string) and `classwt` (object) to `RandomForestClassifier`:
   - `class_weight=""` (default): no weighting
   - `class_weight="balanced"`: auto-computes `classwt[c] = N / (K * n_c)` from training labels
   - `classwt=[w0, w1, ...]`: explicit per-class weights (overrides `class_weight`)

2. **Dense fit path** -- Computes classwt before `rf_ = new RandomForest(config_)`, so the config carries the weights into the C++ backend.

3. **Sparse fit path** -- Same classwt computation before creating the RandomForest.

4. **`py::init<>` type list** -- Updated to include `std::string, py::object` for the two new parameters.

## Unsupervised Mode

All classwt gates check `task_type == CLASSIFICATION`. Unsupervised mode (`task_type == UNSUPERVISED`) is never affected by class weighting, even though it shares some classification code paths. This is intentional -- unsupervised proximity computation should not be biased by class priors.

## Test Results (breast cancer dataset, 100 trees, iseed=42)

```
Config                         Status    OOB    Acc    AUC   R(0)   R(1)
-------------------------------------------------------------------------------------
cpu_dense/no_wt                  PASS 0.0402 0.9357 0.9893 0.9062 0.9533
cpu_dense/balanced               PASS 0.0528 0.9240 0.9893 0.9375 0.9159
cpu_dense/boost_cls0             PASS 0.0578 0.9240 0.9893 0.9531 0.9065
cpu_sparse/no_wt                 PASS 0.0402 0.9357 0.9893 0.9062 0.9533
cpu_sparse/balanced              PASS 0.0528 0.9240 0.9893 0.9375 0.9159
cpu_sparse/boost_cls0            PASS 0.0578 0.9240 0.9893 0.9531 0.9065
gpu_dense/no_wt                  PASS 0.0427 0.9240 0.9893 0.9062 0.9346
gpu_dense/balanced               PASS 0.0503 0.9064 0.9893 0.9062 0.9065
gpu_dense/boost_cls0             PASS 0.0578 0.9357 0.9893 0.9844 0.9065
gpu_sparse/no_wt                 PASS 0.0427 0.9240 0.9893 0.9062 0.9346
gpu_sparse/balanced              PASS 0.0503 0.9064 0.9893 0.9062 0.9065
gpu_sparse/boost_cls0            PASS 0.0578 0.9357 0.9893 0.9844 0.9065

12/12 passed
```

Key observations:
- **CPU dense = CPU sparse** (identical results, as expected)
- **GPU dense = GPU sparse** (identical results, as expected)
- **AUC is constant** across all weighting configs (0.9893) -- class weights shift the decision boundary, not the ranking
- **Boosting class 0** (`classwt=[2,1]`) increases malignant recall: 0.906 -> 0.953 (CPU), 0.906 -> 0.984 (GPU)
- **`classwt=[1,1]`** matches default exactly (verified in earlier run)

## References

- Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32. Section 8.
- Breiman, L. & Cutler, A. (2003). "Using Random Forests V4.0." UC Berkeley.
- Breiman, L. & Cutler, A. "Random Forests -- Classification Manual" (V5). UC Berkeley.
- Liaw, A. (2011). R-help mailing list, September 13, 2011.
- Liaw, A. (2004). R-help mailing list, January 12, 2004.

## Usage

```python
import RFXFuse as rfx

# No weighting (default -- identical to before)
clf = rfx.RandomForestClassifier(ntree=500)

# Auto-balanced: inverse class frequency
clf = rfx.RandomForestClassifier(ntree=500, class_weight="balanced")

# Explicit: weight minority class 3x
clf = rfx.RandomForestClassifier(ntree=500, classwt=[3.0, 1.0])

clf.fit(X_train, y_train)
preds = clf.predict(X_test)        # Weighted voting
probas = clf.predict_proba(X_test) # Prior-adjusted probabilities (sum to 1.0)
```

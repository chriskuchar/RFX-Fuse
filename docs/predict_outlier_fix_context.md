# Predict Outlier Scores Fix -- Session Context

## What Was Done (Previous Session)

Implemented five predict-time explainability methods for new (unseen) data across all three model types (`RandomForestClassifier`, `RandomForestRegressor`, `RandomForestUnsupervised`):

1. `predict_proximity_importance(X, n_repeats)` -- fraction of trees whose leaf changes when a feature is permuted
2. `predict_local_importance(X, method, n_repeats)` -- method=0 (path attribution), method=1 (permutation: vote-change for clf, mean-abs-change for reg)
3. `predict_outlier_scores(X, mode, n_anchors)` -- Breiman-Cutler isolation score
4. `predict_top_k_similar(X, k)` -- top-K nearest training samples by leaf co-occurrence
5. `predict_top_k_similar_with_explanations(X, k, n_explanations)` -- top-K + per-feature similarity explanations

All methods passed basic tests. See `docs/predict_explainability_methods.md` for full formulas and test results.

## What Was Fixed (This Session, Already Built & Installed)

### Problem: `predict_outlier_scores` was broken

The predict-time outlier scores had two critical issues:

1. **No normalization**: Training-time `compute_outlier_scores()` normalizes raw scores as `(raw - median) / MAD` (per-class for classification, global for regression/unsupervised), producing standardized scores where >10 = outlier. The predict-time version returned raw `N / sum(prox^2)` with no normalization, giving wildly different scales:
   - Classification: raw scores 2-9
   - Regression: raw scores 254-1394
   - These are not comparable

2. **Regression/unsupervised faked class=0**: The Python bindings for regressor and unsupervised created `std::vector<rf::integer_t> predictions(n_new, 0)` and passed it as `predicted_classes`. This is meaningless -- regression has no classes.

### Fix Applied (4 files changed)

**`include/rf_random_forest.hpp`:**
- Added three new member variables to store training-time normalization stats:
  - `std::map<integer_t, real_t> outlier_norm_median_` -- per-class median (clf) or global (key -1)
  - `std::map<integer_t, real_t> outlier_norm_mad_` -- per-class MAD (clf) or global (key -1)
  - `bool outlier_norm_computed_ = false`

**`src/rf_random_forest.cpp` -- `compute_outlier_scores()` (training-time):**
- Now stores median/MAD into `outlier_norm_median_` and `outlier_norm_mad_` during normalization
- Sets `outlier_norm_computed_ = true`
- No change to the raw computation or normalization logic itself

**`src/rf_random_forest.cpp` -- `predict_outlier_scores()` (predict-time):**
- Requires `compute_outlier_scores()` called first (throws error otherwise)
- Raw score uses ALL training samples with positive proximity (consistent with training-time raw computation)
- Classification: normalizes using per-class median/MAD looked up by predicted class
- Regression/Unsupervised: normalizes using global median/MAD (key -1)
- Output is now on the standardized scale where >10 = outlier

**`python/randomforest_py.cpp`:**
- Classifier binding: unchanged (still calls `predict_classification` and passes predicted classes)
- Regressor binding: now passes `nullptr` for `predicted_classes` (was faking class=0)
- Unsupervised binding: now passes `nullptr` for `predicted_classes` (was faking class=0)
- Updated docstrings for all three

**`scripts/test_predict_explainability.py`:**
- Added `compute_outlier_scores()` calls before each `predict_outlier_scores()` for all 3 model types
- Removed `assert np.all(outlier_scores > 0)` since normalized scores can be negative

### Design Note: Raw computation consistency

The training-time `compute_outlier_scores()` raw computation uses ALL training samples (not same-class) in the `N / sum(prox^2)` formula, despite the Breiman paper specifying same-class for classification. The per-class distinction only happens in the normalization step (per-class median/MAD). The predict-time implementation matches this: raw uses all samples, normalization is per-class for classification and global for regression/unsupervised.

## What Needs To Be Done

### Must Do
1. **Run the test script** to verify normalized scores are on the correct scale:
   ```bash
   python3 scripts/test_predict_explainability.py
   ```
   Expected: all three model types produce outlier scores on the standardized scale (most samples near 0, outliers > 10)

2. **Validate the scores make intuitive sense** -- e.g., inject a synthetic random-noise sample and verify it gets a high outlier score vs normal test samples

### Optional / Future Considerations
- The training-time raw formula doesn't filter by same-class for classification (diverges from Breiman's paper). Both training and predict are now consistent with each other, but could be updated to match the original formula in a future pass.
- The normalization stats (`outlier_norm_median_`, `outlier_norm_mad_`) are not serialized in save/load. After loading a saved model, `compute_outlier_scores()` must be called again before `predict_outlier_scores()`. Could add serialization if needed.

## Files Modified
- `include/rf_random_forest.hpp`
- `src/rf_random_forest.cpp`
- `python/randomforest_py.cpp`
- `scripts/test_predict_explainability.py`
- `scripts/validate_outlier_scores.py` (new, standalone validation script -- may need cleanup)

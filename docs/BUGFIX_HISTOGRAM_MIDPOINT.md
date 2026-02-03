# Bug Fix: Histogram Split Value Midpoint

**Date**: 2026-02-02
**Severity**: High (caused significant regression prediction errors on CPU)
**Affected Versions**: All versions prior to this fix
**Components**: CPU histogram-based split finding (classification and regression)

## Summary

The CPU histogram split finding code was using the **upper bin edge** as the split value, causing off-by-one errors during tree traversal. The fix uses the **midpoint between bin edges**, matching the GPU implementation.

## Symptoms

- CPU regression predictions were systematically biased (too low)
- Large discrepancy between CPU and GPU results for the same model
- Example on Bike Sharing dataset:
  - **Before fix (CPU)**: Test RMSE = 136.72, predictions biased low
  - **After fix (CPU)**: Test RMSE = 94.67, matches GPU exactly

## Root Cause

During histogram-based split finding, when a best split bin `b` is found, the split value determines how samples are partitioned during tree traversal:

```cpp
if (feature_value <= split_value) {
    go_left();
} else {
    go_right();
}
```

**Bug**: The split value was set to the upper edge of the bin:
```cpp
best_split_value = info.bin_edges[b + 1];  // WRONG
```

**Problem**: Samples with `feature_value == bin_edges[b + 1]` would satisfy `<=` and go left, but during training they were counted in bin `b+1` (right side). This creates inconsistency between training and prediction.

**Fix**: Use the midpoint between bin edges:
```cpp
best_split_value = (info.bin_edges[b] + info.bin_edges[b + 1]) / 2.0f;  // CORRECT
```

This ensures samples are correctly partitioned regardless of exact boundary values.

## Files Changed

| File | Line | Function |
|------|------|----------|
| `src/rf_histogram.cpp` | ~300 | `find_best_split_classification_histogram` |
| `src/rf_histogram.cpp` | ~367 | `find_best_split_regression_histogram` |
| `src/rf_growtree.cpp` | ~100 | `histogram_find_best_split_classification` |
| `src/rf_growtree.cpp` | ~183 | `histogram_find_best_split_regression` |

## Code Diff

```cpp
// BEFORE (all 4 locations):
if (info.is_categorical) {
    best_split_value = static_cast<real_t>(best_b);
} else {
    // FIX: Split point is at upper edge of best_bin
    best_split_value = info.bin_edges[best_b + 1];
}

// AFTER (all 4 locations):
if (info.is_categorical) {
    best_split_value = static_cast<real_t>(best_b);
} else {
    // Use MIDPOINT between edges to avoid off-by-one error with <= comparison
    // Without this fix, samples with value == bin_edges[best_b + 1] go to wrong child
    best_split_value = (info.bin_edges[best_b] + info.bin_edges[best_b + 1]) / 2.0f;
}
```

## Verification

After the fix, CPU and GPU produce identical results:

| Use Case | Metric | CPU (Before) | CPU (After) | GPU |
|----------|--------|--------------|-------------|-----|
| Regression (Bike Sharing) | Test RMSE | 136.72 | **94.67** | 94.67 |
| Regression (Bike Sharing) | Test MSE | 18693.17 | **8962.72** | 8962.72 |
| Regression (Bike Sharing) | Correlation | 0.6351 | **0.7172** | 0.7172 |
| Classification (Finance) | OOB Error | 10.2% | 10.2% | 10.4% |
| Classification (Finance) | Test Accuracy | 93.5% | 93.5% | 92.2% |

## Related GPU Fix

This same fix was previously applied to the GPU code in `cuda/rf_growtree.cu`:
- `gpu_histogram_find_split_classification_device` (line ~589)
- `gpu_histogram_find_split_regression_device` (line ~683)
- `gpu_histogram_find_split_classification_parallel` (line ~843)
- `gpu_histogram_find_split_regression_parallel` (line ~902)

## Testing

To verify the fix:
```bash
# Run regression use case
RFX_USE_GPU=0 python test_output/use_cases/usecase2_timeseries_regression.py

# Expected: Test RMSE ~94.67 (matching GPU)
```

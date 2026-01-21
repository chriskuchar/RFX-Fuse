# RFX-Fuse: Breiman and Cutler's Unified ML Engine (CPU)

[![PyPI](https://img.shields.io/pypi/v/rfx-fuse-cpu.svg)](https://pypi.org/project/rfx-fuse-cpu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**RFX-Fuse** delivers classification, regression, unsupervised learning, similarity search, explainability, and outlier detection from a single model.

**This is the CPU-only version** — no CUDA required.

## Installation

```bash
pip install rfx-fuse-cpu
```

**Prerequisites:** CMake 3.12+, Python 3.9+, C++17 compiler

## Quick Example

```python
import rfx

clf = rfx.RandomForestClassifier(
    ntree=500,
    use_gpu=False,
    compute_importance=True,
    compute_proximity=True
)
clf.fit(X, y)

# Four types of importance
var_imp = clf.feature_importances_()
prox_imp = clf.get_proximity_importance()

# Similarity search with explanations
indices, scores, _, feat_idx, feat_imp = clf.get_top_k_similar_with_explanations(0, k=10)

# Outlier detection
outliers, scores = clf.compute_outliers(k=10)
```

## Documentation

- **GitHub**: https://github.com/chriskuchar/RFX-Fuse
- **API Reference**: https://github.com/chriskuchar/RFX-Fuse/blob/main/docs/API.md
- **Examples**: https://github.com/chriskuchar/RFX-Fuse/tree/main/examples

## GPU Version

For GPU acceleration: `pip install rfx-fuse`

## License

MIT License

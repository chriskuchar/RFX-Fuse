# RFX-Fuse: Breiman and Cutler's Unified ML Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-00599C.svg?logo=cplusplus)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![arXiv](https://img.shields.io/badge/arXiv-submit/7171299-b31b1b.svg)](https://arxiv.org/abs/submit/7171299)

**RFX-Fuse** (Random Forests X [X=compression] — Forest Unified Learning and Similarity Engine) delivers Breiman and Cutler's complete vision for Random Forests as a Forests Unified Machine Learning and Similarity Engine with native GPU/CPU support.

Breiman and Cutler designed Random Forests as more than an ensemble predictor. Their original implementation included classification, regression, unsupervised learning, proximity-based similarity, outlier detection, missing value imputation, and visualization—capabilities that enable it to be a unified learning and simlarity engine with just 1-2 model objects. This vision has also been extended with ML's first Similarity Explanation overall and local, native in the model output. 

## Key Use Cases

| Use Case | RFX-Fuse | Comparable Approach |
|----------|----------|---------------------|
| Recommender Systems | 1–2 models | 5 tools |
| Finance Explainability | 1 model | 3 tools |
| Time Series Regression | 1 model | 4 tools |
| Imputation Validation | 1 model | *none exist* |
| Anomaly Detection | 1 model | 4 tools |

## Novel Contributions

1. **Proximity Importance** — ML's first explainable similarity: proximity measures *that* samples are similar; proximity importance explains *why*.

2. **Imputation Quality Validation** — Rank imputation methods by how "real" the imputed data looks, without ground truth labels.

## Four Types of Importance from One Model

| Type | Question Answered | Scope | Source |
|------|-------------------|-------|--------|
| Overall Variable Importance | Which features drive predictions? | Global | Restored |
| Local Variable Importance | Why THIS prediction? | Per-sample | Restored |
| Overall Proximity Importance | Which features define similarity? | Global | Novel |
| Local Proximity Importance | Why is THIS sample similar to neighbors? | Per-sample | Novel |

## Capabilities Comparison

| Feature | RFX-Fuse | XGBoost | sklearn RF | FAISS |
|---------|----------|---------|------------|-------|
| Classification | ✓ | ✓ | ✓ | — |
| Regression | ✓ | ✓ | ✓ | — |
| Unsupervised | ✓ | — | — | — |
| Overall importance | ✓ | ✓ | ✓ | — |
| Local importance (per-sample) | ✓ | SHAP | — | — |
| Proximity/similarity scoring | ✓ | — | — | ✓ |
| Overall proximity importance | ✓ | — | — | — |
| Local proximity importance | ✓ | — | — | — |
| Top-K similar with explanations | ✓ | — | — | — |
| Outlier detection with explanations | ✓ | — | — | — |
| Missing value imputation | ✓ | — | — | — |

## Installation

### From Source

```bash
git clone https://github.com/chriskuchar/RFX-Fuse.git
cd RFX-Fuse
pip install -e .
pip install -e ".[viz,examples]"
```

### Prerequisites

- **CMake** 3.12+
- **Python** 3.8+
- **C++ compiler** with C++17 support (GCC 7+, Clang 5+)
- **OpenMP** (usually included with compiler)
- **CUDA toolkit** 12.8+ (for GPU acceleration)

### Verify Installation

```python
import rfx
print(f"RFX-Fuse version: {rfx.__version__}")
print(f"CUDA enabled: {rfx.__cuda_enabled__}")
```

## Examples

Each use case has a complete demonstration script in the `examples/` folder:

| Use Case | Demo Script | Description |
|----------|-------------|-------------|
| **Recommender Systems** | [`examples/recommender_system/demo_recommender_system.py`](examples/recommender_system/demo_recommender_system.py) | MovieLens 25M: similarity retrieval + ranking with explanations |
| **Finance Explainability** | [`examples/classification/demo_loan_classification.py`](examples/classification/demo_loan_classification.py) | Loan default prediction with 4-type explainability |
| **Time Series Regression** | [`examples/time_series/demo_time_series.py`](examples/time_series/demo_time_series.py) | Bike sharing: prediction + outlier detection |
| **Imputation Validation** | [`examples/data_imputation/demo_imputation.py`](examples/data_imputation/demo_imputation.py) | Rank imputation methods without ground truth |
| **Anomaly Detection** | [`examples/anomaly_detection/demo_anomaly_detection.py`](examples/anomaly_detection/demo_anomaly_detection.py) | Breiman-Cutler outlier detection |

Run an example:
```bash
cd examples/time_series
python demo_time_series.py
```

## Industry Use Cases

### Use Case 1: Recommender Systems

RFX-Fuse Unsupervised for retrieval + RFX-Fuse Supervised for re-ranking on MovieLens 25M.

![Recommender System Results Stage 1 Similarity Scoring](examples/recommender_system/unsupervised_and_faiss.png)

![Recommender System Results Stage 2 Supervised Modeling](examples/recommender_system/supervised_prediction_similarity.png)

![Recommender System Results Stage 2 Outlier Detection](examples/recommender_system/supervised_outlier_detection.png)

![Recommender System Results Stage 2 Top K Retrieval](examples/recommender_system/unsupervised_supervised_boost.png)

**[View Code →](examples/recommender_system/demo_recommender_system.py)**

---

### Use Case 2: Finance Explainability

Single classifier provides regulatory-compliant explanations (ECOA, GDPR, Fair Lending).

![Finance Explainability Results](examples/classification/loan_classification_9panel_a.png)

![Finance Explainability Results](examples/classification/loan_classification_9panel_b.png)

**[View Code →](examples/classification/demo_loan_classification.py)**

---

### Use Case 3: Time Series Regression

RFX-Fuse Regressor on UCI Bike Sharing dataset with full explainability.

![Time Series Results](examples/time_series/comprehensive_15panel_analysis.png)

**[View Code →](examples/time_series/demo_time_series.py)**

---

### Use Case 4: Imputation Quality Validation

**Novel capability with no traditional alternative.** Rank imputation methods by how "real" the imputed data looks.

![Imputation Validation Results](examples/data_imputation/imputation_validation.png)

**[View Code →](examples/data_imputation/demo_imputation.py)**

---

### Use Case 5: Anomaly Detection

Breiman-Cutler method: train on clean data, anomalies have high P(synthetic).

![Anomaly Detection Results](examples/anomaly_detection/anomaly_detection.png)

**[View Code →](examples/anomaly_detection/demo_anomaly_detection.py)**

## API Reference

For complete API documentation with all parameters, methods, and examples, see **[docs/API.md](docs/API.md)**.

## Performance

**Benchmark Environment:** NVIDIA RTX 3060 (12GB), AMD Ryzen 7 5800X, 32GB RAM

| Use Case | Train Size | Features | Trees | Training Time |
|----------|------------|----------|-------|---------------|
| Recommender (Unsup) | 59,047 (×2) | 23 | 1,000 | 1,254s |
| Recommender (Sup) | 47,237 | 21 | 1,000 | 120s |
| Finance Classification | 46,396 | 15 | 500 | 69s |
| Bike Regression | 5,725 | 4 | 1,000 | 24s |
| Imputation Validation | 3,000 | 12 | 100 | 3.6s |
| Anomaly Detection | 15,000 | 8 | 100 | 112s |

*Training times include predictions, similarity scoring, proximity importance, local importance, and all explainability features where applicable.*

## Methodology

For detailed methodology, see the [arXiv paper](https://arxiv.org/abs/submit/7171299).

## Citation

```bibtex
@article{rfxfuse2025,
  title={RFX-Fuse: Breiman and Cutler's Unified ML Engine},
  author={Kuchar, Chris},
  year={2025}
}
```

## Acknowledgments

This work aims to implement the full unified learning and similarity engine Dr. Leo Breiman and Dr. Cutler created when they made their Fortran/Java implementation in the early 2000s.

Special thanks to Dr. Adele Cutler for generously sharing original Breiman-Cutler Random Forest source materials, which made this faithful restoration and extension possible.

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Breiman, L., & Cutler, A. Random Forests. https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
- Young, J. (2017). Imputation for Random Forests. Utah State University.

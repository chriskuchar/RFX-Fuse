#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eval.h>
#include "rf_random_forest.hpp"
#include "rf_types.hpp"
#include "rf_impute.hpp"
#include "rf_cuda_config.hpp"
#include "rf_config.hpp"
#include "rf_data_loader.hpp"
#include "rf_mds_cpu.hpp"
#include <cmath>
#include <set>
#include <algorithm>
#include <map>
#include <sstream>
#include <cctype>
#include <memory>
#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#include <iomanip>
#endif

namespace py = pybind11;

// ============================================================================
// SPARSE MATRIX SUPPORT - Convert scipy.sparse.csr_matrix to SparseMatrixCSR
// ============================================================================
// scipy.sparse.csr_matrix has attributes:
//   - data: array of non-zero values
//   - indices: array of column indices
//   - indptr: array of row pointers
//   - shape: tuple (nrows, ncols)
// ============================================================================

rf::SparseMatrixCSR scipy_csr_to_rfx(py::object scipy_csr) {
    // Extract CSR components from scipy sparse matrix
    py::array_t<rf::real_t> data = scipy_csr.attr("data").cast<py::array_t<rf::real_t>>();
    py::array_t<rf::integer_t> indices = scipy_csr.attr("indices").cast<py::array_t<rf::integer_t>>();
    py::array_t<rf::integer_t> indptr = scipy_csr.attr("indptr").cast<py::array_t<rf::integer_t>>();
    py::tuple shape = scipy_csr.attr("shape").cast<py::tuple>();
    
    rf::integer_t nrows = shape[0].cast<rf::integer_t>();
    rf::integer_t ncols = shape[1].cast<rf::integer_t>();
    rf::integer_t nnz = data.size();
    
    auto data_buf = data.request();
    auto indices_buf = indices.request();
    auto indptr_buf = indptr.request();
    
    return rf::SparseMatrixCSR(
        static_cast<const rf::real_t*>(data_buf.ptr),
        static_cast<const rf::integer_t*>(indices_buf.ptr),
        static_cast<const rf::integer_t*>(indptr_buf.ptr),
        nrows, ncols, nnz
    );
}

// Check if object is a scipy sparse matrix
bool is_scipy_sparse(py::object obj) {
    try {
        py::module_ scipy_sparse = py::module_::import("scipy.sparse");
        return py::isinstance(obj, scipy_sparse.attr("spmatrix"));
    } catch (...) {
        return false;
    }
}

// Convert scipy sparse to CSR format if needed
py::object ensure_csr(py::object sparse_matrix) {
    try {
        py::module_ scipy_sparse = py::module_::import("scipy.sparse");
        if (!scipy_sparse.attr("isspmatrix_csr")(sparse_matrix).cast<bool>()) {
            // Convert to CSR
            return sparse_matrix.attr("tocsr")();
        }
        return sparse_matrix;
    } catch (...) {
        throw std::runtime_error("Failed to convert sparse matrix to CSR format");
    }
}

// Standalone wrapper function for fit to avoid lambda capture issues in exec() context
// This function must return cleanly to avoid memcpy crashes in exec() context
void fit_wrapper(rf::RandomForest& self,
                 py::array_t<rf::real_t> X,
                 py::array y,
                 py::array_t<rf::real_t> sample_weight) {
    // Wrap entire function in try-catch to ensure clean return
    // This prevents any exceptions from causing stack corruption during return
    try {
    // CRITICAL: Ensure X is C-contiguous (row-major) before accessing data
    // NumPy arrays may be Fortran-contiguous (column-major) which would cause data corruption
    py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contiguous = X;
    
    // Check dimensions
    auto X_buf = X_contiguous.request();
    auto y_buf = y.request();

    if (X_buf.ndim != 2) {
        throw std::runtime_error("X must be 2-dimensional (samples, features)");
    }
    
    // For unsupervised learning, generate synthetic y values
    if (self.get_task_type() == rf::TaskType::UNSUPERVISED) {
        // Create synthetic y array with zeros for unsupervised learning
        py::array_t<rf::real_t> synthetic_y(X_buf.shape[0]);
        auto synthetic_y_buf = synthetic_y.request();
        rf::real_t* synthetic_y_ptr = static_cast<rf::real_t*>(synthetic_y_buf.ptr);
        
        // Fill with zeros (synthetic labels)
        for (py::ssize_t i = 0; i < X_buf.shape[0]; ++i) {
            synthetic_y_ptr[i] = 0.0f;
        }
        
        // Replace y with synthetic y
        y = synthetic_y;
        y_buf = y.request();
    }
    
    if (y_buf.ndim != 1) {
        throw std::runtime_error("y must be 1-dimensional");
    }
    if (X_buf.shape[0] != y_buf.shape[0]) {
        throw std::runtime_error("X and y must have same number of samples");
    }

    // Validate array sizes
    if (X_buf.shape[0] <= 0 || X_buf.shape[1] <= 0) {
        throw std::runtime_error("Invalid array dimensions");
    }
    if (y_buf.shape[0] <= 0) {
        throw std::runtime_error("Invalid y array size");
    }

    // Validate data integrity
    if (X_buf.size == 0) {
        throw std::runtime_error("X data is empty - cannot train on empty data");
    }
    if (y_buf.size == 0) {
        throw std::runtime_error("y data is empty - cannot train on empty targets");
    }
    
    // Validate data ranges for classification
    if (self.get_task_type() == rf::TaskType::CLASSIFICATION) {
        const float* y_data = static_cast<const float*>(y_buf.ptr);
        float min_y = *std::min_element(y_data, y_data + y_buf.size);
        float max_y = *std::max_element(y_data, y_data + y_buf.size);
        
        if (min_y < 0) {
            throw std::runtime_error("Classification labels must be non-negative, got minimum: " + std::to_string(min_y));
        }
        if (max_y >= 1000) {
            throw std::runtime_error("Classification labels too large, got maximum: " + std::to_string(max_y) + 
                                   " (max allowed: 999)");
        }
    }
    
    // Critical dimension validation to prevent buffer overruns
    rf::integer_t config_nsample = self.get_n_samples();
    rf::integer_t config_mdim = self.get_n_features();
    rf::integer_t actual_nsample = X_buf.shape[0];
    rf::integer_t actual_mdim = X_buf.shape[1];
    
    size_t expected_buffer_size = static_cast<size_t>(config_mdim) * static_cast<size_t>(config_nsample);
    size_t actual_buffer_size = static_cast<size_t>(actual_mdim) * static_cast<size_t>(actual_nsample);
    
    if (config_nsample != actual_nsample || config_mdim != actual_mdim) {
        throw std::runtime_error(
            "BUFFER OVERRUN PREVENTION:\n"
            "Config dimensions don't match data dimensions!\n"
            "This would cause a buffer overrun crash in C++ code.\n\n"
            "Config: nsample=" + std::to_string(config_nsample) + ", mdim=" + std::to_string(config_mdim) + 
            " (buffer size: " + std::to_string(expected_buffer_size) + ")\n"
            "Data: nsample=" + std::to_string(actual_nsample) + ", mdim=" + std::to_string(actual_mdim) + 
            " (buffer size: " + std::to_string(actual_buffer_size) + ")\n\n"
            "SOLUTION: Use auto-detection API:\n"
            "  RandomForestClassifier(X, y, ntree=50, ...)\n"
            "This automatically sets correct dimensions from your data."
        );
    }
    
    if (expected_buffer_size != actual_buffer_size) {
        throw std::runtime_error(
            "BUFFER SIZE MISMATCH: Expected " + std::to_string(expected_buffer_size) + 
            " elements, but data has " + std::to_string(actual_buffer_size) + " elements"
        );
    }
    
    if (expected_buffer_size > 100000000) {
        throw std::runtime_error(
            "MEMORY SAFETY: Dataset too large for safe processing.\n"
            "Buffer size: " + std::to_string(expected_buffer_size) + " elements\n"
            "This could cause memory exhaustion or crashes.\n"
            "Consider using a smaller dataset or different parameters."
        );
    }
    
    // Validate sample weights if provided
    if (sample_weight.size() > 0) {
        auto w_buf = sample_weight.request();
        const float* w_data = static_cast<const float*>(w_buf.ptr);
        
        for (py::ssize_t i = 0; i < w_buf.size; ++i) {
            if (w_data[i] < 0) {
                throw std::runtime_error("sample_weight contains negative values at index " + 
                                       std::to_string(i) + " - weights must be non-negative");
            }
        }
    }

    if (!X_buf.ptr || !y_buf.ptr) {
        throw std::runtime_error("Invalid input data pointers");
    }
    
    const rf::real_t* X_ptr = static_cast<const rf::real_t*>(X_buf.ptr);
    const rf::real_t* y_ptr = static_cast<const rf::real_t*>(y_buf.ptr);
    const rf::real_t* weight_ptr = nullptr;

    if (sample_weight.size() > 0) {
        auto w_buf = sample_weight.request();
        if (w_buf.shape[0] != X_buf.shape[0]) {
            throw std::runtime_error("sample_weight must have same length as X");
        }
        if (!w_buf.ptr) {
            throw std::runtime_error("Invalid sample_weight pointer");
        }
        weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
    }

    // Dispatch based on task type
    switch (self.get_task_type()) {
        case rf::TaskType::CLASSIFICATION: {
            // For classification, y must be integer type
            // Use heap allocation to avoid stack overflow in exec() context
            std::unique_ptr<rf::integer_t[]> y_int(new rf::integer_t[y_buf.size]);
            
            // Handle different input formats
            if (y_buf.format == py::format_descriptor<int32_t>::format() || 
                y_buf.format == "i") {
                const int32_t* y_int_ptr = static_cast<const int32_t*>(y_buf.ptr);
                for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                    y_int[i] = static_cast<rf::integer_t>(y_int_ptr[i]);
                }
            } else if (y_buf.format == "l" || y_buf.format == "q") {
                const int64_t* y_int64_ptr = static_cast<const int64_t*>(y_buf.ptr);
                for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                    y_int[i] = static_cast<rf::integer_t>(y_int64_ptr[i]);
                }
            } else if (y_buf.format == py::format_descriptor<rf::integer_t>::format()) {
                const rf::integer_t* y_int_ptr = static_cast<const rf::integer_t*>(y_buf.ptr);
                for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                    y_int[i] = y_int_ptr[i];
                }
            } else if (y_buf.format == "f" || y_buf.format == "d") {
                if (y_buf.format == "f") {
                    const float* y_float_ptr = static_cast<const float*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_int[i] = static_cast<rf::integer_t>(y_float_ptr[i]);
                    }
                } else {
                    const double* y_double_ptr = static_cast<const double*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_int[i] = static_cast<rf::integer_t>(y_double_ptr[i]);
                    }
                }
            } else {
                throw std::runtime_error("Unsupported y data format: " + y_buf.format);
            }
            
            // Call fit_classification and then explicitly clear y_int before return
            self.fit_classification(X_ptr, y_int.get(), weight_ptr);
            
            // Explicitly clear the heap-allocated array before return
            // This ensures clean memory state before returning to Python
            y_int.reset();
            break;
        }
        case rf::TaskType::REGRESSION:
            self.fit_regression(X_ptr, y_ptr, weight_ptr);
            break;
        case rf::TaskType::UNSUPERVISED:
            self.fit_unsupervised(X_ptr, weight_ptr);
            break;
    }
    
    // Ensure all local variables are cleared before return
    // This helps prevent stack corruption during return in exec() context
    // Explicitly clear any large local variables
    } catch (const std::exception& e) {
        // Re-throw as Python exception to ensure clean error handling
        throw py::value_error(std::string("fit failed: ") + e.what());
    } catch (...) {
        // Catch any other exceptions and convert to Python exception
        throw py::value_error("fit failed: unknown error");
    }
    
    // Flush all output streams before return to ensure clean state
    fflush(stdout);
    fflush(stderr);
    
    // Add explicit return to ensure clean stack unwinding
    // This helps prevent memcpy crashes during return in exec() context
    return;
}

// ============================================================================
// SPARSE FIT WRAPPER - Fit using sparse matrix input (scipy.sparse.csr_matrix)
// ============================================================================
// For recommender systems with sparse user-item matrices, this allows training
// directly on sparse data. Pipeline: CSR → fit → importance → proximity
// Memory savings: O(nnz) for data + O(n×rank) for low-rank proximity
// ============================================================================
void fit_sparse_wrapper(rf::RandomForest& self,
                        py::object X_sparse,
                        py::array y,
                        py::array_t<rf::real_t> sample_weight) {
    try {
        // Convert to CSR if not already
        py::object X_csr = ensure_csr(X_sparse);
        
        // Convert scipy CSR to RFX SparseMatrixCSR
        rf::SparseMatrixCSR sparse_matrix = scipy_csr_to_rfx(X_csr);
        
        if (!sparse_matrix.is_valid()) {
            throw std::runtime_error("Invalid sparse matrix structure");
        }
        
        // Get y buffer
        auto y_buf = y.request();
        
        // For unsupervised learning, y might be empty or synthetic
        if (self.get_task_type() == rf::TaskType::UNSUPERVISED) {
            const rf::real_t* weight_ptr = nullptr;
            if (sample_weight.size() > 0) {
                auto w_buf = sample_weight.request();
                weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
            }
            self.fit_unsupervised_sparse(sparse_matrix, weight_ptr);
            return;
        }
        
        // Get sample weights
        const rf::real_t* weight_ptr = nullptr;
        if (sample_weight.size() > 0) {
            auto w_buf = sample_weight.request();
            weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
        }
        
        // Dispatch based on task type
        switch (self.get_task_type()) {
            case rf::TaskType::CLASSIFICATION: {
                // Convert y to integer
                std::unique_ptr<rf::integer_t[]> y_int(new rf::integer_t[y_buf.size]);
                if (y_buf.format == "f" || y_buf.format == "d") {
                    const float* y_float = static_cast<const float*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_int[i] = static_cast<rf::integer_t>(y_float[i]);
                    }
                } else {
                    const int32_t* y_int_ptr = static_cast<const int32_t*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_int[i] = static_cast<rf::integer_t>(y_int_ptr[i]);
                    }
                }
                self.fit_classification_sparse(sparse_matrix, y_int.get(), weight_ptr);
                break;
            }
            case rf::TaskType::REGRESSION: {
                // Convert y to float
                std::vector<rf::real_t> y_float(y_buf.size);
                if (y_buf.format == "d") {
                    const double* y_double = static_cast<const double*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_float[i] = static_cast<rf::real_t>(y_double[i]);
                    }
                } else {
                    const float* y_ptr = static_cast<const float*>(y_buf.ptr);
                    for (py::ssize_t i = 0; i < y_buf.size; ++i) {
                        y_float[i] = y_ptr[i];
                    }
                }
                self.fit_regression_sparse(sparse_matrix, y_float.data(), weight_ptr);
                break;
            }
            default:
                throw std::runtime_error("Unknown task type for sparse fit");
        }
        
    } catch (const std::exception& e) {
        throw py::value_error(std::string("fit_sparse failed: ") + e.what());
    }
}

// Helper function to convert loss function string to integer with validation
inline rf::integer_t loss_function_string_to_int(const std::string& loss_str, 
                                                  rf::TaskType task_type, 
                                                  rf::integer_t nclass) {
    std::string loss_lower = loss_str;
    std::transform(loss_lower.begin(), loss_lower.end(), loss_lower.begin(), ::tolower);
    
    // Only support Gini for classification and MSE for regression
    if (loss_lower == "gini" || loss_lower == "gini_impurity") {
        if (task_type == rf::TaskType::REGRESSION) {
            throw std::runtime_error("Gini impurity can only be used for classification tasks, not regression");
        }
        return 0;  // Gini impurity
    } else if (loss_lower == "mse" || loss_lower == "mean_squared_error" || loss_lower == "l2") {
        if (task_type == rf::TaskType::CLASSIFICATION) {
            throw std::runtime_error("MSE/L2 can only be used for regression tasks, not classification");
        }
        return 3;  // MSE
    } else {
        throw std::runtime_error("Unknown loss function: '" + loss_str + "'. Valid options: "
            "Classification: 'gini'; "
            "Regression: 'mse'; "
            "Ranking/Survival: 'ranking', 'survival'");
    }
}

int quantization_string_to_int(const std::string& quant_str) {
    std::string quant_lower = quant_str;
    std::transform(quant_lower.begin(), quant_lower.end(), quant_lower.begin(), ::tolower);

    if (quant_lower == "fp32") {
        return 0;  // FP32 (single precision) - stored as FP16 internally for proximity
    } else if (quant_lower == "fp16") {
        return 1;  // FP16 (half precision)
    } else if (quant_lower == "int8") {
        return 2;  // INT8 (8-bit integer)
    } else if (quant_lower == "nf4") {
        return 3;  // NF4 (4-bit normalized float)
    } else {
        throw std::runtime_error("Unknown quantization level: '" + quant_str + 
            "'. Valid Options: 'fp32', 'fp16', 'int8', 'nf4'");
    }
}

PYBIND11_MODULE(RFXFuse, m) {
    m.doc() = "Random Forest CUDA Python bindings with exact C++ parameters";

    // Register TaskType enum
    py::enum_<rf::TaskType>(m, "TaskType")
        .value("CLASSIFICATION", rf::TaskType::CLASSIFICATION)
        .value("REGRESSION", rf::TaskType::REGRESSION)
        .value("UNSUPERVISED", rf::TaskType::UNSUPERVISED);

    // Register UnsupervisedMode enum
    py::enum_<rf::UnsupervisedMode>(m, "UnsupervisedMode")
        .value("CLASSIFICATION_STYLE", rf::UnsupervisedMode::CLASSIFICATION_STYLE)
        .value("REGRESSION_STYLE", rf::UnsupervisedMode::REGRESSION_STYLE);

    // Register QuantizationLevel as module-level constants
    m.attr("QuantizationLevel") = py::module_::import("types").attr("SimpleNamespace")(
        py::arg("FP32") = 0,
        py::arg("FP16") = 1,
        py::arg("INT8") = 2,
        py::arg("NF4") = 3
    );
    
    // LoadedModel class - wraps a loaded RandomForest for prediction
    class LoadedModel {
    public:
        std::unique_ptr<rf::RandomForest> rf_;

        LoadedModel(std::unique_ptr<rf::RandomForest> rf) : rf_(std::move(rf)) {}

        rf::real_t get_oob_error() const { return rf_->get_oob_error(); }
        rf::integer_t get_n_samples() const { return rf_->get_n_samples(); }
        rf::integer_t get_n_features() const { return rf_->get_n_features(); }
        rf::integer_t get_n_classes() const { return rf_->get_n_classes(); }
        rf::integer_t get_n_trees() const { return rf_->get_n_trees(); }
        rf::TaskType get_task_type() const { return rf_->get_task_type(); }

        py::array_t<rf::real_t> feature_importances_() {
            rf::integer_t mdim = rf_->get_n_features();
            py::array_t<rf::real_t> importances(mdim);
            const rf::real_t* imp_ptr = rf_->get_feature_importances();
            if (imp_ptr != nullptr) {
                std::copy(imp_ptr, imp_ptr + mdim, importances.mutable_data());
            } else {
                std::fill(importances.mutable_data(), importances.mutable_data() + mdim, 0.0f);
            }
            return importances;
        }

        py::object predict(py::array_t<rf::real_t> X) {
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            rf::integer_t nsamples = static_cast<rf::integer_t>(X_buf.shape[0]);

            // Auto-detect task type and call appropriate predict function
            rf::TaskType task_type = rf_->get_task_type();

            if (task_type == rf::TaskType::REGRESSION) {
                // Regression: return float predictions
                py::array_t<rf::real_t> predictions(nsamples);
                auto pred_buf = predictions.request();
                rf_->predict_regression(static_cast<rf::real_t*>(X_buf.ptr), nsamples,
                                       static_cast<rf::real_t*>(pred_buf.ptr));
                return predictions;
            } else {
                // Classification or Unsupervised: return integer labels
                py::array_t<rf::integer_t> predictions(nsamples);
                auto pred_buf = predictions.request();
                rf_->predict(static_cast<rf::real_t*>(X_buf.ptr), nsamples,
                            static_cast<rf::integer_t*>(pred_buf.ptr));
                return predictions;
            }
        }

        py::array_t<rf::real_t> predict_regression(py::array_t<rf::real_t> X) {
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            rf::integer_t nsamples = static_cast<rf::integer_t>(X_buf.shape[0]);

            py::array_t<rf::real_t> predictions(nsamples);
            auto pred_buf = predictions.request();

            rf_->predict_regression(static_cast<rf::real_t*>(X_buf.ptr), nsamples,
                                   static_cast<rf::real_t*>(pred_buf.ptr));

            return predictions;
        }

        py::array_t<rf::real_t> predict_proba(py::array_t<rf::real_t> X) {
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            rf::integer_t nsamples = static_cast<rf::integer_t>(X_buf.shape[0]);
            rf::integer_t nclasses = rf_->get_n_classes();

            py::array_t<rf::real_t> probabilities({static_cast<py::ssize_t>(nsamples), static_cast<py::ssize_t>(nclasses)});
            rf_->predict_proba(static_cast<rf::real_t*>(X_buf.ptr), nsamples, probabilities.mutable_data());

            return probabilities;
        }

        void save(const std::string& filepath) const {
            rf_->save(filepath);
        }

        // Get local proximity importance (per-sample, per-feature)
        py::array_t<rf::real_t> get_local_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();

            py::array_t<rf::real_t> result({static_cast<py::ssize_t>(nsamples), static_cast<py::ssize_t>(mdim)});
            auto buf = result.request();

            if (prox_imp_ptr) {
                std::copy(prox_imp_ptr, prox_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                std::fill(static_cast<rf::real_t*>(buf.ptr),
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }

        // Get overall proximity importance (aggregated across samples)
        py::array_t<rf::real_t> get_overall_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();

            py::array_t<rf::real_t> result(mdim);
            auto buf = result.request();
            rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);

            if (prox_imp_ptr && nsamples > 0) {
                for (rf::integer_t k = 0; k < mdim; ++k) {
                    rf::real_t sum = 0.0f;
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        sum += prox_imp_ptr[i * mdim + k];
                    }
                    dest[k] = sum / static_cast<rf::real_t>(nsamples);
                }
            } else {
                std::fill(dest, dest + mdim, 0.0f);
            }
            return result;
        }

        // Legacy alias for backward compatibility
        py::array_t<rf::real_t> get_proximity_importance() {
            return get_local_proximity_importance();
        }

        // Get local feature importance (per-sample, per-feature)
        py::array_t<rf::real_t> get_local_importance() {
            const rf::real_t* local_imp_ptr = rf_->get_qimpm();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();

            py::array_t<rf::real_t> result({static_cast<py::ssize_t>(nsamples), static_cast<py::ssize_t>(mdim)});
            auto buf = result.request();

            if (local_imp_ptr) {
                std::copy(local_imp_ptr, local_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                std::fill(static_cast<rf::real_t*>(buf.ptr),
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }

        // Get top-K similar samples
        py::tuple get_top_k_similar(rf::integer_t sample_idx, rf::integer_t k = 5) {
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error(
                    "Leaf assignments required. Set compute_leaf_assignments=True when creating the model."
                );
            }

            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t ntree = rf_->get_n_trees();
            if (sample_idx < 0 || sample_idx >= nsamples) throw std::runtime_error("sample_idx out of range");
            k = std::min(k, nsamples - 1);

            // Compute similarity from leaf assignments
            std::vector<rf::real_t> similarities(nsamples, 0.0f);
            for (rf::integer_t t = 0; t < ntree; ++t) {
                int16_t query_leaf = leaf_ptr[t * nsamples + sample_idx];
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (i != sample_idx && leaf_ptr[t * nsamples + i] == query_leaf) {
                        similarities[i] += 1.0f;
                    }
                }
            }
            // Normalize by ntree
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                similarities[i] /= static_cast<rf::real_t>(ntree);
            }

            // Get top-K
            std::vector<std::pair<rf::real_t, rf::integer_t>> sims;
            for (rf::integer_t j = 0; j < nsamples; ++j) {
                if (j != sample_idx) {
                    sims.push_back({similarities[j], j});
                }
            }
            std::partial_sort(sims.begin(), sims.begin() + k, sims.end(),
                [](auto& a, auto& b) { return a.first > b.first; });

            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> sim_arr(k);
            auto idx_buf = idx_arr.request();
            auto sim_buf = sim_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = sims[i].second;
                static_cast<rf::real_t*>(sim_buf.ptr)[i] = sims[i].first;
            }

            return py::make_tuple(idx_arr, sim_arr);
        }

        // Get top-K similar with explanations
        py::tuple get_top_k_similar_with_explanations(
            rf::integer_t sample_idx, rf::integer_t k = 5, rf::integer_t n_explanations = 3
        ) {
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error(
                    "Leaf assignments required. Set compute_leaf_assignments=True when creating the model."
                );
            }

            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            rf::integer_t ntree = rf_->get_n_trees();
            if (sample_idx < 0 || sample_idx >= nsamples) throw std::runtime_error("sample_idx out of range");
            k = std::min(k, nsamples - 1);
            n_explanations = std::min(n_explanations, mdim);

            // Compute similarity from leaf assignments
            std::vector<rf::real_t> similarities_raw(nsamples, 0.0f);
            for (rf::integer_t t = 0; t < ntree; ++t) {
                int16_t query_leaf = leaf_ptr[t * nsamples + sample_idx];
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (i != sample_idx && leaf_ptr[t * nsamples + i] == query_leaf) {
                        similarities_raw[i] += 1.0f;
                    }
                }
            }
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                similarities_raw[i] /= static_cast<rf::real_t>(ntree);
            }

            // Normalize
            std::vector<rf::real_t> similarities_norm = similarities_raw;
            rf::real_t max_sim = 0.0f;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                if (i != sample_idx && similarities_raw[i] > max_sim) max_sim = similarities_raw[i];
            }
            if (max_sim > 0.0f) {
                for (rf::integer_t i = 0; i < nsamples; ++i) similarities_norm[i] /= max_sim;
            }

            // Get top-K
            std::vector<std::tuple<rf::real_t, rf::real_t, rf::integer_t>> sims;
            for (rf::integer_t j = 0; j < nsamples; ++j) {
                if (j != sample_idx) sims.push_back({similarities_norm[j], similarities_raw[j], j});
            }
            std::partial_sort(sims.begin(), sims.begin() + k, sims.end(),
                [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });

            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> sim_raw_arr(k), sim_norm_arr(k);
            auto idx_buf = idx_arr.request();
            auto sim_raw_buf = sim_raw_arr.request();
            auto sim_norm_buf = sim_norm_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = std::get<2>(sims[i]);
                static_cast<rf::real_t*>(sim_raw_buf.ptr)[i] = std::get<1>(sims[i]);
                static_cast<rf::real_t*>(sim_norm_buf.ptr)[i] = std::get<0>(sims[i]);
            }

            // Get explanations from proximity importance
            const rf::real_t* pi = rf_->get_proximity_importance();
            py::array_t<rf::integer_t> feat_arr(n_explanations);
            py::array_t<rf::real_t> val_arr(n_explanations);
            auto feat_buf = feat_arr.request();
            auto val_buf = val_arr.request();
            if (pi) {
                std::vector<std::pair<rf::real_t, rf::integer_t>> fi;
                for (rf::integer_t f = 0; f < mdim; ++f) fi.push_back({pi[sample_idx * mdim + f], f});
                std::partial_sort(fi.begin(), fi.begin() + n_explanations, fi.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
                for (rf::integer_t i = 0; i < n_explanations; ++i) {
                    static_cast<rf::integer_t*>(feat_buf.ptr)[i] = fi[i].second;
                    static_cast<rf::real_t*>(val_buf.ptr)[i] = fi[i].first;
                }
            } else {
                std::fill_n(static_cast<rf::integer_t*>(feat_buf.ptr), n_explanations, 0);
                std::fill_n(static_cast<rf::real_t*>(val_buf.ptr), n_explanations, 0.0f);
            }

            return py::make_tuple(idx_arr, sim_raw_arr, sim_norm_arr, feat_arr, val_arr);
        }

        // Compute outlier scores
        py::array_t<rf::real_t> compute_outlier_scores(const std::string& mode = "greedy", rf::integer_t n_anchors = 100) {
            rf::integer_t nsamples = rf_->get_n_samples();
            py::array_t<rf::real_t> scores(nsamples);
            auto buf = scores.request();
            rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);

            // Try to use leaf assignments for efficient computation
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (leaf_ptr) {
                rf::integer_t ntree = rf_->get_n_trees();

                if (mode == "greedy") {
                    // Greedy mode: use n_anchors random anchors
                    std::vector<rf::integer_t> anchors(n_anchors);
                    std::srand(42);
                    for (rf::integer_t i = 0; i < n_anchors; ++i) {
                        anchors[i] = std::rand() % nsamples;
                    }

                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        rf::real_t max_prox = 0.0f;
                        for (rf::integer_t a : anchors) {
                            if (a == i) continue;
                            rf::real_t prox = 0.0f;
                            for (rf::integer_t t = 0; t < ntree; ++t) {
                                if (leaf_ptr[t * nsamples + i] == leaf_ptr[t * nsamples + a]) {
                                    prox += 1.0f;
                                }
                            }
                            prox /= static_cast<rf::real_t>(ntree);
                            if (prox > max_prox) max_prox = prox;
                        }
                        // Outlier score = 1 - max_proximity (lower proximity = higher outlier score)
                        dest[i] = 1.0f - max_prox;
                    }
                } else {
                    // Full mode: compute against all samples
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        rf::real_t sum_prox = 0.0f;
                        for (rf::integer_t j = 0; j < nsamples; ++j) {
                            if (j == i) continue;
                            rf::real_t prox = 0.0f;
                            for (rf::integer_t t = 0; t < ntree; ++t) {
                                if (leaf_ptr[t * nsamples + i] == leaf_ptr[t * nsamples + j]) {
                                    prox += 1.0f;
                                }
                            }
                            sum_prox += prox;
                        }
                        rf::real_t mean_prox = sum_prox / (static_cast<rf::real_t>(ntree) * (nsamples - 1));
                        dest[i] = 1.0f - mean_prox;
                    }
                }
            } else {
                // Fallback: use proximity matrix if available
                const rf::dp_t* prox_ptr = rf_->get_proximity_matrix();
                if (prox_ptr) {
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        rf::real_t sum_prox = 0.0f;
                        for (rf::integer_t j = 0; j < nsamples; ++j) {
                            if (j != i) sum_prox += static_cast<rf::real_t>(prox_ptr[i * nsamples + j]);
                        }
                        rf::real_t mean_prox = (nsamples > 1) ? sum_prox / (nsamples - 1) : 0.0f;
                        dest[i] = 1.0f - mean_prox;
                    }
                } else {
                    std::fill(dest, dest + nsamples, 0.0f);
                }
            }
            return scores;
        }

        // Compute top-k outliers
        py::tuple compute_outliers(rf::integer_t k = 5) {
            py::array_t<rf::real_t> scores = compute_outlier_scores("greedy", 100);
            auto buf = scores.request();
            rf::real_t* score_ptr = static_cast<rf::real_t*>(buf.ptr);
            rf::integer_t nsamples = rf_->get_n_samples();

            std::vector<std::pair<rf::real_t, rf::integer_t>> scored;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                scored.push_back({score_ptr[i], i});
            }
            k = std::min(k, nsamples);
            std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                [](auto& a, auto& b) { return a.first > b.first; });

            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> score_arr(k);
            auto idx_buf = idx_arr.request();
            auto score_buf = score_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = scored[i].second;
                static_cast<rf::real_t*>(score_buf.ptr)[i] = scored[i].first;
            }

            return py::make_tuple(idx_arr, score_arr);
        }
    };
    
    py::class_<LoadedModel>(m, "LoadedModel")
        .def("get_oob_error", &LoadedModel::get_oob_error)
        .def("get_n_samples", &LoadedModel::get_n_samples)
        .def("get_n_features", &LoadedModel::get_n_features)
        .def("get_n_classes", &LoadedModel::get_n_classes)
        .def("get_n_trees", &LoadedModel::get_n_trees)
        .def("get_task_type", &LoadedModel::get_task_type)
        .def("feature_importances_", &LoadedModel::feature_importances_)
        .def("predict", &LoadedModel::predict, py::arg("X"),
             "Predict values. Auto-detects task type: returns float for regression, int for classification/unsupervised.")
        .def("predict_regression", &LoadedModel::predict_regression, py::arg("X"),
             "Predict values (regression only).")
        .def("predict_proba", &LoadedModel::predict_proba, py::arg("X"),
             "Predict class probabilities (classification only).")
        .def("save", &LoadedModel::save, py::arg("filepath"),
             "Save model to file.")
        .def("get_proximity_importance", &LoadedModel::get_proximity_importance,
             "Get local proximity importance matrix (n_samples x n_features).")
        .def("get_local_proximity_importance", &LoadedModel::get_local_proximity_importance,
             "Get local proximity importance matrix (n_samples x n_features).")
        .def("get_overall_proximity_importance", &LoadedModel::get_overall_proximity_importance,
             "Get overall proximity importance vector (n_features,).")
        .def("get_local_importance", &LoadedModel::get_local_importance,
             "Get local feature importance matrix (n_samples x n_features).")
        .def("get_top_k_similar", &LoadedModel::get_top_k_similar,
             py::arg("sample_idx"), py::arg("k") = 5,
             "Get top-K similar samples to a query sample.")
        .def("get_top_k_similar_with_explanations", &LoadedModel::get_top_k_similar_with_explanations,
             py::arg("sample_idx"), py::arg("k") = 5, py::arg("n_explanations") = 3,
             "Get top-K similar samples with feature explanations.")
        .def("compute_outlier_scores", &LoadedModel::compute_outlier_scores,
             py::arg("mode") = "greedy", py::arg("n_anchors") = 100,
             "Compute outlier scores for all samples.")
        .def("compute_outliers", &LoadedModel::compute_outliers,
             py::arg("k") = 5,
             "Get top-k outliers (indices and scores).");
    
    // Module-level load function for loading saved models
    m.def("load", [](const std::string& filepath) {
        auto model_ptr = rf::RandomForest::load(filepath);
        return LoadedModel(std::move(model_ptr));
    }, py::arg("filepath"),
       "Load a saved RFX model from file. Returns a LoadedModel that supports predict(), "
       "feature_importances_(), and save(). Use model.save() to save trained models.");

    // Register RandomForestConfig
    py::class_<rf::RandomForestConfig>(m, "RandomForestConfig")
        .def(py::init<>())
        .def_readwrite("nsample", &rf::RandomForestConfig::nsample, "Number of samples")
        .def_readwrite("ntree", &rf::RandomForestConfig::ntree, "Number of trees")
        .def_readwrite("mdim", &rf::RandomForestConfig::mdim, "Number of features")
        .def_readwrite("nclass", &rf::RandomForestConfig::nclass, "Number of classes")
        .def_readwrite("maxcat", &rf::RandomForestConfig::maxcat, "Maximum categories per feature")
        .def_readwrite("mtry", &rf::RandomForestConfig::mtry, "Number of variables to try at each split")
        .def_readwrite("maxnode", &rf::RandomForestConfig::maxnode, "Maximum number of nodes")
        .def_readwrite("minndsize", &rf::RandomForestConfig::minndsize, "Minimum node size")
        .def_readwrite("ncsplit", &rf::RandomForestConfig::ncsplit, "Number of random splits for big categoricals")
        .def_readwrite("ncmax", &rf::RandomForestConfig::ncmax, "Threshold for big categorical")
        .def_readwrite("iseed", &rf::RandomForestConfig::iseed, "Random seed")
        .def_readwrite("task_type", &rf::RandomForestConfig::task_type, "Task type")
        .def_readwrite("compute_proximity", &rf::RandomForestConfig::compute_proximity, "Compute proximity matrix")
        .def_readwrite("use_rfgap", &rf::RandomForestConfig::use_rfgap, "Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity instead of standard proximity")
        .def_readwrite("compute_importance", &rf::RandomForestConfig::compute_importance, "Compute feature importance")
        .def_readwrite("compute_local_importance", &rf::RandomForestConfig::compute_local_importance, "Compute local importance")
        .def_readwrite("use_gpu", &rf::RandomForestConfig::use_gpu, "Use GPU acceleration")
        // execution_mode is deprecated - use use_gpu instead
        .def_readwrite("use_qlora", &rf::RandomForestConfig::use_qlora, "Use QLORA-Proximity compression")
        .def_readwrite("quant_mode", &rf::RandomForestConfig::quant_mode, "Quantization mode")
        .def_readwrite("use_sparse", &rf::RandomForestConfig::use_sparse, "Use block-sparse mode")
        .def_readwrite("sparsity_threshold", &rf::RandomForestConfig::sparsity_threshold, "Sparsity threshold")
        .def_readwrite("batch_size", &rf::RandomForestConfig::batch_size, "GPU batch size for tree processing (0=auto, uses 20% less than available CUDA memory for WSL safety)")
        .def_readwrite("nodesize", &rf::RandomForestConfig::nodesize, "Minimum terminal node size for regression")
        .def_readwrite("cutoff", &rf::RandomForestConfig::cutoff, "Early stopping threshold")
        .def_readwrite("use_unsupervised", &rf::RandomForestConfig::use_unsupervised, "Enable unsupervised mode")
        .def_readwrite("outlier_threshold", &rf::RandomForestConfig::outlier_threshold, "Threshold for outlier detection")
        .def_readwrite("unsupervised_mode", &rf::RandomForestConfig::unsupervised_mode, "Unsupervised mode: CLASSIFICATION_STYLE or REGRESSION_STYLE");

    // XGBoost-style wrapper class for lazy CUDA initialization
    class RandomForestWrapper {
    private:
        rf::RandomForestConfig config_;
        std::shared_ptr<rf::RandomForest> rf_;
        bool initialized_;
        
    public:
        RandomForestWrapper(const rf::RandomForestConfig& config) 
            : config_(config), initialized_(false) {}
        
        void fit(py::array_t<rf::real_t> X, py::array y, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // Initialize RandomForest lazily when fit is called
            if (!initialized_) {
                try {
                    // Add safety checks before creating RandomForest
                    if (config_.nsample <= 0 || config_.ntree <= 0 || config_.mdim <= 0) {
                        throw std::runtime_error("Invalid configuration: nsample, ntree, and mdim must be positive");
                    }
                    
                    // Check memory requirements before attempting GPU initialization
                    if (config_.use_gpu) {
                        try {
                            // Initialize CUDA config to check memory
                            rf::cuda::CudaConfig::instance().initialize();
                            
                            // Check if execution should stop instead of falling back to CPU
                            if (rf::cuda::should_stop_on_insufficient_memory(config_.nsample, config_.mdim, config_.ntree)) {
                                throw std::runtime_error(
                                    "MEMORY SAFETY: Insufficient GPU memory for this problem size.\n"
                                    "Problem requires more than 50% of available GPU memory.\n"
                                    "This would likely cause WSL crashes.\n"
                                    "SOLUTION: Reduce problem size (nsample, ntree, or mdim) or use CPU mode."
                                );
                            }
                            
                            // Check if GPU can handle the problem
                            if (!rf::cuda::gpu_can_handle_problem(config_.nsample, config_.mdim, config_.ntree)) {
                                throw std::runtime_error(
                                    "MEMORY SAFETY: GPU cannot handle this problem size safely.\n"
                                    "Problem requires more memory than available with safety margins.\n"
                                    "This would likely cause WSL crashes.\n"
                                    "SOLUTION: Reduce problem size (nsample, ntree, or mdim) or use CPU mode."
                                );
                            }
                        } catch (const std::exception& e) {
                            // If memory check fails, stop instead of falling back
                            throw std::runtime_error("GPU memory safety check failed: " + std::string(e.what()));
                        }
                    } else {
                        // Check CPU memory requirements
                        try {
                            if (!rf::cuda::cpu_can_handle_problem(config_.nsample, config_.mdim, config_.ntree)) {
                                throw std::runtime_error(
                                    "MEMORY SAFETY: CPU cannot handle this problem size safely.\n"
                                    "Problem requires more memory than available for safe processing.\n"
                                    "This would likely cause WSL crashes.\n"
                                    "SOLUTION: Reduce problem size (nsample, ntree, or mdim)."
                                );
                            }
                        } catch (const std::exception& e) {
                            throw std::runtime_error("CPU memory safety check failed: " + std::string(e.what()));
                        }
                    }
                    
                    // Limit memory allocation to prevent crashes
                    size_t max_nodes = static_cast<size_t>(config_.ntree) * static_cast<size_t>(config_.maxnode);
                    if (max_nodes > 1000000) {  // 1M nodes limit
                        throw std::runtime_error("Configuration would allocate too much memory. Reduce ntree or maxnode.");
                    }
                    
                    // Use Python print instead of std::cout to avoid stream conflicts
                    py::module_ builtins = py::module_::import("builtins");
                    builtins.attr("print")(py::str("Creating RandomForest with nsample={}, ntree={}, mdim={}, maxnode={}")
                                           .format(config_.nsample, config_.ntree, config_.mdim, config_.maxnode));
                    
                    rf_ = std::make_shared<rf::RandomForest>(config_);
                    initialized_ = true;
                    
                    builtins.attr("print")("RandomForest created successfully!");
                    
                } catch (const std::exception& e) {
                    // Use Python print instead of std::cerr to avoid stream conflicts
                    py::module_ builtins = py::module_::import("builtins");
                    builtins.attr("print")(py::str("RandomForest creation failed: {}").format(e.what()), py::arg("file")=py::module_::import("sys").attr("stderr"));
                    
                    // NO MORE FALLBACK TO CPU - Stop the job instead
                    throw std::runtime_error("RandomForest creation failed: " + std::string(e.what()));
                }
            }
            
            // Ensure C-contiguous arrays
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            auto y_buf = y.request();
            
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional (samples, features)");
            }
            
            // Handle different y data types
            if (y_buf.format == py::format_descriptor<rf::real_t>::format()) {
                rf_->fit(static_cast<rf::real_t*>(X_buf.ptr), 
                        static_cast<rf::real_t*>(y_buf.ptr), 
                        sample_weight.size() > 0 ? static_cast<rf::real_t*>(sample_weight.request().ptr) : nullptr);
            } else {
                // Convert y to integer format for classification
                py::array_t<rf::integer_t> y_int = py::cast<py::array_t<rf::integer_t>>(y);
                rf_->fit(static_cast<rf::real_t*>(X_buf.ptr), 
                        static_cast<rf::integer_t*>(y_int.request().ptr), 
                        sample_weight.size() > 0 ? static_cast<rf::real_t*>(sample_weight.request().ptr) : nullptr);
            }
        }
        
        py::array_t<rf::real_t> predict(py::array_t<rf::real_t> X) {
            if (!initialized_) {
                throw std::runtime_error("Model must be fitted before prediction");
            }
            
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            rf::integer_t nsamples = X_buf.shape[0];
            
            // Create output array
            py::array_t<rf::real_t> predictions(nsamples);
            auto pred_buf = predictions.request();
            
            rf_->predict(static_cast<rf::real_t*>(X_buf.ptr), nsamples, pred_buf.ptr);
            return predictions;
        }
        
        rf::real_t get_oob_error() {
            if (!initialized_) {
                throw std::runtime_error("Model must be fitted before getting OOB error");
            }
            return rf_->get_oob_error();
        }
        
        py::array_t<rf::real_t> feature_importances_() {
            if (!initialized_) {
                throw std::runtime_error("Model must be fitted before getting feature importance");
            }
            // Get the feature importance pointer and convert to numpy array
            const rf::real_t* importance_ptr = rf_->get_feature_importances();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result(mdim);
            auto buf = result.request();
            std::copy(importance_ptr, importance_ptr + mdim, static_cast<rf::real_t*>(buf.ptr));
            return result;
        }
        
        py::array_t<rf::real_t> get_local_importance() {
            if (!initialized_) {
                throw std::runtime_error("Model must be fitted before getting local importance");
            }
            // Get local importance from qimpm_ (works for both standard and CLIQUE methods)
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            const rf::real_t* qimpm_ptr = rf_->get_qimpm();
            
            if (!qimpm_ptr) {
                // Return zero array if local importance not computed
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            std::fill(static_cast<rf::real_t*>(buf.ptr), 
                     static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
                return result;
            }
            
            // Copy local importance matrix to numpy array
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            std::copy(qimpm_ptr, qimpm_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            return result;
        }
        
        py::array_t<rf::real_t> compute_proximity_matrix() {
            if (!initialized_) {
                throw std::runtime_error("Model must be fitted before computing proximity matrix");
            }
            // Get the proximity matrix pointer and convert to numpy array
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            rf::integer_t nsamples = rf_->get_n_samples();
            
            if (!proximity_ptr) {
                // Check if low-rank mode is active
                if (rf_->get_compute_proximity()) {
                                throw std::runtime_error(
                        "Proximity matrix not available in full form. "
                        "Low-rank mode is active (use_qlora=True). "
                        "Full matrix reconstruction would require 80GB+ memory. "
                        "Proximity is stored in low-rank factors (A and B). "
                        "Use get_lowrank_factors() or compute_distances_from_factors() instead. "
                        "Or disable use_qlora=True for smaller datasets that can fit full matrix."
                    );
                    } else {
                    throw std::runtime_error("Proximity matrix not available. Did you enable compute_proximity=True?");
                }
            }
            
            py::array_t<rf::real_t> result({nsamples, nsamples});
            auto buf = result.request();
            std::copy(proximity_ptr, proximity_ptr + nsamples * nsamples, static_cast<rf::real_t*>(buf.ptr));
            return result;
        }
        
        rf::TaskType get_task_type() const {
            return config_.task_type;
        }
    };


    // Register RandomForest class with shared_ptr
    py::class_<rf::RandomForest, std::shared_ptr<rf::RandomForest>>(m, "RandomForest")
        .def("fit", &fit_wrapper,
             py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
             "Fit the random forest model")
        
        .def("fit_sparse", &fit_sparse_wrapper,
             py::arg("X_sparse"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
             "Fit the random forest model using sparse matrix input (scipy.sparse.csr_matrix)")

        .def("predict", [](rf::RandomForest& self,
                          py::array_t<rf::real_t> X) -> py::object {
            // Ensure C-contiguous array
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional (samples, features), got " + std::to_string(X_buf.ndim) + " dimensions");
            }
            if (!X_buf.ptr) {
                throw std::runtime_error("Invalid input data pointer");
            }
            
            rf::integer_t nsamples = X_buf.shape[0];
            rf::integer_t nfeatures = X_buf.shape[1];
            
            if (nsamples <= 0) {
                throw std::runtime_error("Invalid number of samples: " + std::to_string(nsamples));
            }
            if (nfeatures <= 0) {
                throw std::runtime_error("Invalid number of features: " + std::to_string(nfeatures));
            }
            
            // Get task type and dispatch accordingly
            rf::TaskType task_type = self.get_task_type();
            
            if (task_type == rf::TaskType::CLASSIFICATION) {
                // Classification: return class predictions
                py::array_t<rf::integer_t> predictions(nsamples);
                auto pred_buf = predictions.request();
                self.predict(static_cast<rf::real_t*>(X_buf.ptr), nsamples, pred_buf.ptr);
                return predictions;
            } else if (task_type == rf::TaskType::REGRESSION) {
                // Regression: return continuous predictions
                py::array_t<rf::real_t> predictions(nsamples);
                auto pred_buf = predictions.request();
                self.predict(static_cast<rf::real_t*>(X_buf.ptr), nsamples, pred_buf.ptr);
                return predictions;
            } else {
                // Unsupervised: return outlier scores
                py::array_t<rf::real_t> predictions(nsamples);
                auto pred_buf = predictions.request();
                self.predict(static_cast<rf::real_t*>(X_buf.ptr), nsamples, pred_buf.ptr);
                return predictions;
            }
        }, py::arg("X"), "Predict using the random forest model");

    // Progress bar wrapper class for RandomForestClassifier
    class RandomForestClassifier {
    private:
        rf::RandomForestConfig config_;  // Store config to allow updates
        rf::RandomForest* rf_;  // Use raw pointer to avoid shared_ptr destruction issues
        bool show_progress_;
        std::string progress_desc_;
        py::object progress_bar_;  // Store tqdm progress bar object
        int last_progress_refresh_pos_;  // Track last refresh position for throttling
        std::string gpu_loss_function_str_;  // Store original loss function string for re-validation
        py::object categorical_features_;  // User-specified categorical feature indices (list or None)
        
    public:
        // Getter for internal RF pointer (needed for lambda bindings)
        rf::RandomForest* get_rf_ptr() { return rf_; }
        RandomForestClassifier(int nsample = 1000,
                                int ntree = 100,
                                int mdim = 10,
                                int nclass = 2,
                                int maxcat = 10,
                                int mtry = 0,
                                int maxnode = 0,
                                int minndsize = 1,
                                int ncsplit = 25,
                                int ncmax = 25,
                                int iseed = 12345,
                                bool compute_proximity = false,
                                bool compute_importance = true,
                                bool compute_local_importance = false,
                                bool use_gpu = false,
                                bool use_qlora = false,
                                std::string quant_mode = "nf4",
                                bool use_sparse = false,
                                float sparsity_threshold = 1e-6f,
                                int batch_size = 0,
                                int nodesize = 5,
                                        float cutoff = 0.01f,
                                        bool show_progress = true,
                                        std::string progress_desc = "Training Random Forest",
                                std::string gpu_loss_function = "gini",  // "gini" for classification, "mse" for regression
                                int rank = 32,  // Low-rank proximity matrix rank (32 preserves 99%+ geometry)
                                int power_iterations = 30,  // Power iteration count for low-rank SVD (30=99% convergence, 100=99.9%)
                                int n_threads_cpu = 0,  // Number of CPU threads for multi-threading (0 = auto-detect)
                                bool use_rfgap = false,  // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
                                std::string importance_method = "local_imp",  // Local importance method: "local_imp" or "clique"
                                int clique_M = -1,  // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)
                                bool use_casewise = false,  // Use case-wise calculations (per-sample) vs non-case-wise (aggregated). Non-case-wise follows UC Berkeley standard.
                                bool compute_proximity_importance = false,  // Measures % of trees where permuting each feature changes terminal node. For interpretability.
                                bool compute_leaf_assignments = false,  // Store terminal node per sample per tree for on-demand proximity. Memory: ntree*nsample*2 bytes.
                                py::object categorical_features = py::none(),  // List of column indices (0-based) for categorical features, or None for auto-detection
                                bool use_histogram = true,  // Use histogram-based split finding (faster for large datasets)
                                int n_bins = 256) {  // Number of bins for histogram (max 256)
        categorical_features_ = categorical_features;
        config_.use_histogram = use_histogram;
        config_.n_bins = n_bins;
        config_.task_type = rf::TaskType::CLASSIFICATION;
        config_.nsample = nsample;
        config_.ntree = ntree;
        config_.mdim = mdim;
        config_.nclass = nclass;
        config_.maxcat = maxcat;
        config_.mtry = mtry > 0 ? mtry : static_cast<rf::integer_t>(std::sqrt(static_cast<rf::real_t>(mdim)));
        config_.maxnode = rf::calculate_maxnode(nsample, minndsize, maxnode);
        config_.minndsize = minndsize;
        config_.ncsplit = ncsplit;
        config_.ncmax = ncmax;
        config_.iseed = iseed;
        config_.compute_proximity = compute_proximity;
        config_.compute_importance = compute_importance;
        config_.compute_local_importance = compute_local_importance;
        config_.use_gpu = use_gpu;
        // execution_mode is deprecated - only use use_gpu
        config_.execution_mode = use_gpu ? "gpu" : "cpu";  // Set for backward compatibility only
        config_.use_qlora = use_qlora;
        config_.quant_mode = quantization_string_to_int(quant_mode);
        config_.use_sparse = use_sparse;
        config_.sparsity_threshold = sparsity_threshold;
        config_.batch_size = batch_size;
        config_.nodesize = nodesize;
        config_.cutoff = cutoff;
        // Convert loss function string to integer with validation (only Gini or MSE)
        config_.gpu_loss_function = loss_function_string_to_int(gpu_loss_function, config_.task_type, config_.nclass);
        // gpu_parallel_mode0 is now automatically set based on batch_size in fit() method
        config_.lowrank_rank = rank;
        config_.lowrank_power_iterations = power_iterations;
        config_.n_threads_cpu = n_threads_cpu;
        config_.use_rfgap = use_rfgap;
        config_.importance_method = importance_method;
        config_.clique_M = clique_M;
        config_.use_casewise = use_casewise;
        config_.compute_proximity_importance = compute_proximity_importance;
        config_.compute_leaf_assignments = compute_leaf_assignments;
        
        // Store original string for re-validation in fit()
        gpu_loss_function_str_ = gpu_loss_function;
        
        // Don't create RandomForest yet - will be created in fit() with actual data dimensions
        // rf_ will be nullptr until fit() is called
        rf_ = nullptr;  // Initialize to nullptr
        show_progress_ = show_progress;
        progress_desc_ = progress_desc;
        progress_bar_ = py::none();  // Initialize progress bar to None
        last_progress_refresh_pos_ = 0;  // Initialize refresh position tracker
        }
        
        // Destructor - safely clean up raw pointer and progress bar
        // This must be exception-safe for Jupyter notebooks
        ~RandomForestClassifier() {
            // Wrap entire destructor in try-catch to prevent any exceptions from propagating
            // In Jupyter, exceptions in destructors can cause kernel crashes
            try {
                // Clear progress bar reference first (safest operation)
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore - progress bar might already be invalid
                }
                
                // Delete RandomForest object with extra safety
                if (rf_ != nullptr) {
                    try {
                    delete rf_;
                    } catch (const std::exception& e) {
                        // Log but don't rethrow - we're in a destructor
                        // CUDA context might be corrupted in Jupyter
                    } catch (...) {
                        // Ignore all other errors during cleanup
                    }
                    rf_ = nullptr;  // Always set to nullptr after delete attempt
                }
            } catch (...) {
                // Ultimate safety net - catch absolutely everything
                // Don't do anything, just ensure pointers are null
                rf_ = nullptr;
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore
                }
            }
        }
        
        // Set use_sparse flag (for auto-detection in fit)
        void set_use_sparse(bool use_sparse) {
            config_.use_sparse = use_sparse;
        }
        
        void fit(py::array_t<rf::real_t> X, py::array y, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // Ensure C-contiguous array
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            auto y_buf = y.request();
            auto sample_weight_buf = sample_weight.request();
            
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional (samples, features)");
            }
            
            // Get actual data dimensions
            rf::integer_t actual_nsample = static_cast<rf::integer_t>(X_buf.shape[0]);
            rf::integer_t actual_mdim = static_cast<rf::integer_t>(X_buf.shape[1]);
            
            // Determine actual nclass from y data
            rf::integer_t actual_nclass = config_.nclass;
            if (y_buf.format != py::format_descriptor<rf::real_t>::format()) {
                // Classification: count unique classes
                py::array_t<rf::integer_t> y_int = py::cast<py::array_t<rf::integer_t>>(y);
                auto y_int_buf = y_int.request();
                std::set<rf::integer_t> unique_classes;
                for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                    unique_classes.insert(static_cast<rf::integer_t*>(y_int_buf.ptr)[i]);
                }
                actual_nclass = static_cast<rf::integer_t>(unique_classes.size());
            }
            
            // Update config with actual dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            config_.nclass = actual_nclass;
            config_.maxnode = rf::calculate_maxnode(config_.nsample, config_.minndsize, config_.maxnode);
            if (config_.mtry == 0) {
                config_.mtry = static_cast<rf::integer_t>(std::sqrt(static_cast<rf::real_t>(config_.mdim)));
            }
            
            // Re-validate loss function with actual nclass (in case it changed)
            config_.gpu_loss_function = loss_function_string_to_int(gpu_loss_function_str_, config_.task_type, config_.nclass);
            
            // Determine categorical features: user-specified or auto-detected
            std::vector<rf::integer_t> detected_cat(actual_mdim, 1);  // Default: all quantitative
            const rf::real_t* X_ptr = static_cast<const rf::real_t*>(X_buf.ptr);
            
            if (!categorical_features_.is_none()) {
                // User specified categorical features - use their list
                py::list cat_list = categorical_features_.cast<py::list>();
                for (auto item : cat_list) {
                    rf::integer_t col_idx = item.cast<rf::integer_t>();
                    if (col_idx >= 0 && col_idx < actual_mdim) {
                        // Count unique values for this categorical feature
                        std::set<rf::real_t> unique_values;
                        for (rf::integer_t n = 0; n < actual_nsample; ++n) {
                            unique_values.insert(X_ptr[n * actual_mdim + col_idx]);
                        }
                        detected_cat[col_idx] = static_cast<rf::integer_t>(unique_values.size());
                    }
                }
            } else {
                // Auto-detect categorical features: integer-like with small number of unique values
                for (rf::integer_t m = 0; m < actual_mdim; ++m) {
                    std::set<rf::real_t> unique_values;
                    bool all_integers = true;
                    rf::real_t min_val = 1e10f;
                    rf::real_t max_val = -1e10f;
                    
                    // Check all samples for this feature
                    for (rf::integer_t n = 0; n < actual_nsample; ++n) {
                        rf::real_t val = X_ptr[n * actual_mdim + m];
                        unique_values.insert(val);
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        
                        // Check if value is close to an integer
                        rf::real_t rounded = std::round(val);
                        if (std::abs(val - rounded) > 1e-5f) {
                            all_integers = false;
                        }
                    }
                    
                    // Check if feature is categorical (3+ unique values only)
                    // Binary features (2 unique values) use quantitative path for better sparsity handling
                    rf::integer_t num_unique = static_cast<rf::integer_t>(unique_values.size());
                    if (all_integers && num_unique > 2 && num_unique <= config_.maxcat && 
                        min_val >= 0.0f && max_val < static_cast<rf::real_t>(config_.maxcat)) {
                        // Feature is categorical - set cat[m] to number of categories
                        detected_cat[m] = num_unique;
                    }
                }
            }
            
            // Create or recreate RandomForest with correct dimensions
            rf_ = new rf::RandomForest(config_);
            
            // Set categorical features
            rf_->set_categorical_features(detected_cat.data(), actual_mdim);
            
            // Print training message
            py::module_ builtins = py::module_::import("builtins");
            builtins.attr("print")(py::str("Training Random Forest Classifier with {} trees...").format(config_.ntree));
            
            // Print batch size info for GPU mode
            if (config_.use_gpu) {
                rf::integer_t batch_size_info;
                std::string batch_mode;
                if (config_.batch_size > 0) {
                    batch_size_info = std::min(config_.batch_size, config_.ntree);
                    batch_mode = "manual";
                } else {
                    // Auto batch size: call actual function to get correct value
                    try {
                        batch_size_info = rf::cuda::get_recommended_batch_size(config_.ntree);
                        batch_size_info = std::min(batch_size_info, config_.ntree);
                    } catch (...) {
                        batch_size_info = std::min(static_cast<rf::integer_t>(100), config_.ntree);
                    }
                    batch_mode = "auto";
                }
                builtins.attr("print")(py::str("GPU batch_size={} ({})").format(batch_size_info, batch_mode));
            }
            
            // Print memory information based on execution mode (CUDA-safe)
            try {
                if (config_.use_gpu) {
                    // GPU mode - print GPU memory safely
                    py::module_ rf_module = py::module_::import("RFX");
                    rf_module.attr("print_gpu_memory_status")();
                } else {
                    // CPU mode - print CPU memory
                    try {
                        py::module_ psutil_module = py::module_::import("psutil");
                        py::object memory = psutil_module.attr("virtual_memory")();
                        double total_gb = memory.attr("total").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double available_gb = memory.attr("available").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double used_gb = memory.attr("used").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double percent = memory.attr("percent").cast<double>();
                        
                        // Use Python print for proper formatting
                        py::module_ builtins = py::module_::import("builtins");
                        builtins.attr("print")("\n💻 CPU MEMORY INFORMATION");
                        builtins.attr("print")("==================================================");
                        builtins.attr("print")("📊 System Memory:");
                        builtins.attr("print")(py::str("   Total: {:.1f} GB").format(total_gb));
                        builtins.attr("print")(py::str("   Available: {:.1f} GB").format(available_gb));
                        builtins.attr("print")(py::str("   Used: {:.1f} GB").format(used_gb));
                        builtins.attr("print")(py::str("   Usage: {:.1f}%").format(percent));
                        builtins.attr("print")();
                    } catch (...) {
                        // psutil not available, skip CPU memory info
                    }
                }
            } catch (...) {
                // Ignore errors - don't crash if memory info unavailable
            }
            
            if (show_progress_) {
                // Set up tqdm-style progress bar using Python callback for both CPU and GPU mode
                // This is safe for Jupyter notebooks with proper throttling
                try {
                        // Import regular tqdm (not tqdm.auto) to avoid notebook widget cleanup issues
                        py::object tqdm_module;
                        try {
                            // Use regular tqdm (works in both terminal and Jupyter, avoids widget cleanup issues)
                            tqdm_module = py::module_::import("tqdm");
                        } catch (...) {
                            // tqdm not available, disable progress silently
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                        // Get total number of trees from config
                        rf::integer_t total_trees = config_.ntree;
                        
                        // Use regular tqdm (not tqdm.auto) to avoid notebook widget issues
                        // tqdm.auto can cause cleanup problems in Jupyter
                        try {
                            // Try to use regular tqdm (works in both terminal and Jupyter)
                            // Use iterable with total explicitly set for proper formatting
                            // Create a range object to iterate over (but we'll update manually)
                            py::object range_obj = py::module_::import("builtins").attr("range")(py::cast(total_trees));
                            progress_bar_ = tqdm_module.attr("tqdm")(
                                range_obj,
                                py::arg("total") = py::cast(total_trees),  // Explicitly set total
                                py::arg("desc") = progress_desc_,
                                py::arg("unit") = "tree",
                                py::arg("unit_scale") = false,  // Don't scale units (keep "tree" not "Ktree")
                                py::arg("leave") = false,
                                py::arg("dynamic_ncols") = true,
                                py::arg("miniters") = 1,
                                py::arg("disable") = false,
                                py::arg("bar_format") = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
                            );
                        } catch (...) {
                            // If tqdm creation fails, disable progress
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                            // Force initial display of progress bar
                            try {
                                progress_bar_.attr("refresh")();
                            } catch (...) {
                                // Ignore refresh errors
                            }
                            
                            // Throttle progress updates to avoid Jupyter IOPub rate limit
                            // Reset refresh position tracker for this training session
                            last_progress_refresh_pos_ = 0;
                            int refresh_interval = config_.batch_size > 0 ? config_.batch_size : 5; // Refresh every batch
                            
                            // Ultra-safe callback - minimal Python interaction, no refresh unless critical
                            rf_->set_progress_callback([this, refresh_interval](rf::integer_t current, rf::integer_t total) {
                                // Wrap everything in try-catch to prevent any crashes
                                try {
                                    // Check if progress bar exists and is valid
                                    if (progress_bar_.is_none()) {
                                        return;  // Progress bar cleared, skip
                                    }
                                    
                                    // Very minimal update - just update the counter, no refresh
                                    if (current > last_progress_refresh_pos_) {
                                        try {
                                            // Get current position safely
                                            int current_pos = progress_bar_.attr("n").cast<int>();
                                            if (current > current_pos) {
                                                int update_amount = current - current_pos;
                                                progress_bar_.attr("update")(py::cast(update_amount));
                                            }
                                        } catch (...) {
                                            // Ignore update errors - don't crash
                                        }
                                        
                                        // Only refresh very infrequently to avoid Jupyter crashes
                                        if (current >= total || (current - last_progress_refresh_pos_ >= refresh_interval)) {
                                            try {
                                                progress_bar_.attr("refresh")();
                                                last_progress_refresh_pos_ = current;
                                            } catch (...) {
                                                // Ignore refresh errors - critical to not crash here
                                            }
                                        }
                                    }
                                    
                                    // Handle completion
                                    if (current >= total) {
                                        try {
                                            // Final update
                                            int final_pos = progress_bar_.attr("n").cast<int>();
                                            if (final_pos < total) {
                                                progress_bar_.attr("update")(py::cast(total - final_pos));
                                            }
                                            // Close and clear
                                            if (py::hasattr(progress_bar_, "close")) {
                                                progress_bar_.attr("close")();
                                            }
                                            progress_bar_ = py::none();
                                        } catch (...) {
                                            // Ignore all errors during cleanup
                                            progress_bar_ = py::none();
                                        }
                                    }
                                } catch (...) {
                                    // Ultimate safety net - catch absolutely everything
                                    // Don't do anything, just return
                                }
                            });
                        }
                    }
                } catch (...) {
                    // If tqdm fails, disable progress silently
                    show_progress_ = false;
                }
            }
            
            // Print auto-selected batch size if using GPU with auto-scaling (batch_size=0)
            if (config_.use_gpu && config_.batch_size == 0 && config_.ntree >= 10) {
                try {
                    rf::integer_t recommended_batch = rf::cuda::get_recommended_batch_size(config_.ntree);
                    recommended_batch = std::min(recommended_batch, config_.ntree);
                    py::module_ builtins = py::module_::import("builtins");
                    builtins.attr("print")(py::str("🔧 Auto-scaling: Selected batch size = {} trees (out of {} total)").format(recommended_batch, config_.ntree));
                } catch (...) {
                    // Ignore errors - the C++ code will print it anyway via std::cout
                }
            }
            
            // Call the fit method with actual data
            rf_->fit(static_cast<rf::real_t*>(X_buf.ptr), 
                     y_buf.ptr, 
                     sample_weight_buf.size > 0 ? static_cast<rf::real_t*>(sample_weight_buf.ptr) : nullptr);
            
            // Print GPU memory status after training completes (if GPU was used)
            // Query GPU memory directly using CUDA API (safe after training completes)
            try {
                if (config_.use_gpu) {
                    // GPU mode - query GPU memory directly using CUDA API
                    // This is safe because training just completed, so CUDA context is active
                    try {
#ifdef CUDA_FOUND
                        size_t free_mem, total_mem;
                        ::cudaError_t err = ::cudaMemGetInfo(&free_mem, &total_mem);
                        
                        if (err == ::cudaSuccess) {
                            // Successfully queried GPU memory - print it in GB format to match CPU format
                            size_t used_mem = total_mem - free_mem;
                            double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
                            double total_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
                            double used_gb = used_mem / (1024.0 * 1024.0 * 1024.0);
                            double usage_percent = (used_mem / (double)total_mem) * 100.0;
                            
                            py::module_ builtins = py::module_::import("builtins");
                            builtins.attr("print")("\n🚀 GPU MEMORY STATUS (After Training):");
                            builtins.attr("print")("==================================================");
                            builtins.attr("print")(py::str("📊 GPU Memory:"));
                            builtins.attr("print")(py::str("   Total: {:.1f} GB").format(total_gb));
                            builtins.attr("print")(py::str("   Available: {:.1f} GB").format(free_gb));
                            builtins.attr("print")(py::str("   Used: {:.1f} GB").format(used_gb));
                            builtins.attr("print")(py::str("   Usage: {:.1f}%").format(usage_percent));
                        }
                        // If cudaMemGetInfo fails, just silently skip (don't print error messages)
#endif
                    } catch (...) {
                        // Ignore errors - don't crash if memory query fails
                    }
                }
            } catch (...) {
                // Ignore all errors - don't crash if memory info unavailable
            }
            
            // Close progress bar immediately after training completes to prevent GC issues
            if (show_progress_ && !progress_bar_.is_none()) {
                try {
                    // Progress bar should already be closed by callback, but ensure it's cleared
                    progress_bar_ = py::none();
                } catch (...) {
                    // Ignore any errors
                    progress_bar_ = py::none();
                }
            }
        }
        
        // ====================================================================
        // SPARSE FIT - accepts scipy.sparse.csr_matrix
        // ====================================================================
        void fit_sparse(py::object X_sparse, py::array y, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // Convert to CSR if not already
            py::object X_csr = ensure_csr(X_sparse);
            
            // Convert scipy CSR to RFX SparseMatrixCSR
            rf::SparseMatrixCSR sparse_matrix = scipy_csr_to_rfx(X_csr);
            
            if (!sparse_matrix.is_valid()) {
                throw std::runtime_error("Invalid sparse matrix structure");
            }
            
            // Get dimensions from sparse matrix
            rf::integer_t actual_nsample = sparse_matrix.nrows;
            rf::integer_t actual_mdim = sparse_matrix.ncols;
            
            // Determine nclass from y
            auto y_buf = y.request();
            rf::integer_t actual_nclass = config_.nclass;
            std::set<rf::integer_t> unique_classes;
            
            if (y_buf.format == "i" || y_buf.format == "l") {
                const int32_t* y_ptr = static_cast<const int32_t*>(y_buf.ptr);
                for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                    unique_classes.insert(static_cast<rf::integer_t>(y_ptr[i]));
                }
            } else if (y_buf.format == "f") {
                const float* y_ptr = static_cast<const float*>(y_buf.ptr);
                for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                    unique_classes.insert(static_cast<rf::integer_t>(y_ptr[i]));
                }
            }
            actual_nclass = std::max(actual_nclass, static_cast<rf::integer_t>(unique_classes.size()));
            
            // Update config with sparse dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            config_.nclass = actual_nclass;
            config_.maxnode = rf::calculate_maxnode(actual_nsample, config_.minndsize, config_.maxnode);
            config_.use_sparse = true;  // Ensure sparse mode is set

            // Recalculate mtry if not set
            if (config_.mtry <= 0 || config_.mtry > actual_mdim) {
                config_.mtry = std::max(1, static_cast<rf::integer_t>(std::sqrt(static_cast<float>(actual_mdim))));
            }
            
            // Always recreate RandomForest for sparse to ensure correct config
            if (rf_ != nullptr) {
                delete rf_;
                rf_ = nullptr;
            }
                rf_ = new rf::RandomForest(config_);
            
            // Convert y to integers
            std::vector<rf::integer_t> y_int(actual_nsample);
            if (y_buf.format == "f") {
                const float* y_ptr = static_cast<const float*>(y_buf.ptr);
                for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                    y_int[i] = static_cast<rf::integer_t>(y_ptr[i]);
                }
            } else {
                const int32_t* y_ptr = static_cast<const int32_t*>(y_buf.ptr);
                for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                    y_int[i] = static_cast<rf::integer_t>(y_ptr[i]);
                }
            }
            
            // Get sample weights
            const rf::real_t* weight_ptr = nullptr;
            if (sample_weight.size() > 0) {
                auto w_buf = sample_weight.request();
                weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
            }
            
            // Call sparse fit
            rf_->fit_classification_sparse(sparse_matrix, y_int.data(), weight_ptr);
        }
        
        // Delegate all other methods to the underlying RandomForest
        py::array_t<rf::integer_t> predict(py::array_t<rf::real_t> X) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict()");
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            py::array_t<rf::integer_t> predictions(X_buf.shape[0]);
            rf_->predict(static_cast<rf::real_t*>(X_buf.ptr), X_buf.shape[0], predictions.mutable_data());
            return predictions;
        }
        
        py::array_t<rf::real_t> predict_proba(py::array_t<rf::real_t> X) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict_proba()");
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            py::array_t<rf::real_t> probabilities({X_buf.shape[0], static_cast<py::ssize_t>(rf_->get_n_classes())});
            rf_->predict_proba(static_cast<rf::real_t*>(X_buf.ptr), X_buf.shape[0], probabilities.mutable_data());
            return probabilities;
        }
        
        // Get prototypes - most representative samples for each class (Breiman & Cutler's concept)
        // For classification: returns per-class prototypes (samples most similar to their class)
        // Returns dict: {class_id: [(sample_idx, prototype_score), ...]}
        py::dict get_prototypes(rf::integer_t n_prototypes = 3) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_prototypes()");
            
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t nclass = rf_->get_n_classes();
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            const rf::integer_t* cl = rf_->get_class_labels();
            
            if (!proximity_ptr) throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True");
            if (!cl) throw std::runtime_error("Class labels not available");
            
            py::dict result;
            
            for (rf::integer_t c = 0; c < nclass; ++c) {
                std::vector<rf::integer_t> class_indices;
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (cl[i] == c) class_indices.push_back(i);
                }
                if (class_indices.empty()) continue;
                
                std::vector<std::pair<rf::real_t, rf::integer_t>> scores;
                for (rf::integer_t idx : class_indices) {
                    rf::real_t sum = 0.0f;
                    for (rf::integer_t jdx : class_indices) {
                        if (idx != jdx) sum += static_cast<rf::real_t>(proximity_ptr[idx * nsamples + jdx]);
                    }
                    rf::real_t mean_prox = (class_indices.size() > 1) ? sum / (class_indices.size() - 1) : 0.0f;
                    scores.push_back({mean_prox, idx});
                }
                
                rf::integer_t n_proto = std::min(n_prototypes, static_cast<rf::integer_t>(scores.size()));
                std::partial_sort(scores.begin(), scores.begin() + n_proto, scores.end(),
                                 [](const auto& a, const auto& b) { return a.first > b.first; });
                
                py::list proto_list;
                for (rf::integer_t i = 0; i < n_proto; ++i) {
                    proto_list.append(py::make_tuple(scores[i].second, scores[i].first));
                }
                result[py::int_(c)] = proto_list;
            }
            return result;
        }
        
        rf::real_t get_oob_error() const { return rf_->get_oob_error(); }
        
        // Save model to file
        void save(const std::string& filepath) const {
            if (rf_ == nullptr) {
                throw std::runtime_error("Model must be fitted before saving");
            }
            rf_->save(filepath);
        }
        
        py::array_t<rf::real_t> feature_importances_() {
            rf::integer_t mdim = rf_->get_n_features();
            py::array_t<rf::real_t> importances(mdim);
            const rf::real_t* imp_ptr = rf_->get_feature_importances();
            rf::integer_t imp_size = rf_->get_feature_importances_size();
            
            // Check if pointer is valid before copying
            // If compute_importance=False, feature_importances_ may be empty
            if (imp_ptr != nullptr && imp_size == mdim) {
                std::copy(imp_ptr, imp_ptr + mdim, importances.mutable_data());
            } else {
                // If importance was not computed or size mismatch, return zeros
                std::fill(importances.mutable_data(), importances.mutable_data() + mdim, 0.0f);
            }
            return importances;
        }
        
        rf::TaskType get_task_type() const { return rf_->get_task_type(); }
        rf::integer_t get_n_samples() const { return rf_->get_n_samples(); }
        rf::integer_t get_n_features() const { return rf_->get_n_features(); }
        rf::integer_t get_n_classes() const { return rf_->get_n_classes(); }
        
        // Add local importance and proximity matrix methods
        py::array_t<rf::real_t> get_local_importance() {
            // Get the local importance pointer from C++ and convert to numpy array
            const rf::real_t* local_imp_ptr = rf_->get_qimpm();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (local_imp_ptr) {
                // Copy from C++ array to Python array
                std::copy(local_imp_ptr, local_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Fallback: initialize with zeros if not computed
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get local proximity importance (per-sample, per-feature)
        // Shape: (n_samples, n_features)
        // Value at [i, k] = % of trees where permuting feature k changed terminal node for OOB sample i
        py::array_t<rf::real_t> get_local_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (prox_imp_ptr) {
                std::copy(prox_imp_ptr, prox_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Not computed - return zeros
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get overall proximity importance (aggregated across samples)
        // Shape: (n_features,)
        // Value at [k] = mean % of trees where permuting feature k changed terminal node
        py::array_t<rf::real_t> get_overall_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result(mdim);
            auto buf = result.request();
            rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
            
            if (prox_imp_ptr && nsamples > 0) {
                for (rf::integer_t k = 0; k < mdim; ++k) {
                    rf::real_t sum = 0.0f;
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        sum += prox_imp_ptr[i * mdim + k];
                    }
                    dest[k] = sum / static_cast<rf::real_t>(nsamples);
                }
            } else {
                std::fill(dest, dest + mdim, 0.0f);
            }
            return result;
        }
        
        // Legacy alias for backward compatibility
        py::array_t<rf::real_t> get_proximity_importance() {
            return get_local_proximity_importance();
        }
        
        // Get top-K similar samples with feature explanations
        // Uses leaf assignments ONLY for similarity (Breiman RF proximity)
        // Uses proximity importance for explanations (which features affect similarity)
        py::tuple get_top_k_similar_with_explanations(
            rf::integer_t sample_idx, rf::integer_t k = 5, rf::integer_t n_explanations = 3
        ) {
            if (!rf_) throw std::runtime_error("Model must be fitted first");
            
            // REQUIRE leaf assignments - no fallback to proximity matrix
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error(
                    "Leaf assignments required. Set compute_leaf_assignments=True when creating the model. "
                    "This method uses exact RF proximity from leaf co-occurrence."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            rf::integer_t ntree = rf_->get_n_trees();
            if (sample_idx < 0 || sample_idx >= nsamples) throw std::runtime_error("sample_idx out of range");
            k = std::min(k, nsamples - 1);
            n_explanations = std::min(n_explanations, mdim);
            
            // Compute similarity from leaf assignments (exact RF proximity)
            std::vector<rf::real_t> similarities_raw(nsamples, 0.0f);
            for (rf::integer_t t = 0; t < ntree; ++t) {
                int16_t query_leaf = leaf_ptr[t * nsamples + sample_idx];
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (i != sample_idx && leaf_ptr[t * nsamples + i] == query_leaf) {
                        similarities_raw[i] += 1.0f;
                    }
                }
            }
            // Normalize by ntree (fraction of trees where samples co-occur)
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                similarities_raw[i] /= static_cast<rf::real_t>(ntree);
            }
            
            // Create normalized version (max similarity = 1.0)
            // This ensures comparable results between dense and sparse modes
            std::vector<rf::real_t> similarities_norm = similarities_raw;
            rf::real_t max_sim = 0.0f;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                if (i != sample_idx && similarities_raw[i] > max_sim) {
                    max_sim = similarities_raw[i];
                }
            }
            if (max_sim > 0.0f) {
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    similarities_norm[i] /= max_sim;
                }
            }
            
            // Get top-K by similarity (use normalized for ranking)
            std::vector<std::tuple<rf::real_t, rf::real_t, rf::integer_t>> sims;
            for (rf::integer_t j = 0; j < nsamples; ++j) {
                if (j != sample_idx) {
                    sims.push_back({similarities_norm[j], similarities_raw[j], j});
                }
            }
            std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), 
                [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
            
            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> sim_raw_arr(k);
            py::array_t<rf::real_t> sim_norm_arr(k);
            auto idx_buf = idx_arr.request();
            auto sim_raw_buf = sim_raw_arr.request();
            auto sim_norm_buf = sim_norm_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = std::get<2>(sims[i]);
                static_cast<rf::real_t*>(sim_raw_buf.ptr)[i] = std::get<1>(sims[i]);
                static_cast<rf::real_t*>(sim_norm_buf.ptr)[i] = std::get<0>(sims[i]);
            }
            
            // Get explanations from proximity importance (Breiman/Cutler: which features affect similarity)
            const rf::real_t* pi = rf_->get_proximity_importance();
            py::array_t<rf::integer_t> feat_arr(n_explanations); py::array_t<rf::real_t> val_arr(n_explanations);
            auto feat_buf = feat_arr.request(); auto val_buf = val_arr.request();
            if (pi) {
                std::vector<std::pair<rf::real_t, rf::integer_t>> fi;
                for (rf::integer_t f = 0; f < mdim; ++f) fi.push_back({pi[sample_idx * mdim + f], f});
                std::partial_sort(fi.begin(), fi.begin() + n_explanations, fi.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (rf::integer_t i = 0; i < n_explanations; ++i) {
                    static_cast<rf::integer_t*>(feat_buf.ptr)[i] = fi[i].second;
                    static_cast<rf::real_t*>(val_buf.ptr)[i] = fi[i].first;
                }
            } else { 
                std::fill_n(static_cast<rf::integer_t*>(feat_buf.ptr), n_explanations, 0); 
                std::fill_n(static_cast<rf::real_t*>(val_buf.ptr), n_explanations, 0.0f); 
            }
            // Return: (indices, raw_similarities, normalized_similarities, feature_indices, feature_scores)
            return py::make_tuple(idx_arr, sim_raw_arr, sim_norm_arr, feat_arr, val_arr);
        }
        
        py::array_t<rf::real_t> get_proximity_matrix() {
            // Get the proximity matrix pointer and convert to numpy array
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            rf::integer_t nsamples = rf_->get_n_samples();
            
            py::array_t<rf::real_t> result({nsamples, nsamples});
            auto buf = result.request();
            
            if (proximity_ptr) {
                // Convert from dp_t to real_t
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                // Use element-by-element copy with bounds checking to avoid memory corruption
                rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
                const rf::dp_t* src = proximity_ptr;
                size_t n_elements = static_cast<size_t>(nsamples) * nsamples;
                
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                for (size_t i = 0; i < n_elements; ++i) {
                    dest[i] = static_cast<rf::real_t>(src[i]);
                }
            } else {
                // Low-rank mode: full matrix not available
                if (rf_->get_compute_proximity()) {
                    std::string error_msg = 
                        "Proximity matrix not available in full form. "
                        "Low-rank mode is active (use_qlora=True). "
                        "Full matrix reconstruction would require O(n²) memory and can CRASH your system! "
                        "Example: 100k samples = ~80 GB (likely to crash!). "
                        "Proximity is stored in low-rank factors (A and B). "
                        "Use get_lowrank_factors() or compute_mds_3d_from_factors() instead. "
                        "Or disable use_qlora=True for smaller datasets that can fit full matrix.";
                    
                    if (nsamples > 50000) {
                        error_msg += " ERROR: Dataset too large (" + std::to_string(nsamples) + 
                                    " samples) - reconstruction aborted to prevent system crash.";
                    }
                    
                    throw std::runtime_error(error_msg);
                } else {
                    // Fallback: initialize with identity matrix if not computed
                    std::fill(static_cast<rf::real_t*>(buf.ptr), 
                             static_cast<rf::real_t*>(buf.ptr) + nsamples * nsamples, 0.0f);
                    // Set diagonal to 1.0
                    for (rf::integer_t i = 0; i < nsamples; i++) {
                        static_cast<rf::real_t*>(buf.ptr)[i * nsamples + i] = 1.0f;
                    }
                }
            }
            
            return result;
        }
        
        py::tuple get_lowrank_factors() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_lowrank_factors()");
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::dp_t* A_host = nullptr;
            rf::dp_t* B_host = nullptr;
            rf::integer_t rank = 0;
            
            bool success = rf_->get_lowrank_factors(&A_host, &B_host, &rank);
            
            if (!success) {
                throw std::runtime_error(
                    "Low-rank factors not available. "
                    "Either low-rank mode is not active (use_qlora=True required), "
                    "or factors have not been computed yet."
                );
            }
            
            // Create numpy arrays from host memory
            py::array_t<rf::dp_t> A_array(std::vector<py::ssize_t>{nsamples, rank});
            py::array_t<rf::dp_t> B_array(std::vector<py::ssize_t>{nsamples, rank});
            
            auto A_buf = A_array.request();
            auto B_buf = B_array.request();
            
            if (!A_buf.ptr || !B_host) {
                delete[] A_host;
                delete[] B_host;
                throw std::runtime_error("Failed to allocate memory for low-rank factors");
            }
            
            // Copy data
            size_t factor_size = static_cast<size_t>(nsamples) * rank;
            std::copy(A_host, A_host + factor_size, static_cast<rf::dp_t*>(A_buf.ptr));
            std::copy(B_host, B_host + factor_size, static_cast<rf::dp_t*>(B_buf.ptr));
            
            // Free host memory
            delete[] A_host;
            delete[] B_host;
            
            return py::make_tuple(A_array, B_array, rank);
        }
        
        py::array_t<rf::integer_t> get_oob_counts() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_oob_counts()");
            rf::integer_t nsamples = rf_->get_n_samples();
            const rf::integer_t* nout_ptr = rf_->get_nout_ptr();
            
            if (!nout_ptr) {
                throw std::runtime_error("OOB counts not available.");
            }
            
            py::array_t<rf::integer_t> result(nsamples);
            auto result_buf = result.request();
            rf::integer_t* result_ptr = static_cast<rf::integer_t*>(result_buf.ptr);
            std::copy(nout_ptr, nout_ptr + nsamples, result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_3d_cpu() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_3d_cpu()");
            
            // Get proximity matrix
            py::array_t<rf::dp_t> prox_array = get_proximity_matrix();
            if (prox_array.size() == 0) {
                throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True when fitting.");
            }
            
            auto prox_buf = prox_array.request();
            rf::dp_t* prox_ptr = static_cast<rf::dp_t*>(prox_buf.ptr);
            rf::integer_t n_samples = static_cast<rf::integer_t>(std::sqrt(prox_array.size()));
            
            // Call C++ CPU MDS implementation (uses LAPACK)
            // Pass OOB counts for RF-GAP normalization only if RF-GAP is actually enabled
            // Note: If RF-GAP was computed as full matrix, it's already normalized by |Si| in cpu_proximity_rfgap
            // But if proximity matrix was reconstructed from low-rank factors (QLoRA), normalization is required here
            // Get OOB counts for RF-GAP normalization if available
            const rf::integer_t* oob_counts = rf_->get_nout_ptr();
            // Only normalize by OOB counts if RF-GAP is actually enabled
            // Standard proximity matrices are already normalized and don't need RF-GAP normalization
            bool use_rfgap = rf_->get_use_rfgap();
            std::vector<double> coords_3d = rf::compute_mds_3d_cpu(prox_ptr, n_samples, true, oob_counts, use_rfgap);
            
            // Convert to numpy array
            py::array_t<double> result(std::vector<py::ssize_t>{n_samples, 3});
            auto result_buf = result.request();
            double* result_ptr = static_cast<double*>(result_buf.ptr);
            std::copy(coords_3d.begin(), coords_3d.end(), result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_from_factors(rf::integer_t k = 3) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_from_factors()");
            
            if (k < 1) {
                throw std::runtime_error("MDS dimension k must be >= 1");
            }
            
            std::vector<double> coords = rf_->compute_mds_from_factors(k);
            
            if (coords.empty()) {
                throw std::runtime_error(
                    "MDS computation failed. "
                    "Low-rank factors may not be available (use_qlora=True required), "
                    "or computation failed."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            py::array_t<double> coords_array(std::vector<py::ssize_t>{nsamples, static_cast<py::ssize_t>(k)});
            auto coords_buf = coords_array.request();
            
            std::copy(coords.begin(), coords.end(), static_cast<double*>(coords_buf.ptr));
            
            // Check for duplicate coordinates (indicates insufficient tree coverage)
            py::module_ np = py::module_::import("numpy");
            py::module_ warnings = py::module_::import("warnings");
            py::array_t<double> rounded = np.attr("round")(coords_array, 6);
            py::array_t<double> unique_coords = np.attr("unique")(rounded, py::arg("axis")=0);
            py::ssize_t n_unique = unique_coords.request().shape[0];
            py::ssize_t n_duplicates = nsamples - n_unique;
            if (n_duplicates > 0) {
                double pct_duplicates = (static_cast<double>(n_duplicates) / nsamples) * 100.0;
                
                // Calculate recommended tree count based on sample size
                int recommended_trees;
                std::string size_category;
                if (nsamples < 500) {
                    recommended_trees = nsamples * 8;
                    size_category = "8-10× samples";
                } else if (nsamples < 2000) {
                    recommended_trees = nsamples * 6;
                    size_category = "6-8× samples";
                } else if (nsamples < 10000) {
                    recommended_trees = nsamples * 5;
                    size_category = "5-7× samples";
                } else {
                    recommended_trees = nsamples * 4;
                    size_category = "3-5× samples";
                }
                
                std::string warn_msg = "MDS coordinates have " + std::to_string(n_duplicates) + 
                    " duplicate points (" + std::to_string(static_cast<int>(pct_duplicates)) + "% of " + 
                    std::to_string(nsamples) + " samples). Only " + std::to_string(n_unique) + 
                    " unique positions will be visible. This typically indicates insufficient tree coverage " +
                    "for proximity computation. Recommend ntree ≥ " + std::to_string(recommended_trees) + 
                    " (" + size_category + ") for complete MDS coverage.";
                warnings.attr("warn")(warn_msg, py::module_::import("builtins").attr("UserWarning"));
            }
            
            return coords_array;
        }
        
        py::array_t<double> compute_mds_3d_from_factors() {
            return compute_mds_from_factors(3);
        }
        
        py::array_t<rf::real_t> get_oob_predictions() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_oob_predictions()");
            rf::integer_t nsample = rf_->get_n_samples();
            py::array_t<rf::real_t> oob_predictions(nsample);
            auto oob_buf = oob_predictions.request();
            if (rf_->get_task_type() == rf::TaskType::CLASSIFICATION) {
                const rf::integer_t* oob_class_ptr = rf_->get_oob_class_predictions();
                if (oob_class_ptr) {
                    for (rf::integer_t i = 0; i < nsample; i++) {
                        static_cast<rf::real_t*>(oob_buf.ptr)[i] = static_cast<rf::real_t>(oob_class_ptr[i]);
                    }
                } else {
                    std::fill(static_cast<rf::real_t*>(oob_buf.ptr),
                             static_cast<rf::real_t*>(oob_buf.ptr) + nsample, 0.0);
                }
            } else if (rf_->get_task_type() == rf::TaskType::REGRESSION) {
                const rf::real_t* oob_ptr = rf_->get_oob_predictions();
                if (oob_ptr) {
                    std::copy(oob_ptr, oob_ptr + nsample, static_cast<rf::real_t*>(oob_buf.ptr));
                } else {
                    std::fill(static_cast<rf::real_t*>(oob_buf.ptr),
                             static_cast<rf::real_t*>(oob_buf.ptr) + nsample, 0.0);
                }
            } else {
                std::fill(static_cast<rf::real_t*>(oob_buf.ptr),
                         static_cast<rf::real_t*>(oob_buf.ptr) + nsample, 0.0);
            }
            return oob_predictions;
        }
        
        // Get leaf assignments (terminal node IDs) for all samples in all trees
        // Returns 2D array of shape (ntree, nsample) with terminal node IDs
        py::array_t<int16_t> get_leaf_assignments() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_leaf_assignments()");
            
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            rf::integer_t ntree = rf_->get_n_trees();
            rf::integer_t nsample = rf_->get_n_samples();
            
            // Create numpy array of shape (ntree, nsample)
            py::array_t<int16_t> result({ntree, nsample});
            auto buf = result.request();
            int16_t* result_ptr = static_cast<int16_t*>(buf.ptr);
            
            // Copy data (stored as [tree * nsample + sample])
            std::copy(leaf_ptr, leaf_ptr + ntree * nsample, result_ptr);
            
            return result;
        }
        
        // Get top-K similar samples using leaf assignments (exact proximity)
        // Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity
        // Returns tuple of (indices, similarity_scores)
        py::tuple get_top_k_similar(rf::integer_t query_idx, int k = 10, bool exclude_self = true) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_top_k_similar()");
            
            // Lazy build inverted index on first call for optimal performance
            if (!rf_->has_inverted_leaf_index() && rf_->has_leaf_assignments()) {
                rf_->build_inverted_leaf_index();
            }
            
            auto results = rf_->get_top_k_similar(query_idx, k, exclude_self);
            
            if (results.empty()) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            // Create output arrays
            rf::integer_t actual_k = static_cast<rf::integer_t>(results.size());
            py::array_t<rf::integer_t> top_k_indices(actual_k);
            py::array_t<rf::real_t> top_k_scores(actual_k);
            auto idx_buf = top_k_indices.request();
            auto score_buf = top_k_scores.request();
            
            for (rf::integer_t i = 0; i < actual_k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = results[i].first;
                static_cast<rf::real_t*>(score_buf.ptr)[i] = results[i].second;
            }
            
            return py::make_tuple(top_k_indices, top_k_scores);
        }
        
        // Build inverted leaf index for fast top-K lookups (usually called automatically)
        void build_inverted_leaf_index() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling build_inverted_leaf_index()");
            rf_->build_inverted_leaf_index();
        }
        
        // Check if inverted leaf index is built
        bool has_inverted_leaf_index() const {
            if (!rf_) return false;
            return rf_->has_inverted_leaf_index();
        }
        
        // ====================================================================
        // PREDICT METHODS FOR NEW DATA (inference-time)
        // ====================================================================
        
        // Predict leaf assignments for new samples (dense input)
        py::array_t<int16_t> predict_leaf_assignments(py::array_t<rf::real_t> X_new) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict_leaf_assignments()");
            
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X_new);
            auto X_buf = X_contig.request();
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional");
            }
            
            rf::integer_t nsamples_new = X_buf.shape[0];
            rf::integer_t ntree = rf_->get_n_trees();
            
            // Create output array (nsamples_new, ntree)
            py::array_t<int16_t> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nsamples_new), static_cast<py::ssize_t>(ntree)});
            auto result_buf = result.request();
            
            rf_->predict_leaf_assignments(
                static_cast<rf::real_t*>(X_buf.ptr),
                nsamples_new,
                static_cast<int16_t*>(result_buf.ptr)
            );
            
            return result;
        }
        
        // Predict leaf assignments for new sparse samples
        py::array_t<int16_t> predict_leaf_assignments_sparse(py::object X_sparse) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict_leaf_assignments_sparse()");
            
            // Convert scipy CSR to RFX SparseMatrixCSR
            py::object X_csr = ensure_csr(X_sparse);
            rf::SparseMatrixCSR sparse_matrix = scipy_csr_to_rfx(X_csr);
            
            if (!sparse_matrix.is_valid()) {
                throw std::runtime_error("Invalid sparse matrix structure");
            }
            
            rf::integer_t nsamples_new = sparse_matrix.nrows;
            rf::integer_t ntree = rf_->get_n_trees();
            
            // Create output array (nsamples_new, ntree)
            py::array_t<int16_t> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nsamples_new), static_cast<py::ssize_t>(ntree)});
            auto result_buf = result.request();
            
            rf_->predict_leaf_assignments_sparse(sparse_matrix, static_cast<int16_t*>(result_buf.ptr));
            
            return result;
        }
        
        // Predict top-K similar training samples for new data
        py::tuple predict_top_k_similar(py::array_t<rf::real_t> X_new, int k = 10) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict_top_k_similar()");
            
            // Ensure inverted index is built for fast lookups
            if (!rf_->has_inverted_leaf_index() && rf_->has_leaf_assignments()) {
                rf_->build_inverted_leaf_index();
            }
            
            const int16_t* train_leaf_ptr = rf_->get_leaf_assignments();
            if (!train_leaf_ptr) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X_new);
            auto X_buf = X_contig.request();
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional");
            }
            
            rf::integer_t nsamples_new = X_buf.shape[0];
            rf::integer_t ntree = rf_->get_n_trees();
            rf::integer_t nsample_train = rf_->get_n_samples();
            
            // Get leaf assignments for new samples
            std::vector<int16_t> new_leaves(nsamples_new * ntree);
            rf_->predict_leaf_assignments(
                static_cast<rf::real_t*>(X_buf.ptr),
                nsamples_new,
                new_leaves.data()
            );
            
            // Output: (nsamples_new, k) for both indices and scores
            py::array_t<rf::integer_t> top_k_indices(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nsamples_new), static_cast<py::ssize_t>(k)});
            py::array_t<rf::real_t> top_k_scores(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nsamples_new), static_cast<py::ssize_t>(k)});
            auto idx_buf = top_k_indices.request();
            auto score_buf = top_k_scores.request();
            
            for (rf::integer_t q = 0; q < nsamples_new; ++q) {
                // Compute similarity to all training samples
                std::vector<rf::real_t> similarities(nsample_train, 0.0f);
                
                for (rf::integer_t t = 0; t < ntree; ++t) {
                    int16_t query_leaf = new_leaves[q * ntree + t];
                    for (rf::integer_t i = 0; i < nsample_train; ++i) {
                        if (train_leaf_ptr[t * nsample_train + i] == query_leaf) {
                            similarities[i] += 1.0f;
                        }
                    }
                }
                
                // Normalize
                for (rf::integer_t i = 0; i < nsample_train; ++i) {
                    similarities[i] /= static_cast<rf::real_t>(ntree);
                }
                
                // Get top-K
                std::vector<rf::integer_t> indices(nsample_train);
                std::iota(indices.begin(), indices.end(), 0);
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                    [&similarities](rf::integer_t a, rf::integer_t b) {
                        return similarities[a] > similarities[b];
                    });
                
                // Store results
                for (int i = 0; i < k; ++i) {
                    static_cast<rf::integer_t*>(idx_buf.ptr)[q * k + i] = indices[i];
                    static_cast<rf::real_t*>(score_buf.ptr)[q * k + i] = similarities[indices[i]];
                }
            }
            
            return py::make_tuple(top_k_indices, top_k_scores);
        }
    };

    // Register RandomForestClassifier class
    py::class_<RandomForestClassifier>(m, "RandomForestClassifier")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, int, bool, bool, bool, 
                      bool, bool, std::string, bool, float, int, int, float, bool, std::string, std::string, int, int, int, bool, std::string, int, bool, bool, bool, py::object, bool, int>(),
    py::arg("nsample") = 1000,
    py::arg("ntree") = 100,
    py::arg("mdim") = 10,
    py::arg("nclass") = 2,
    py::arg("maxcat") = 10,
    py::arg("mtry") = 0,
    py::arg("maxnode") = 0,
    py::arg("minndsize") = 1,
    py::arg("ncsplit") = 25,
    py::arg("ncmax") = 25,
    py::arg("iseed") = 12345,
    py::arg("compute_proximity") = false,
    py::arg("compute_importance") = true,
    py::arg("compute_local_importance") = false,
    py::arg("use_gpu") = false,
    py::arg("use_qlora") = false,
    py::arg("quant_mode") = "nf4",
    py::arg("use_sparse") = false,
    py::arg("sparsity_threshold") = 1e-6f,
    py::arg("batch_size") = 0,
    py::arg("nodesize") = 5,
    py::arg("cutoff") = 0.01f,
             py::arg("show_progress") = true,
             py::arg("progress_desc") = "Training Random Forest",
             py::arg("gpu_loss_function") = "gini",  // "gini" for classification
             py::arg("rank") = 32,  // Low-rank proximity matrix rank (32 preserves 99%+ geometry)
             py::arg("power_iterations") = 30,  // Power iteration count for low-rank SVD
             py::arg("n_threads_cpu") = 0,  // Number of CPU threads (0 = auto-detect)
             py::arg("use_rfgap") = false,  // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
             py::arg("importance_method") = "local_imp",  // Local importance method: "local_imp" or "clique"
             py::arg("clique_M") = -1,  // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)
             py::arg("use_casewise") = false,  // Use case-wise calculations (bootstrap frequency weighted) vs non-case-wise (simple averaging)
             py::arg("compute_proximity_importance") = false,  // Measures % of trees where permuting each feature changes terminal node
             py::arg("compute_leaf_assignments") = false,  // Store terminal node per sample per tree for on-demand proximity
             py::arg("categorical_features") = py::none(),  // List of column indices (0-based) for categorical features, or None for auto-detection
             py::arg("use_histogram") = true,  // Use histogram-based split finding (faster for large datasets >= 1000 samples)
             py::arg("n_bins") = 256)  // Number of bins for histogram (max 256)
        // Auto-detecting fit: accepts both dense numpy arrays and scipy sparse matrices
        .def("fit", [](RandomForestClassifier& self, py::object X, py::array y, py::array_t<rf::real_t> sample_weight) {
            if (is_scipy_sparse(X)) {
                // Auto-detected sparse input - enable sparse mode and call fit_sparse
                self.set_use_sparse(true);
                return self.fit_sparse(X, y, sample_weight);
            } else {
                // Dense input - call regular fit
                py::array_t<rf::real_t> X_dense = X.cast<py::array_t<rf::real_t>>();
                return self.fit(X_dense, y, sample_weight);
            }
        }, py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit the random forest model. Auto-detects scipy.sparse matrices and uses sparse kernels automatically.")
        .def("fit_sparse", [](RandomForestClassifier& self, py::object X_sparse, py::array y) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, y, py::array_t<rf::real_t>());
        }, py::arg("X"), py::arg("y"),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix). "
           "For recommender systems with sparse user-item matrices.")
        .def("fit_sparse", [](RandomForestClassifier& self, py::object X_sparse, py::array y, py::array_t<rf::real_t> sample_weight) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, y, sample_weight);
        }, py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix) with sample weights.")
        .def("predict", &RandomForestClassifier::predict)
        .def("predict_proba", &RandomForestClassifier::predict_proba)
        .def("get_oob_error", &RandomForestClassifier::get_oob_error)
        .def("save", &RandomForestClassifier::save, py::arg("filepath"),
             "Save trained model to file (binary format). Can be loaded with rfx.load().")
        .def("feature_importances_", &RandomForestClassifier::feature_importances_)
        .def("get_task_type", &RandomForestClassifier::get_task_type)
        .def("get_n_samples", &RandomForestClassifier::get_n_samples)
        .def("get_n_features", &RandomForestClassifier::get_n_features)
        .def("get_n_classes", &RandomForestClassifier::get_n_classes)
        .def("get_prototypes", &RandomForestClassifier::get_prototypes, py::arg("n_prototypes") = 3,
             "Get prototypes - most representative samples for each class (Breiman & Cutler's concept).\n\n"
             "For each class, finds samples with highest mean proximity to other samples in that class.\n"
             "These are the 'typical' or 'archetypal' examples for each class.\n\n"
             "Args:\n"
             "    n_prototypes: Number of prototypes per class (default 3)\n\n"
             "Returns:\n"
             "    dict: {class_id: [(sample_idx, prototype_score), ...]}")
        .def("get_local_importance", &RandomForestClassifier::get_local_importance)
        .def("get_local_proximity_importance", &RandomForestClassifier::get_local_proximity_importance,
             "Get local proximity importance matrix (n_samples, n_features). "
             "Value at [i, k] = fraction of trees where permuting feature k changed terminal node for OOB sample i. "
             "Requires compute_proximity_importance=True when fitting.")
        .def("get_overall_proximity_importance", &RandomForestClassifier::get_overall_proximity_importance,
             "Get overall proximity importance vector (n_features,). "
             "Value at [k] = mean fraction of trees where permuting feature k changed terminal node. "
             "Requires compute_proximity_importance=True when fitting.")
        .def("get_top_k_similar_with_explanations", &RandomForestClassifier::get_top_k_similar_with_explanations,
             py::arg("sample_idx"), py::arg("k") = 5, py::arg("n_explanations") = 3,
             "Get top-K most similar samples with feature explanations for a given sample.\n\n"
             "Args:\n"
             "    sample_idx: Index of the query sample\n"
             "    k: Number of similar samples to return (default 5)\n"
             "    n_explanations: Number of top features to explain similarity (default 3)\n\n"
             "Returns:\n"
             "    Tuple of (similar_indices, similarities, explanation_features, explanation_values)")
        .def("get_proximity_importance", &RandomForestClassifier::get_proximity_importance,
             "Legacy alias for get_local_proximity_importance(). Use get_local_proximity_importance() instead.")
        .def("get_proximity_matrix", &RandomForestClassifier::get_proximity_matrix)
        .def("compute_proximity_matrix", &RandomForestClassifier::get_proximity_matrix, "Alias for get_proximity_matrix()")
        .def("get_oob_predictions", &RandomForestClassifier::get_oob_predictions)
        .def("get_lowrank_factors", &RandomForestClassifier::get_lowrank_factors)
        .def("get_oob_counts", &RandomForestClassifier::get_oob_counts,
             "Get the OOB counts for each sample (number of trees where sample was OOB).")
        .def("compute_mds_3d_cpu", &RandomForestClassifier::compute_mds_3d_cpu,
             "Compute 3D MDS coordinates from proximity matrix (CPU implementation, fast C++). "
             "Returns numpy array of shape (n_samples, 3) with [x, y, z] coordinates. "
             "Requires compute_proximity=True when fitting.")
        .def("compute_mds_from_factors", &RandomForestClassifier::compute_mds_from_factors,
             py::arg("k") = 3,
             "Compute k-dimensional MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, k) with MDS coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        .def("compute_mds_3d_from_factors", &RandomForestClassifier::compute_mds_3d_from_factors,
             "Compute 3D MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, 3) with [x, y, z] coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        .def("cleanup", [](RandomForestClassifier& self) {
            // Explicit cleanup of GPU memory
            // Safe to call multiple times, idempotent (cuda_cleanup() is idempotent)
            rf::cuda::cuda_cleanup();
        }, "Explicitly clean up GPU memory. Safe to call multiple times. Useful for Jupyter notebook memory management.")
        .def("get_leaf_assignments", &RandomForestClassifier::get_leaf_assignments,
             "Get leaf (terminal node) assignments for all samples in all trees.\n\n"
             "Returns:\n"
             "    2D numpy array of shape (n_trees, n_samples) with terminal node IDs.\n"
             "    Each entry leaf_assignments[t, i] is the terminal node ID for sample i in tree t.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("get_top_k_similar", &RandomForestClassifier::get_top_k_similar,
             py::arg("query_idx"), py::arg("k") = 10, py::arg("exclude_self") = true,
             "Get top-K most similar TRAINING samples using exact leaf-based proximity.\n\n"
             "Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity.\n"
             "This computes similarity as the fraction of trees where two samples land in the same terminal node.\n\n"
             "Args:\n"
             "    query_idx: Index of the query sample (must be in training set)\n"
             "    k: Number of similar samples to return (default 10)\n"
             "    exclude_self: Exclude the query sample from results (default True)\n\n"
             "Returns:\n"
             "    Tuple of (indices, similarity_scores) where indices are the top-K similar sample indices\n"
             "    and similarity_scores are the corresponding proximity values (0-1 range).\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("build_inverted_leaf_index", &RandomForestClassifier::build_inverted_leaf_index,
             "Pre-build inverted leaf index for faster top-K lookups.\n\n"
             "Usually called automatically by get_top_k_similar() on first use.\n"
             "Call explicitly if you want to control when the index is built.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("has_inverted_leaf_index", &RandomForestClassifier::has_inverted_leaf_index,
             "Check if inverted leaf index has been built.")
        // ====================================================================
        // PREDICT METHODS FOR NEW DATA (inference-time)
        // ====================================================================
        .def("predict_leaf_assignments", &RandomForestClassifier::predict_leaf_assignments,
             py::arg("X"),
             "Predict leaf (terminal node) assignments for NEW samples.\n\n"
             "Args:\n"
             "    X: New samples to predict, shape (n_samples_new, n_features)\n\n"
             "Returns:\n"
             "    2D numpy array of shape (n_samples_new, n_trees) with terminal node IDs.\n"
             "    Each entry [i, t] is the terminal node ID for new sample i in tree t.")
        .def("predict_leaf_assignments_sparse", &RandomForestClassifier::predict_leaf_assignments_sparse,
             py::arg("X"),
             "Predict leaf assignments for NEW sparse samples (scipy.sparse.csr_matrix).")
        .def("predict_top_k_similar", &RandomForestClassifier::predict_top_k_similar,
             py::arg("X"), py::arg("k") = 10,
             "Find top-K most similar TRAINING samples for NEW samples.\n\n"
             "Args:\n"
             "    X: New samples, shape (n_samples_new, n_features)\n"
             "    k: Number of similar training samples to return (default 10)\n\n"
             "Returns:\n"
             "    Tuple of (indices, similarity_scores) both of shape (n_samples_new, k).\n"
             "    indices[i, j] is the j-th most similar training sample index for new sample i.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        // ====================================================================
        // BREIMAN-CUTLER OUTLIER DETECTION
        // ====================================================================
        .def("compute_outlier_scores", 
             [](RandomForestClassifier& self, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::real_t> scores = rf->compute_outlier_scores(mode, n_anchors);
                 py::array_t<rf::real_t> result(scores.size());
                 auto buf = result.request();
                 std::copy(scores.begin(), scores.end(), static_cast<rf::real_t*>(buf.ptr));
                 return result;
             },
             py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Compute Breiman-Cutler outlier scores for all training samples.\n\n"
             "Uses the original formula: raw = N_class / sum(prox^2), normalized by MAD.\n"
             "Scores > 10 typically indicate outliers.\n\n"
             "Args:\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    1D array of normalized outlier scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("compute_outliers", 
             [](RandomForestClassifier& self, rf::integer_t k, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 auto outliers = rf->compute_outliers(k, mode, n_anchors);
                 py::array_t<rf::integer_t> indices(k);
                 py::array_t<rf::real_t> scores(k);
                 auto idx_buf = indices.request();
                 auto score_buf = scores.request();
                 for (size_t i = 0; i < outliers.size(); ++i) {
                     static_cast<rf::integer_t*>(idx_buf.ptr)[i] = outliers[i].first;
                     static_cast<rf::real_t*>(score_buf.ptr)[i] = outliers[i].second;
                 }
                 return py::make_tuple(indices, scores);
             },
             py::arg("k") = 10, py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Get top-K outliers (convenience wrapper).\n\n"
             "Args:\n"
             "    k: Number of top outliers to return (default 10)\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    Tuple of (indices, scores) - top-K outlier sample indices and their scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("select_outlier_anchors",
             [](RandomForestClassifier& self, rf::integer_t n_anchors, const std::string& method) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::integer_t> anchors = rf->select_outlier_anchors(n_anchors, method);
                 py::array_t<rf::integer_t> result(anchors.size());
                 auto buf = result.request();
                 std::copy(anchors.begin(), anchors.end(), static_cast<rf::integer_t*>(buf.ptr));
                 return result;
             },
             py::arg("n_anchors") = 100, py::arg("method") = "random",
             "Select anchor points for greedy outlier detection.\n\n"
             "Args:\n"
             "    n_anchors: Number of anchors to select (default 100)\n"
             "    method: 'random' (fast) or 'diverse' (better coverage)\n\n"
             "Returns:\n"
             "    1D array of anchor sample indices.");

    // RandomForestRegressor class with progress bar
    class RandomForestRegressor {
    private:
        rf::RandomForestConfig config_;  // Store config to allow updates
        rf::RandomForest* rf_;  // Use raw pointer to avoid shared_ptr destruction issues
        bool show_progress_;
        std::string progress_desc_;
        py::object progress_bar_;  // Store tqdm progress bar object
        int last_progress_refresh_pos_;  // Track last refresh position for throttling
        std::string gpu_loss_function_str_;  // Store original loss function string for re-validation
        
    public:
        // Getter for internal RF pointer (needed for lambda bindings)
        rf::RandomForest* get_rf_ptr() { return rf_; }
        
        RandomForestRegressor(int nsample = 1000,
                                int ntree = 100,
                                int mdim = 10,
                                int maxcat = 10,
                                int mtry = 0,
                                int maxnode = 0,
                                int minndsize = 1,
                                int ncsplit = 25,
                                int ncmax = 25,
                                int iseed = 12345,
                                bool compute_proximity = false,
                                bool compute_importance = true,
                                bool compute_local_importance = false,
                                bool use_gpu = false,
                                bool use_qlora = false,
                                std::string quant_mode="nf4",
                                bool use_sparse = false,
                                float sparsity_threshold = 1e-6f,
                                int batch_size = 0,
                                int nodesize = 5,
                            float cutoff = 0.01f,
                            bool show_progress = true,
                            std::string progress_desc = "Training Random Forest Regressor",
                            std::string gpu_loss_function = "mse",  // "mse" for regression
                            int rank = 100,  // Low-rank proximity matrix rank (default: 100)
                            int power_iterations = 30,  // Power iteration count for low-rank SVD (30=99% convergence, 100=99.9%)
                            int n_threads_cpu = 0,  // Number of CPU threads for multi-threading (0 = auto-detect)
                            bool use_rfgap = false,  // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
                            std::string importance_method = "local_imp",  // Local importance method: "local_imp" or "clique"
                            int clique_M = -1,  // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)
                            bool use_casewise = false,  // Use case-wise calculations (per-sample) vs non-case-wise (aggregated). Non-case-wise follows UC Berkeley standard.
                            bool compute_proximity_importance = false,  // Measures % of trees where permuting each feature changes terminal node
                            bool compute_leaf_assignments = false,  // Store terminal node per sample per tree for on-demand proximity
                            bool use_histogram = true,  // Use histogram-based split finding (faster for large datasets)
                            int n_bins = 256) {  // Number of bins for histogram (max 256)
        config_.use_histogram = use_histogram;
        config_.n_bins = n_bins;
        config_.task_type = rf::TaskType::REGRESSION;
        config_.nsample = nsample;
        config_.ntree = ntree;
        config_.mdim = mdim;
        config_.nclass = 1;  // Regression has 1 class
        config_.maxcat = maxcat;
        config_.mtry = mtry > 0 ? mtry : static_cast<rf::integer_t>(std::sqrt(static_cast<rf::real_t>(mdim)));
        config_.maxnode = rf::calculate_maxnode(nsample, minndsize, maxnode);
        config_.minndsize = minndsize;
        config_.ncsplit = ncsplit;
        config_.ncmax = ncmax;
        config_.iseed = iseed;
        config_.compute_proximity = compute_proximity;
        config_.compute_importance = compute_importance;
        config_.compute_local_importance = compute_local_importance;
        config_.use_gpu = use_gpu;
        // execution_mode is deprecated - only use use_gpu
        config_.execution_mode = use_gpu ? "gpu" : "cpu";  // Set for backward compatibility only
        config_.use_qlora = use_qlora;
        config_.quant_mode = quantization_string_to_int(quant_mode);
        config_.use_sparse = use_sparse;
        config_.sparsity_threshold = sparsity_threshold;
        config_.batch_size = batch_size;
        config_.nodesize = nodesize;
        config_.cutoff = cutoff;
        // Convert loss function string to integer with validation (only Gini or MSE)
        config_.gpu_loss_function = loss_function_string_to_int(gpu_loss_function, config_.task_type, config_.nclass);
        // gpu_parallel_mode0 is now automatically set based on batch_size in fit() method
        config_.lowrank_rank = rank;
        config_.lowrank_power_iterations = power_iterations;
        config_.n_threads_cpu = n_threads_cpu;
        config_.use_rfgap = use_rfgap;
        config_.importance_method = importance_method;
        config_.clique_M = clique_M;
        config_.use_casewise = use_casewise;
        config_.compute_proximity_importance = compute_proximity_importance;
        config_.compute_leaf_assignments = compute_leaf_assignments;
        
        // Store original string for re-validation in fit()
        gpu_loss_function_str_ = gpu_loss_function;
        
        // Don't create RandomForest yet - will be created in fit() with actual data dimensions
        rf_ = nullptr;  // Initialize to nullptr
        show_progress_ = show_progress;
        progress_desc_ = progress_desc;
        }
        
        // Destructor - safely clean up raw pointer and progress bar
        // This must be exception-safe for Jupyter notebooks
        ~RandomForestRegressor() {
            // Wrap entire destructor in try-catch to prevent any exceptions from propagating
            // In Jupyter, exceptions in destructors can cause kernel crashes
            try {
                // Clear progress bar reference first (safest operation)
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore - progress bar might already be invalid
                }
                
                // Delete RandomForest object with extra safety
                if (rf_ != nullptr) {
                    try {
                    delete rf_;
                    } catch (const std::exception& e) {
                        // Log but don't rethrow - we're in a destructor
                        // CUDA context might be corrupted in Jupyter
                    } catch (...) {
                        // Ignore all other errors during cleanup
                    }
                    rf_ = nullptr;  // Always set to nullptr after delete attempt
                }
            } catch (...) {
                // Ultimate safety net - catch absolutely everything
                // Don't do anything, just ensure pointers are null
                rf_ = nullptr;
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore
                }
            }
        }
        
        // Set use_sparse flag (for auto-detection in fit)
        void set_use_sparse(bool use_sparse) {
            config_.use_sparse = use_sparse;
        }
        
        void fit(py::array_t<rf::real_t> X, py::array_t<rf::real_t> y, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // CRITICAL: Ensure X is C-contiguous (row-major) before accessing data
            // NumPy arrays may be Fortran-contiguous (column-major) which would cause data corruption
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contiguous = X;
            
            // Extract data dimensions from actual data
            auto X_buf = X_contiguous.request();
            auto y_buf = y.request();
            auto sample_weight_buf = sample_weight.request();
            
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional (samples, features)");
            }
            
            // Get actual data dimensions
            rf::integer_t actual_nsample = static_cast<rf::integer_t>(X_buf.shape[0]);
            rf::integer_t actual_mdim = static_cast<rf::integer_t>(X_buf.shape[1]);
            
            // Update config with actual dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            config_.maxnode = rf::calculate_maxnode(config_.nsample, config_.minndsize, config_.maxnode);
            if (config_.mtry == 0) {
                config_.mtry = static_cast<rf::integer_t>(config_.mdim / 3);  // Regression uses mdim/3
            }
            
            // Re-validate loss function (regression always has nclass=1)
            config_.gpu_loss_function = loss_function_string_to_int(gpu_loss_function_str_, config_.task_type, config_.nclass);
            
            // Create or recreate RandomForest with correct dimensions
            rf_ = new rf::RandomForest(config_);
            
            // Print training message
            py::module_ builtins = py::module_::import("builtins");
            builtins.attr("print")(py::str("Training Random Forest Regressor with {} trees...").format(config_.ntree));
            
            // Print batch size info for GPU mode
            if (config_.use_gpu) {
                rf::integer_t batch_size_info;
                std::string batch_mode;
                if (config_.batch_size > 0) {
                    batch_size_info = std::min(config_.batch_size, config_.ntree);
                    batch_mode = "manual";
                } else {
                    try {
                        batch_size_info = rf::cuda::get_recommended_batch_size(config_.ntree);
                        batch_size_info = std::min(batch_size_info, config_.ntree);
                    } catch (...) {
                        batch_size_info = std::min(static_cast<rf::integer_t>(100), config_.ntree);
                    }
                    batch_mode = "auto";
                }
                builtins.attr("print")(py::str("GPU batch_size={} ({})").format(batch_size_info, batch_mode));
            }
            
            // Print memory information based on execution mode (CUDA-safe)
            try {
                if (config_.use_gpu) {
                    // GPU mode - print GPU memory safely
                    py::module_ rf_module = py::module_::import("RFX");
                    rf_module.attr("print_gpu_memory_status")();
                } else {
                    // CPU mode - print CPU memory
                    try {
                        py::module_ psutil_module = py::module_::import("psutil");
                        py::object memory = psutil_module.attr("virtual_memory")();
                        double total_gb = memory.attr("total").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double available_gb = memory.attr("available").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double used_gb = memory.attr("used").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double percent = memory.attr("percent").cast<double>();
                        
                        // Use Python print for proper formatting
                        py::module_ builtins = py::module_::import("builtins");
                        builtins.attr("print")("\n💻 CPU MEMORY INFORMATION");
                        builtins.attr("print")("==================================================");
                        builtins.attr("print")("📊 System Memory:");
                        builtins.attr("print")(py::str("   Total: {:.1f} GB").format(total_gb));
                        builtins.attr("print")(py::str("   Available: {:.1f} GB").format(available_gb));
                        builtins.attr("print")(py::str("   Used: {:.1f} GB").format(used_gb));
                        builtins.attr("print")(py::str("   Usage: {:.1f}%").format(percent));
                        builtins.attr("print")();
                    } catch (...) {
                        // psutil not available, skip CPU memory info
                    }
                }
            } catch (...) {
                // Ignore errors - don't crash if memory info unavailable
            }
            
            if (show_progress_) {
                // Set up tqdm-style progress bar using Python callback for both CPU and GPU mode
                // This is safe for Jupyter notebooks with proper throttling
                try {
                        // Import regular tqdm (not tqdm.auto) to avoid notebook widget cleanup issues
                        py::object tqdm_module;
                        try {
                            // Use regular tqdm (works in both terminal and Jupyter, avoids widget cleanup issues)
                            tqdm_module = py::module_::import("tqdm");
                        } catch (...) {
                            // tqdm not available, disable progress silently
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                        // Get total number of trees from config
                        rf::integer_t total_trees = config_.ntree;
                        
                        // Use regular tqdm (not tqdm.auto) to avoid notebook widget issues
                        // tqdm.auto can cause cleanup problems in Jupyter
                        try {
                            // Try to use regular tqdm (works in both terminal and Jupyter)
                            // Use iterable with total explicitly set for proper formatting
                            // Create a range object to iterate over (but we'll update manually)
                            py::object range_obj = py::module_::import("builtins").attr("range")(py::cast(total_trees));
                            progress_bar_ = tqdm_module.attr("tqdm")(
                                range_obj,
                                py::arg("total") = py::cast(total_trees),  // Explicitly set total
                                py::arg("desc") = progress_desc_,
                                py::arg("unit") = "tree",
                                py::arg("unit_scale") = false,  // Don't scale units (keep "tree" not "Ktree")
                                py::arg("leave") = false,
                                py::arg("dynamic_ncols") = true,
                                py::arg("miniters") = 1,
                                py::arg("disable") = false,
                                py::arg("bar_format") = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
                            );
                        } catch (...) {
                            // If tqdm creation fails, disable progress
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                            // Force initial display of progress bar
                            try {
                                progress_bar_.attr("refresh")();
                            } catch (...) {
                                // Ignore refresh errors
                            }
                            
                            // Throttle progress updates to avoid Jupyter IOPub rate limit
                            // Reset refresh position tracker for this training session
                            last_progress_refresh_pos_ = 0;
                            int refresh_interval = config_.batch_size > 0 ? config_.batch_size : 5; // Refresh every batch
                            
                            // Ultra-safe callback - minimal Python interaction, no refresh unless critical
                            rf_->set_progress_callback([this, refresh_interval](rf::integer_t current, rf::integer_t total) {
                                // Wrap everything in try-catch to prevent any crashes
                                try {
                                    // Check if progress bar exists and is valid
                                    if (progress_bar_.is_none()) {
                                        return;  // Progress bar cleared, skip
                                    }
                                    
                                    // Very minimal update - just update the counter, no refresh
                                    if (current > last_progress_refresh_pos_) {
                                        try {
                                            // Get current position safely
                                            int current_pos = progress_bar_.attr("n").cast<int>();
                                            if (current > current_pos) {
                                                int update_amount = current - current_pos;
                                                progress_bar_.attr("update")(py::cast(update_amount));
                                            }
                                        } catch (...) {
                                            // Ignore update errors - don't crash
                                        }
                                        
                                        // Only refresh very infrequently to avoid Jupyter crashes
                                        if (current >= total || (current - last_progress_refresh_pos_ >= refresh_interval)) {
                                            try {
                                                progress_bar_.attr("refresh")();
                                                last_progress_refresh_pos_ = current;
                                            } catch (...) {
                                                // Ignore refresh errors - critical to not crash here
                                            }
                                        }
                                    }
                                    
                                    // Handle completion
                                    if (current >= total) {
                                        try {
                                            // Final update
                                            int final_pos = progress_bar_.attr("n").cast<int>();
                                            if (final_pos < total) {
                                                progress_bar_.attr("update")(py::cast(total - final_pos));
                                            }
                                            // Close and clear
                                            if (py::hasattr(progress_bar_, "close")) {
                                                progress_bar_.attr("close")();
                                            }
                                            progress_bar_ = py::none();
                                        } catch (...) {
                                            // Ignore all errors during cleanup
                                            progress_bar_ = py::none();
                                        }
                                    }
                                } catch (...) {
                                    // Ultimate safety net - catch absolutely everything
                                    // Don't do anything, just return
                                }
                            });
                        }
                    }
                } catch (...) {
                    // If tqdm fails, disable progress silently
                    show_progress_ = false;
                }
            }
            
            // Print auto-selected batch size if using GPU with auto-scaling (batch_size=0)
            if (config_.use_gpu && config_.batch_size == 0 && config_.ntree >= 10) {
                try {
                    rf::integer_t recommended_batch = rf::cuda::get_recommended_batch_size(config_.ntree);
                    recommended_batch = std::min(recommended_batch, config_.ntree);
                    py::module_ builtins = py::module_::import("builtins");
                    builtins.attr("print")(py::str("🔧 Auto-scaling: Selected batch size = {} trees (out of {} total)").format(recommended_batch, config_.ntree));
                } catch (...) {
                    // Ignore errors - the C++ code will print it anyway via std::cout
                }
            }
            
            // Call the fit method with actual data
            rf_->fit(static_cast<rf::real_t*>(X_buf.ptr), 
                     y_buf.ptr, 
                     sample_weight_buf.size > 0 ? static_cast<rf::real_t*>(sample_weight_buf.ptr) : nullptr);
            
            // Print GPU memory status after training completes (if GPU was used)
            // Query GPU memory directly using CUDA API (safe after training completes)
            try {
                if (config_.use_gpu) {
                    // GPU mode - query GPU memory directly using CUDA API
                    // This is safe because training just completed, so CUDA context is active
                    try {
#ifdef CUDA_FOUND
                        size_t free_mem, total_mem;
                        ::cudaError_t err = ::cudaMemGetInfo(&free_mem, &total_mem);
                        
                        if (err == ::cudaSuccess) {
                            // Successfully queried GPU memory - print it in GB format to match CPU format
                            size_t used_mem = total_mem - free_mem;
                            double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
                            double total_gb = total_mem / (1024.0 * 1024.0 * 1024.0);
                            double used_gb = used_mem / (1024.0 * 1024.0 * 1024.0);
                            double usage_percent = (used_mem / (double)total_mem) * 100.0;
                            
                            py::module_ builtins = py::module_::import("builtins");
                            builtins.attr("print")("\n🚀 GPU MEMORY STATUS (After Training):");
                            builtins.attr("print")("==================================================");
                            builtins.attr("print")(py::str("📊 GPU Memory:"));
                            builtins.attr("print")(py::str("   Total: {:.1f} GB").format(total_gb));
                            builtins.attr("print")(py::str("   Available: {:.1f} GB").format(free_gb));
                            builtins.attr("print")(py::str("   Used: {:.1f} GB").format(used_gb));
                            builtins.attr("print")(py::str("   Usage: {:.1f}%").format(usage_percent));
                        }
                        // If cudaMemGetInfo fails, just silently skip (don't print error messages)
#endif
                    } catch (...) {
                        // Ignore errors - don't crash if memory query fails
                    }
                }
            } catch (...) {
                // Ignore all errors - don't crash if memory info unavailable
            }
            
            // Close progress bar immediately after training completes to prevent GC issues
            if (show_progress_ && !progress_bar_.is_none()) {
                try {
                    // Progress bar should already be closed by callback, but ensure it's cleared
                    progress_bar_ = py::none();
                } catch (...) {
                    // Ignore any errors
                    progress_bar_ = py::none();
                }
            }
        }
        
        // ====================================================================
        // SPARSE FIT - accepts scipy.sparse.csr_matrix for regression
        // ====================================================================
        void fit_sparse(py::object X_sparse, py::array_t<rf::real_t> y, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // Convert to CSR if not already
            py::object X_csr = ensure_csr(X_sparse);
            
            // Convert scipy CSR to RFX SparseMatrixCSR
            rf::SparseMatrixCSR sparse_matrix = scipy_csr_to_rfx(X_csr);
            
            if (!sparse_matrix.is_valid()) {
                throw std::runtime_error("Invalid sparse matrix structure");
            }
            
            // Get dimensions from sparse matrix
            rf::integer_t actual_nsample = sparse_matrix.nrows;
            rf::integer_t actual_mdim = sparse_matrix.ncols;
            
            // Update config with sparse dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            config_.nclass = 1;  // Regression
            config_.maxnode = rf::calculate_maxnode(actual_nsample, config_.minndsize, config_.maxnode);
            config_.use_sparse = true;  // Ensure sparse mode is set

            // Recalculate mtry for regression (mdim/3)
            if (config_.mtry <= 0 || config_.mtry > actual_mdim) {
                config_.mtry = std::max(1, static_cast<rf::integer_t>(actual_mdim / 3));
            }
            
            // Always recreate RandomForest for sparse to ensure correct config
            if (rf_ != nullptr) {
                delete rf_;
                rf_ = nullptr;
            }
            rf_ = new rf::RandomForest(config_);
            
            // Get y buffer
            auto y_buf = y.request();
            std::vector<rf::real_t> y_float(actual_nsample);
            const float* y_ptr = static_cast<const float*>(y_buf.ptr);
            for (rf::integer_t i = 0; i < actual_nsample; ++i) {
                y_float[i] = y_ptr[i];
            }
            
            // Get sample weights
            const rf::real_t* weight_ptr = nullptr;
            if (sample_weight.size() > 0) {
                auto w_buf = sample_weight.request();
                weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
            }
            
            // Call sparse fit
            rf_->fit_regression_sparse(sparse_matrix, y_float.data(), weight_ptr);
        }
        
        // Delegate all other methods to the underlying RandomForest
        py::array_t<rf::real_t> predict(py::array_t<rf::real_t> X) {
            // Ensure C-contiguous (row-major) array
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            py::array_t<rf::real_t> predictions(X_buf.shape[0]);
            rf_->predict_regression(static_cast<rf::real_t*>(X_buf.ptr), X_buf.shape[0], predictions.mutable_data());
            return predictions;
        }
        
        // Get prototypes - most representative/central samples (highest mean proximity to all samples)
        // For regression: returns overall central samples (not per-class)
        // Returns list: [(sample_idx, prototype_score), ...]
        py::list get_prototypes(rf::integer_t n_prototypes = 5) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_prototypes()");
            
            rf::integer_t nsamples = rf_->get_n_samples();
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            
            if (!proximity_ptr) throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True");
            
            std::vector<std::pair<rf::real_t, rf::integer_t>> scores;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                rf::real_t sum = 0.0f;
                for (rf::integer_t j = 0; j < nsamples; ++j) {
                    if (i != j) sum += static_cast<rf::real_t>(proximity_ptr[i * nsamples + j]);
                }
                rf::real_t mean_prox = (nsamples > 1) ? sum / (nsamples - 1) : 0.0f;
                scores.push_back({mean_prox, i});
            }
            
            rf::integer_t n_proto = std::min(n_prototypes, nsamples);
            std::partial_sort(scores.begin(), scores.begin() + n_proto, scores.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            
            py::list result;
            for (rf::integer_t i = 0; i < n_proto; ++i) {
                result.append(py::make_tuple(scores[i].second, scores[i].first));
            }
            return result;
        }
        
        rf::real_t get_oob_error() const { return rf_->get_oob_error(); }
        
        // Save model to file
        void save(const std::string& filepath) const {
            if (rf_ == nullptr) {
                throw std::runtime_error("Model must be fitted before saving");
            }
            rf_->save(filepath);
        }
        
        py::array_t<rf::real_t> feature_importances_() {
            rf::integer_t mdim = rf_->get_n_features();
            py::array_t<rf::real_t> importances(mdim);
            const rf::real_t* imp_ptr = rf_->get_feature_importances();
            // Check if pointer is valid before copying
            // If compute_importance=False, feature_importances_ may be empty
            if (imp_ptr != nullptr) {
                std::copy(imp_ptr, imp_ptr + mdim, importances.mutable_data());
            } else {
                // If importance was not computed, return zeros
                std::fill(importances.mutable_data(), importances.mutable_data() + mdim, 0.0f);
            }
            return importances;
        }
        
        rf::TaskType get_task_type() const { return rf_->get_task_type(); }
        rf::integer_t get_n_samples() const { return rf_->get_n_samples(); }
        rf::integer_t get_n_features() const { return rf_->get_n_features(); }
        rf::integer_t get_n_classes() const { return rf_->get_n_classes(); }
        
        // Add local importance and proximity matrix methods
        py::array_t<rf::real_t> get_local_importance() {
            // Get the local importance pointer from C++ and convert to numpy array
            const rf::real_t* local_imp_ptr = rf_->get_qimpm();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (local_imp_ptr) {
                // Copy from C++ array to Python array
                std::copy(local_imp_ptr, local_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Fallback: initialize with zeros if not computed
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get local proximity importance (per-sample, per-feature)
        // Shape: (n_samples, n_features)
        // Value at [i, k] = % of trees where permuting feature k changed terminal node for OOB sample i
        py::array_t<rf::real_t> get_local_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (prox_imp_ptr) {
                std::copy(prox_imp_ptr, prox_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Not computed - return zeros
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get overall proximity importance (aggregated across samples)
        // Shape: (n_features,)
        // Value at [k] = mean % of trees where permuting feature k changed terminal node
        py::array_t<rf::real_t> get_overall_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result(mdim);
            auto buf = result.request();
            rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
            
            if (prox_imp_ptr && nsamples > 0) {
                for (rf::integer_t k = 0; k < mdim; ++k) {
                    rf::real_t sum = 0.0f;
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        sum += prox_imp_ptr[i * mdim + k];
                    }
                    dest[k] = sum / static_cast<rf::real_t>(nsamples);
                }
            } else {
                std::fill(dest, dest + mdim, 0.0f);
            }
            return result;
        }
        
        // Legacy alias for backward compatibility
        py::array_t<rf::real_t> get_proximity_importance() {
            return get_local_proximity_importance();
        }
        
        // Get top-K similar samples with feature explanations
        // Uses leaf assignments ONLY for similarity (Breiman RF proximity)
        // Uses proximity importance for explanations (which features affect similarity)
        py::tuple get_top_k_similar_with_explanations(
            rf::integer_t sample_idx, rf::integer_t k = 5, rf::integer_t n_explanations = 3
        ) {
            if (!rf_) throw std::runtime_error("Model must be fitted first");
            
            // REQUIRE leaf assignments - no fallback to proximity matrix
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error(
                    "Leaf assignments required. Set compute_leaf_assignments=True when creating the model. "
                    "This method uses exact RF proximity from leaf co-occurrence."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            rf::integer_t ntree = rf_->get_n_trees();
            if (sample_idx < 0 || sample_idx >= nsamples) throw std::runtime_error("sample_idx out of range");
            k = std::min(k, nsamples - 1);
            n_explanations = std::min(n_explanations, mdim);
            
            // Compute similarity from leaf assignments (exact RF proximity)
            std::vector<rf::real_t> similarities_raw(nsamples, 0.0f);
            for (rf::integer_t t = 0; t < ntree; ++t) {
                int16_t query_leaf = leaf_ptr[t * nsamples + sample_idx];
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (i != sample_idx && leaf_ptr[t * nsamples + i] == query_leaf) {
                        similarities_raw[i] += 1.0f;
                    }
                }
            }
            // Normalize by ntree (fraction of trees where samples co-occur)
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                similarities_raw[i] /= static_cast<rf::real_t>(ntree);
            }
            
            // Create normalized version (max similarity = 1.0)
            // This ensures comparable results between dense and sparse modes
            std::vector<rf::real_t> similarities_norm = similarities_raw;
            rf::real_t max_sim = 0.0f;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                if (i != sample_idx && similarities_raw[i] > max_sim) {
                    max_sim = similarities_raw[i];
                }
            }
            if (max_sim > 0.0f) {
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    similarities_norm[i] /= max_sim;
                }
            }
            
            // Get top-K by similarity (use normalized for ranking)
            std::vector<std::tuple<rf::real_t, rf::real_t, rf::integer_t>> sims;
            for (rf::integer_t j = 0; j < nsamples; ++j) {
                if (j != sample_idx) {
                    sims.push_back({similarities_norm[j], similarities_raw[j], j});
                }
            }
            std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), 
                [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
            
            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> sim_raw_arr(k);
            py::array_t<rf::real_t> sim_norm_arr(k);
            auto idx_buf = idx_arr.request();
            auto sim_raw_buf = sim_raw_arr.request();
            auto sim_norm_buf = sim_norm_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = std::get<2>(sims[i]);
                static_cast<rf::real_t*>(sim_raw_buf.ptr)[i] = std::get<1>(sims[i]);
                static_cast<rf::real_t*>(sim_norm_buf.ptr)[i] = std::get<0>(sims[i]);
            }
            
            // Get explanations from proximity importance (Breiman/Cutler: which features affect similarity)
            const rf::real_t* pi = rf_->get_proximity_importance();
            py::array_t<rf::integer_t> feat_arr(n_explanations); py::array_t<rf::real_t> val_arr(n_explanations);
            auto feat_buf = feat_arr.request(); auto val_buf = val_arr.request();
            if (pi) {
                std::vector<std::pair<rf::real_t, rf::integer_t>> fi;
                for (rf::integer_t f = 0; f < mdim; ++f) fi.push_back({pi[sample_idx * mdim + f], f});
                std::partial_sort(fi.begin(), fi.begin() + n_explanations, fi.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (rf::integer_t i = 0; i < n_explanations; ++i) {
                    static_cast<rf::integer_t*>(feat_buf.ptr)[i] = fi[i].second;
                    static_cast<rf::real_t*>(val_buf.ptr)[i] = fi[i].first;
                }
            } else { 
                std::fill_n(static_cast<rf::integer_t*>(feat_buf.ptr), n_explanations, 0); 
                std::fill_n(static_cast<rf::real_t*>(val_buf.ptr), n_explanations, 0.0f); 
            }
            // Return: (indices, raw_similarities, normalized_similarities, feature_indices, feature_scores)
            return py::make_tuple(idx_arr, sim_raw_arr, sim_norm_arr, feat_arr, val_arr);
        }
        
        py::array_t<rf::real_t> get_proximity_matrix() {
            // Get the proximity matrix pointer and convert to numpy array
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            rf::integer_t nsamples = rf_->get_n_samples();
            
            py::array_t<rf::real_t> result({nsamples, nsamples});
            auto buf = result.request();
            
            if (proximity_ptr) {
                // Convert from dp_t to real_t
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                // Use element-by-element copy with bounds checking to avoid memory corruption
                rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
                const rf::dp_t* src = proximity_ptr;
                size_t n_elements = static_cast<size_t>(nsamples) * nsamples;
                
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                for (size_t i = 0; i < n_elements; ++i) {
                    dest[i] = static_cast<rf::real_t>(src[i]);
                }
            } else {
                // Low-rank mode: full matrix not available
                if (rf_->get_compute_proximity()) {
                    std::string error_msg = 
                        "Proximity matrix not available in full form. "
                        "Low-rank mode is active (use_qlora=True). "
                        "Full matrix reconstruction would require O(n²) memory and can CRASH your system! "
                        "Example: 100k samples = ~80 GB (likely to crash!). "
                        "Proximity is stored in low-rank factors (A and B). "
                        "Use get_lowrank_factors() or compute_mds_3d_from_factors() instead. "
                        "Or disable use_qlora=True for smaller datasets that can fit full matrix.";
                    
                    if (nsamples > 50000) {
                        error_msg += " ERROR: Dataset too large (" + std::to_string(nsamples) + 
                                    " samples) - reconstruction aborted to prevent system crash.";
                    }
                    
                    throw std::runtime_error(error_msg);
                } else {
                    // Fallback: initialize with identity matrix if not computed
                    std::fill(static_cast<rf::real_t*>(buf.ptr), 
                             static_cast<rf::real_t*>(buf.ptr) + nsamples * nsamples, 0.0f);
                    // Set diagonal to 1.0
                    for (rf::integer_t i = 0; i < nsamples; i++) {
                        static_cast<rf::real_t*>(buf.ptr)[i * nsamples + i] = 1.0f;
                    }
                }
            }
            
            return result;
        }
        
        py::tuple get_lowrank_factors() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_lowrank_factors()");
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::dp_t* A_host = nullptr;
            rf::dp_t* B_host = nullptr;
            rf::integer_t rank = 0;
            
            bool success = rf_->get_lowrank_factors(&A_host, &B_host, &rank);
            
            if (!success) {
                throw std::runtime_error(
                    "Low-rank factors not available. "
                    "Either low-rank mode is not active (use_qlora=True required), "
                    "or factors have not been computed yet."
                );
            }
            
            // Create numpy arrays from host memory
            py::array_t<rf::dp_t> A_array(std::vector<py::ssize_t>{nsamples, rank});
            py::array_t<rf::dp_t> B_array(std::vector<py::ssize_t>{nsamples, rank});
            
            auto A_buf = A_array.request();
            auto B_buf = B_array.request();
            
            if (!A_buf.ptr || !B_host) {
                delete[] A_host;
                delete[] B_host;
                throw std::runtime_error("Failed to allocate memory for low-rank factors");
            }
            
            // Copy data
            size_t factor_size = static_cast<size_t>(nsamples) * rank;
            std::copy(A_host, A_host + factor_size, static_cast<rf::dp_t*>(A_buf.ptr));
            std::copy(B_host, B_host + factor_size, static_cast<rf::dp_t*>(B_buf.ptr));
            
            // Free host memory
            delete[] A_host;
            delete[] B_host;
            
            return py::make_tuple(A_array, B_array, rank);
        }

        // Get leaf assignments (terminal node IDs) for all samples in all trees
        // Returns 2D array of shape (ntree, nsample) with terminal node IDs
        py::array_t<int16_t> get_leaf_assignments() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_leaf_assignments()");
            
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            rf::integer_t ntree = rf_->get_n_trees();
            rf::integer_t nsample = rf_->get_n_samples();
            
            // Create numpy array of shape (ntree, nsample)
            py::array_t<int16_t> result({ntree, nsample});
            auto buf = result.request();
            int16_t* result_ptr = static_cast<int16_t*>(buf.ptr);
            
            // Copy data (stored as [tree * nsample + sample])
            std::copy(leaf_ptr, leaf_ptr + ntree * nsample, result_ptr);
            
            return result;
        }
        
        // Get top-K similar samples using leaf assignments (exact proximity)
        // Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity
        // Returns tuple of (indices, similarity_scores)
        py::tuple get_top_k_similar(rf::integer_t query_idx, int k = 10, bool exclude_self = true) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_top_k_similar()");
            
            // Lazy build inverted index on first call for optimal performance
            if (!rf_->has_inverted_leaf_index() && rf_->has_leaf_assignments()) {
                rf_->build_inverted_leaf_index();
            }
            
            auto results = rf_->get_top_k_similar(query_idx, k, exclude_self);
            
            if (results.empty()) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            // Create output arrays
            rf::integer_t actual_k = static_cast<rf::integer_t>(results.size());
            py::array_t<rf::integer_t> top_k_indices(actual_k);
            py::array_t<rf::real_t> top_k_scores(actual_k);
            auto idx_buf = top_k_indices.request();
            auto score_buf = top_k_scores.request();
            
            for (rf::integer_t i = 0; i < actual_k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = results[i].first;
                static_cast<rf::real_t*>(score_buf.ptr)[i] = results[i].second;
            }
            
            return py::make_tuple(top_k_indices, top_k_scores);
        }
        
        // Build inverted leaf index for fast top-K lookups (usually called automatically)
        void build_inverted_leaf_index() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling build_inverted_leaf_index()");
            rf_->build_inverted_leaf_index();
        }
        
        // Check if inverted leaf index is built
        bool has_inverted_leaf_index() const {
            if (!rf_) return false;
            return rf_->has_inverted_leaf_index();
        }

        
        py::array_t<rf::integer_t> get_oob_counts() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_oob_counts()");
            rf::integer_t nsamples = rf_->get_n_samples();
            const rf::integer_t* nout_ptr = rf_->get_nout_ptr();
            
            if (!nout_ptr) {
                throw std::runtime_error("OOB counts not available.");
            }
            
            py::array_t<rf::integer_t> result(nsamples);
            auto result_buf = result.request();
            rf::integer_t* result_ptr = static_cast<rf::integer_t*>(result_buf.ptr);
            std::copy(nout_ptr, nout_ptr + nsamples, result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_3d_cpu() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_3d_cpu()");
            
            // Get proximity matrix
            py::array_t<rf::dp_t> prox_array = get_proximity_matrix();
            if (prox_array.size() == 0) {
                throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True when fitting.");
            }
            
            auto prox_buf = prox_array.request();
            rf::dp_t* prox_ptr = static_cast<rf::dp_t*>(prox_buf.ptr);
            rf::integer_t n_samples = static_cast<rf::integer_t>(std::sqrt(prox_array.size()));
            
            // Call C++ CPU MDS implementation (uses LAPACK)
            // Pass OOB counts for RF-GAP normalization only if RF-GAP is actually enabled
            // Note: If RF-GAP was computed as full matrix, it's already normalized by |Si| in cpu_proximity_rfgap
            // But if proximity matrix was reconstructed from low-rank factors (QLoRA), normalization is required here
            // Get OOB counts for RF-GAP normalization if available
            const rf::integer_t* oob_counts = rf_->get_nout_ptr();
            // Only normalize by OOB counts if RF-GAP is actually enabled
            // Standard proximity matrices are already normalized and don't need RF-GAP normalization
            bool use_rfgap = rf_->get_use_rfgap();
            std::vector<double> coords_3d = rf::compute_mds_3d_cpu(prox_ptr, n_samples, true, oob_counts, use_rfgap);
            
            // Convert to numpy array
            py::array_t<double> result(std::vector<py::ssize_t>{n_samples, 3});
            auto result_buf = result.request();
            double* result_ptr = static_cast<double*>(result_buf.ptr);
            std::copy(coords_3d.begin(), coords_3d.end(), result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_from_factors(rf::integer_t k = 3) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_from_factors()");
            
            if (k < 1) {
                throw std::runtime_error("MDS dimension k must be >= 1");
            }
            
            std::vector<double> coords = rf_->compute_mds_from_factors(k);
            
            if (coords.empty()) {
                throw std::runtime_error(
                    "MDS computation failed. "
                    "Low-rank factors may not be available (use_qlora=True required), "
                    "or computation failed."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            py::array_t<double> coords_array(std::vector<py::ssize_t>{nsamples, static_cast<py::ssize_t>(k)});
            auto coords_buf = coords_array.request();
            
            std::copy(coords.begin(), coords.end(), static_cast<double*>(coords_buf.ptr));
            
            // Check for duplicate coordinates (indicates insufficient tree coverage)
            py::module_ np = py::module_::import("numpy");
            py::module_ warnings = py::module_::import("warnings");
            py::array_t<double> rounded = np.attr("round")(coords_array, 6);
            py::array_t<double> unique_coords = np.attr("unique")(rounded, py::arg("axis")=0);
            py::ssize_t n_unique = unique_coords.request().shape[0];
            py::ssize_t n_duplicates = nsamples - n_unique;
            if (n_duplicates > 0) {
                double pct_duplicates = (static_cast<double>(n_duplicates) / nsamples) * 100.0;
                
                // Calculate recommended tree count based on sample size
                int recommended_trees;
                std::string size_category;
                if (nsamples < 500) {
                    recommended_trees = nsamples * 8;
                    size_category = "8-10× samples";
                } else if (nsamples < 2000) {
                    recommended_trees = nsamples * 6;
                    size_category = "6-8× samples";
                } else if (nsamples < 10000) {
                    recommended_trees = nsamples * 5;
                    size_category = "5-7× samples";
                } else {
                    recommended_trees = nsamples * 4;
                    size_category = "3-5× samples";
                }
                
                std::string warn_msg = "MDS coordinates have " + std::to_string(n_duplicates) + 
                    " duplicate points (" + std::to_string(static_cast<int>(pct_duplicates)) + "% of " + 
                    std::to_string(nsamples) + " samples). Only " + std::to_string(n_unique) + 
                    " unique positions will be visible. This typically indicates insufficient tree coverage " +
                    "for proximity computation. Recommend ntree ≥ " + std::to_string(recommended_trees) + 
                    " (" + size_category + ") for complete MDS coverage.";
                warnings.attr("warn")(warn_msg, py::module_::import("builtins").attr("UserWarning"));
            }
            
            return coords_array;
        }
        
        py::array_t<double> compute_mds_3d_from_factors() {
            return compute_mds_from_factors(3);
        }
    };

    // Register RandomForestRegressor class
    py::class_<RandomForestRegressor>(m, "RandomForestRegressor")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool, bool, 
                      bool, bool, std::string, bool, float, int, int, float, bool, std::string, std::string, int, int, int, bool, std::string, int, bool, bool, bool, bool, int>(),
    py::arg("nsample") = 1000,
    py::arg("ntree") = 100,
    py::arg("mdim") = 10,
    py::arg("maxcat") = 10,
    py::arg("mtry") = 0,
    py::arg("maxnode") = 0,
    py::arg("minndsize") = 1,
    py::arg("ncsplit") = 25,
    py::arg("ncmax") = 25,
    py::arg("iseed") = 12345,
    py::arg("compute_proximity") = false,
    py::arg("compute_importance") = true,
    py::arg("compute_local_importance") = false,
    py::arg("use_gpu") = false,
    py::arg("use_qlora") = false,
    py::arg("quant_mode") = "nf4",
    py::arg("use_sparse") = false,
    py::arg("sparsity_threshold") = 1e-6f,
    py::arg("batch_size") = 0,
    py::arg("nodesize") = 5,
    py::arg("cutoff") = 0.01f,
             py::arg("show_progress") = true,
             py::arg("progress_desc") = "Training Random Forest Regressor",
             py::arg("gpu_loss_function") = "mse",  // "mse" for regression
             py::arg("rank") = 32,  // Low-rank proximity matrix rank (32 preserves 99%+ geometry)
             py::arg("power_iterations") = 30,  // Power iteration count for low-rank SVD
             py::arg("n_threads_cpu") = 0,  // Number of CPU threads (0 = auto-detect)
             py::arg("use_rfgap") = false,  // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
             py::arg("importance_method") = "local_imp",  // Local importance method: "local_imp" or "clique"
             py::arg("clique_M") = -1,  // CLIQUE quantile grid size (-1 = auto, positive integer = explicit M)
             py::arg("use_casewise") = false,  // Use case-wise calculations (bootstrap frequency weighted) vs non-case-wise (simple averaging)
             py::arg("compute_proximity_importance") = false,  // Measures % of trees where permuting each feature changes terminal node
             py::arg("compute_leaf_assignments") = false,  // Store terminal node per sample per tree for on-demand proximity
             py::arg("use_histogram") = true,  // Use histogram-based split finding (faster for large datasets >= 1000 samples)
             py::arg("n_bins") = 256)  // Number of bins for histogram (max 256)
        // Auto-detecting fit: accepts both dense numpy arrays and scipy sparse matrices
        .def("fit", [](RandomForestRegressor& self, py::object X, py::array_t<rf::real_t> y, py::array_t<rf::real_t> sample_weight) {
            if (is_scipy_sparse(X)) {
                // Auto-detected sparse input - enable sparse mode and call fit_sparse
                self.set_use_sparse(true);
                return self.fit_sparse(X, y, sample_weight);
            } else {
                // Dense input - call regular fit
                py::array_t<rf::real_t> X_dense = X.cast<py::array_t<rf::real_t>>();
                return self.fit(X_dense, y, sample_weight);
            }
        }, py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit the random forest model. Auto-detects scipy.sparse matrices and uses sparse kernels automatically.")
        .def("fit_sparse", [](RandomForestRegressor& self, py::object X_sparse, py::array_t<rf::real_t> y) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, y, py::array_t<rf::real_t>());
        }, py::arg("X"), py::arg("y"),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix). "
           "For recommender systems with sparse user-item rating matrices.")
        .def("fit_sparse", [](RandomForestRegressor& self, py::object X_sparse, py::array_t<rf::real_t> y, py::array_t<rf::real_t> sample_weight) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, y, sample_weight);
        }, py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix) with sample weights.")
        .def("predict", &RandomForestRegressor::predict)
        .def("get_oob_error", &RandomForestRegressor::get_oob_error)
        .def("get_oob_mse", &RandomForestRegressor::get_oob_error, "Alias for get_oob_error() - returns OOB Mean Squared Error for regression.")
        .def("save", &RandomForestRegressor::save, py::arg("filepath"),
             "Save trained model to file (binary format). Can be loaded with rfx.load().")
        .def("feature_importances_", &RandomForestRegressor::feature_importances_)
        .def("get_task_type", &RandomForestRegressor::get_task_type)
        .def("get_n_samples", &RandomForestRegressor::get_n_samples)
        .def("get_n_features", &RandomForestRegressor::get_n_features)
        .def("get_n_classes", &RandomForestRegressor::get_n_classes)
        .def("get_prototypes", &RandomForestRegressor::get_prototypes, py::arg("n_prototypes") = 5,
             "Get prototypes - most representative/central samples in proximity space.\n\n"
             "Finds samples with highest mean proximity to all other samples.\n"
             "These are the most 'typical' or 'central' examples in the dataset.\n\n"
             "Args:\n"
             "    n_prototypes: Number of prototypes to return (default 5)\n\n"
             "Returns:\n"
             "    list: [(sample_idx, prototype_score), ...]")
        .def("get_local_importance", &RandomForestRegressor::get_local_importance)
        .def("get_local_proximity_importance", &RandomForestRegressor::get_local_proximity_importance,
             "Get local proximity importance matrix (n_samples, n_features). "
             "Value at [i, k] = fraction of trees where permuting feature k changed terminal node for OOB sample i. "
             "Requires compute_proximity_importance=True when fitting.")
        .def("get_overall_proximity_importance", &RandomForestRegressor::get_overall_proximity_importance,
             "Get overall proximity importance vector (n_features,). "
             "Value at [k] = mean fraction of trees where permuting feature k changed terminal node. "
             "Requires compute_proximity_importance=True when fitting.")
        .def("get_top_k_similar_with_explanations", &RandomForestRegressor::get_top_k_similar_with_explanations,
             py::arg("sample_idx"), py::arg("k") = 5, py::arg("n_explanations") = 3,
             "Get top-K most similar samples with feature explanations for a given sample.\n\n"
             "Args:\n"
             "    sample_idx: Index of the query sample\n"
             "    k: Number of similar samples to return (default 5)\n"
             "    n_explanations: Number of top features to explain similarity (default 3)\n\n"
             "Returns:\n"
             "    Tuple of (similar_indices, similarities, explanation_features, explanation_values)")
        .def("get_proximity_importance", &RandomForestRegressor::get_proximity_importance,
             "Legacy alias for get_local_proximity_importance(). Use get_local_proximity_importance() instead.")
        .def("get_leaf_assignments", &RandomForestRegressor::get_leaf_assignments,
             "Get leaf (terminal node) assignments for all samples in all trees.\n\n"
             "Returns:\n"
             "    numpy.ndarray: 2D array of shape (ntree, nsample) with terminal node IDs.\n"
             "    Each entry leaf_assignments[t, i] is the terminal node ID for sample i in tree t.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("get_top_k_similar", &RandomForestRegressor::get_top_k_similar,
             py::arg("query_idx"), py::arg("k") = 10, py::arg("exclude_self") = true,
             "Get top-K most similar TRAINING samples using exact leaf-based proximity.\n\n"
             "Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity.\n\n"
             "Args:\n"
             "    query_idx: Index of the query sample (must be in training set)\n"
             "    k: Number of similar samples to return (default 10)\n"
             "    exclude_self: Exclude the query sample from results (default True)\n\n"
             "Returns:\n"
             "    Tuple of (indices, similarity_scores) arrays.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("build_inverted_leaf_index", &RandomForestRegressor::build_inverted_leaf_index,
             "Pre-build inverted leaf index for faster top-K lookups.\n\n"
             "Usually called automatically by get_top_k_similar() on first use.\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("has_inverted_leaf_index", &RandomForestRegressor::has_inverted_leaf_index,
             "Check if inverted leaf index has been built.")
        .def("get_proximity_matrix", &RandomForestRegressor::get_proximity_matrix)
        .def("compute_proximity_matrix", &RandomForestRegressor::get_proximity_matrix, "Alias for get_proximity_matrix()")
        .def("get_lowrank_factors", &RandomForestRegressor::get_lowrank_factors)
        .def("get_oob_counts", &RandomForestRegressor::get_oob_counts,
             "Get the OOB counts for each sample (number of trees where sample was OOB).")
        .def("compute_mds_3d_cpu", &RandomForestRegressor::compute_mds_3d_cpu,
             "Compute 3D MDS coordinates from proximity matrix (CPU implementation, fast C++). "
             "Returns numpy array of shape (n_samples, 3) with [x, y, z] coordinates. "
             "Requires compute_proximity=True when fitting.")
        .def("compute_mds_from_factors", &RandomForestRegressor::compute_mds_from_factors,
             py::arg("k") = 3,
             "Compute k-dimensional MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, k) with MDS coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        .def("compute_mds_3d_from_factors", &RandomForestRegressor::compute_mds_3d_from_factors,
             "Compute 3D MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, 3) with [x, y, z] coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        // ====================================================================
        // BREIMAN-CUTLER OUTLIER DETECTION (Regressor)
        // ====================================================================
        .def("compute_outlier_scores", 
             [](RandomForestRegressor& self, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::real_t> scores = rf->compute_outlier_scores(mode, n_anchors);
                 py::array_t<rf::real_t> result(scores.size());
                 auto buf = result.request();
                 std::copy(scores.begin(), scores.end(), static_cast<rf::real_t*>(buf.ptr));
                 return result;
             },
             py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Compute Breiman-Cutler outlier scores for all training samples.\n\n"
             "Uses the original formula: raw = N / sum(prox^2), normalized by MAD.\n"
             "Scores > 10 typically indicate outliers.\n\n"
             "Args:\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    1D array of normalized outlier scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("compute_outliers", 
             [](RandomForestRegressor& self, rf::integer_t k, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 auto outliers = rf->compute_outliers(k, mode, n_anchors);
                 py::array_t<rf::integer_t> indices(k);
                 py::array_t<rf::real_t> scores(k);
                 auto idx_buf = indices.request();
                 auto score_buf = scores.request();
                 for (size_t i = 0; i < outliers.size(); ++i) {
                     static_cast<rf::integer_t*>(idx_buf.ptr)[i] = outliers[i].first;
                     static_cast<rf::real_t*>(score_buf.ptr)[i] = outliers[i].second;
                 }
                 return py::make_tuple(indices, scores);
             },
             py::arg("k") = 10, py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Get top-K outliers (convenience wrapper).\n\n"
             "Args:\n"
             "    k: Number of top outliers to return (default 10)\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    Tuple of (indices, scores) - top-K outlier sample indices and their scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("select_outlier_anchors",
             [](RandomForestRegressor& self, rf::integer_t n_anchors, const std::string& method) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::integer_t> anchors = rf->select_outlier_anchors(n_anchors, method);
                 py::array_t<rf::integer_t> result(anchors.size());
                 auto buf = result.request();
                 std::copy(anchors.begin(), anchors.end(), static_cast<rf::integer_t*>(buf.ptr));
                 return result;
             },
             py::arg("n_anchors") = 100, py::arg("method") = "random",
             "Select anchor points for greedy outlier detection.\n\n"
             "Args:\n"
             "    n_anchors: Number of anchors to select (default 100)\n"
             "    method: 'random' (fast) or 'diverse' (better coverage)\n\n"
             "Returns:\n"
             "    1D array of anchor sample indices.")
        .def("cleanup", [](RandomForestRegressor& self) {
            // Explicit cleanup of GPU memory
            // Safe to call multiple times, idempotent (cuda_cleanup() is idempotent)
            rf::cuda::cuda_cleanup();
        }, "Explicitly clean up GPU memory. Safe to call multiple times. Useful for Jupyter notebook memory management.");

    // RandomForestUnsupervised class with progress bar
    class RandomForestUnsupervised {
    private:
        rf::RandomForestConfig config_;  // Store config to allow updates
        rf::RandomForest* rf_;  // Use raw pointer to avoid shared_ptr destruction issues
        bool show_progress_;
        std::string progress_desc_;
        py::object progress_bar_;  // Store tqdm progress bar object
        int last_progress_refresh_pos_;  // Track last refresh position for throttling
        std::string gpu_loss_function_str_;  // Store original loss function string for re-validation
        
    public:
        // Getter for internal RF pointer (needed for lambda bindings)
        rf::RandomForest* get_rf_ptr() { return rf_; }
        
        RandomForestUnsupervised(int nsample = 1000,
                                int ntree = 100,
                                int mdim = 10,
                                int maxcat = 10,
                                int mtry = 0,
                                int maxnode = 0,
                                int minndsize = 1,
                                int ncsplit = 25,
                                int ncmax = 25,
                                int iseed = 12345,
                               bool compute_proximity = true,  // Proximity is the main output for unsupervised
                               bool compute_importance = true,  // With synthetic data, importance is meaningful for unsupervised!
                               bool compute_local_importance = false,  // Per-sample importance
                                bool use_gpu = false,
                                bool use_qlora = false,
                                std::string quant_mode = "nf4",
                                bool use_sparse = false,
                                float sparsity_threshold = 1e-6f,
                                int batch_size = 0,
                                int nodesize = 5,
                                float cutoff = 0.01f,
                               bool show_progress = true,
                               std::string progress_desc = "Training Random Forest Unsupervised",
                               float synthetic_ratio = 1.0f,  // Ratio of synthetic to real samples (default 1:1)
                               int rank = 100,  // Low-rank proximity matrix rank (default: 100)
                               int power_iterations = 30,  // Power iteration count for low-rank SVD (30=99% convergence, 100=99.9%)
                            int n_threads_cpu = 0,  // Number of CPU threads for multi-threading (0 = auto-detect)
                            bool use_rfgap = false,  // Use RF-GAP (Random Forest-Geometry- and Accuracy-Preserving) proximity
                            std::string importance_method = "local_imp",  // Local importance method (not used for unsupervised, kept for API consistency)
                            int clique_M = -1,  // CLIQUE quantile grid size (not used for unsupervised)
                            bool use_casewise = false,  // Use case-wise calculations (bootstrap frequency weighted) vs non-case-wise (simple averaging)
                            bool compute_proximity_importance = false,  // Measures % of trees where permuting each feature changes terminal node. For recommender system interpretability.
                            bool compute_leaf_assignments = false,  // Store terminal node per sample per tree for on-demand proximity
                            bool use_histogram = true,  // Use histogram-based split finding (faster for large datasets)
                            int n_bins = 256) {  // Number of bins for histogram (max 256)
        config_.use_histogram = use_histogram;
        config_.n_bins = n_bins;
        config_.task_type = rf::TaskType::UNSUPERVISED;
        config_.nsample = nsample;
        config_.ntree = ntree;
        config_.mdim = mdim;
        config_.nclass = 1;  // Unsupervised has 1 class
        config_.maxcat = maxcat;
        // Set mtry based on mode (will be overridden in setup_unsupervised_task if mtry==0)
        if (mtry > 0) {
            config_.mtry = mtry;
        } else {
            // Set default based on mode (will be finalized in setup_unsupervised_task)
            config_.mtry = 0;  // Let setup_unsupervised_task determine based on mode
        }
        config_.maxnode = rf::calculate_maxnode(nsample, minndsize, maxnode);
        config_.minndsize = minndsize;
        config_.ncsplit = ncsplit;
        config_.ncmax = ncmax;
        config_.iseed = iseed;
        config_.compute_proximity = compute_proximity;
        // With proper synthetic data generation (real=1, synthetic=0), importance IS meaningful for unsupervised!
        // Overall importance: measures which features help distinguish real from synthetic (i.e., have real structure)
        // Local importance: measures which features make each sample look "real" vs "random"
        config_.compute_importance = compute_importance;
        config_.compute_local_importance = compute_local_importance;
        config_.use_gpu = use_gpu;
        // execution_mode is deprecated - only use use_gpu
        config_.execution_mode = use_gpu ? "gpu" : "cpu";  // Set for backward compatibility only
        config_.use_qlora = use_qlora;
        config_.quant_mode = quantization_string_to_int(quant_mode);
        config_.use_sparse = use_sparse;
        config_.sparsity_threshold = sparsity_threshold;
        config_.batch_size = batch_size;
        config_.nodesize = nodesize;
        config_.cutoff = cutoff;
        // Unsupervised always uses Gini (classification-style) - no MSE option
        config_.unsupervised_mode = rf::UnsupervisedMode::CLASSIFICATION_STYLE;
        config_.gpu_loss_function = 0;  // 0 = Gini (always for unsupervised)
        config_.synthetic_ratio = synthetic_ratio;  // Ratio of synthetic to real samples
        // gpu_parallel_mode0 is now automatically set based on batch_size in fit() method
        config_.lowrank_rank = rank;
        config_.lowrank_power_iterations = power_iterations;
        config_.n_threads_cpu = n_threads_cpu;
        config_.use_rfgap = use_rfgap;
        config_.importance_method = importance_method;
        config_.clique_M = clique_M;
        config_.use_casewise = use_casewise;
        // NOTE: compute_proximity_importance WORKS for unsupervised (unlike regular importance)!
        // It measures terminal node changes when permuting features - no target variable needed.
        // Perfect for recommender system interpretability.
        config_.compute_proximity_importance = compute_proximity_importance;
        config_.compute_leaf_assignments = compute_leaf_assignments;
        
        // No gpu_loss_function_str_ for unsupervised - always uses Gini
        gpu_loss_function_str_ = "gini";
        
        // Don't create RandomForest yet - will be created in fit() with actual data dimensions
        rf_ = nullptr;  // Initialize to nullptr
        show_progress_ = show_progress;
        progress_desc_ = progress_desc;
        }
        
        // Destructor - safely clean up raw pointer and progress bar
        // This must be exception-safe for Jupyter notebooks
        ~RandomForestUnsupervised() {
            // Wrap entire destructor in try-catch to prevent any exceptions from propagating
            // In Jupyter, exceptions in destructors can cause kernel crashes
            try {
                // Clear progress bar reference first (safest operation)
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore - progress bar might already be invalid
                }
                
                // Delete RandomForest object with extra safety
                if (rf_ != nullptr) {
                    try {
                    delete rf_;
                    } catch (const std::exception& e) {
                        // Log but don't rethrow - we're in a destructor
                        // CUDA context might be corrupted in Jupyter
                    } catch (...) {
                        // Ignore all other errors during cleanup
                    }
                    rf_ = nullptr;  // Always set to nullptr after delete attempt
                }
            } catch (...) {
                // Ultimate safety net - catch absolutely everything
                // Don't do anything, just ensure pointers are null
                rf_ = nullptr;
                try {
                progress_bar_ = py::none();
                } catch (...) {
                    // Ignore
                }
            }
        }
        
        // Set use_sparse flag (for auto-detection in fit)
        void set_use_sparse(bool use_sparse) {
            config_.use_sparse = use_sparse;
        }
        
        void fit(py::array_t<rf::real_t> X, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // CRITICAL: Ensure X is C-contiguous (row-major) before accessing data
            // NumPy arrays may be Fortran-contiguous (column-major) which would cause data corruption
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contiguous = X;
            
            // Extract data dimensions from actual data
            auto X_buf = X_contiguous.request();
            auto sample_weight_buf = sample_weight.request();
            
            if (X_buf.ndim != 2) {
                throw std::runtime_error("X must be 2-dimensional (samples, features)");
            }
            
            // Get actual data dimensions
            rf::integer_t actual_nsample = static_cast<rf::integer_t>(X_buf.shape[0]);
            rf::integer_t actual_mdim = static_cast<rf::integer_t>(X_buf.shape[1]);
            
            // Update config with actual dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            // mtry will be set in setup_unsupervised_task based on unsupervised_mode
            
            // Re-validate loss function with actual unsupervised_mode
            rf::TaskType validation_task_type = (config_.unsupervised_mode == rf::UnsupervisedMode::CLASSIFICATION_STYLE) 
                ? rf::TaskType::CLASSIFICATION : rf::TaskType::REGRESSION;
            config_.gpu_loss_function = loss_function_string_to_int(gpu_loss_function_str_, validation_task_type, config_.nclass);
            
            // Create or recreate RandomForest with correct dimensions
            rf_ = new rf::RandomForest(config_);
            
            // Print training message
            py::module_ builtins = py::module_::import("builtins");
            builtins.attr("print")(py::str("Training Random Forest Unsupervised with {} trees...").format(config_.ntree));
            
            // Print batch size info for GPU mode
            if (config_.use_gpu) {
                rf::integer_t batch_size_info;
                std::string batch_mode;
                if (config_.batch_size > 0) {
                    batch_size_info = std::min(config_.batch_size, config_.ntree);
                    batch_mode = "manual";
                } else {
                    try {
                        batch_size_info = rf::cuda::get_recommended_batch_size(config_.ntree);
                        batch_size_info = std::min(batch_size_info, config_.ntree);
                    } catch (...) {
                        batch_size_info = std::min(static_cast<rf::integer_t>(100), config_.ntree);
                    }
                    batch_mode = "auto";
                }
                builtins.attr("print")(py::str("GPU batch_size={} ({})").format(batch_size_info, batch_mode));
            }
            
            // Print memory information based on execution mode (CUDA-safe)
            try {
                if (config_.use_gpu) {
                    // GPU mode - print GPU memory safely
                    py::module_ rf_module = py::module_::import("RFX");
                    rf_module.attr("print_gpu_memory_status")();
                } else {
                    // CPU mode - print CPU memory
                    try {
                        py::module_ psutil_module = py::module_::import("psutil");
                        py::object memory = psutil_module.attr("virtual_memory")();
                        double total_gb = memory.attr("total").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double available_gb = memory.attr("available").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double used_gb = memory.attr("used").cast<double>() / (1024.0 * 1024.0 * 1024.0);
                        double percent = memory.attr("percent").cast<double>();
                        
                        // Use Python print for proper formatting
                        py::module_ builtins = py::module_::import("builtins");
                        builtins.attr("print")("\n💻 CPU MEMORY INFORMATION");
                        builtins.attr("print")("==================================================");
                        builtins.attr("print")("📊 System Memory:");
                        builtins.attr("print")(py::str("   Total: {:.1f} GB").format(total_gb));
                        builtins.attr("print")(py::str("   Available: {:.1f} GB").format(available_gb));
                        builtins.attr("print")(py::str("   Used: {:.1f} GB").format(used_gb));
                        builtins.attr("print")(py::str("   Usage: {:.1f}%").format(percent));
                        builtins.attr("print")();
                    } catch (...) {
                        // psutil not available, skip CPU memory info
                    }
                }
            } catch (...) {
                // Ignore errors - don't crash if memory info unavailable
            }
            
            if (show_progress_) {
                // Set up tqdm-style progress bar using Python callback for both CPU and GPU mode
                // This is safe for Jupyter notebooks with proper throttling
                try {
                        // Import regular tqdm (not tqdm.auto) to avoid notebook widget cleanup issues
                        py::object tqdm_module;
                        try {
                            // Use regular tqdm (works in both terminal and Jupyter, avoids widget cleanup issues)
                            tqdm_module = py::module_::import("tqdm");
                        } catch (...) {
                            // tqdm not available, disable progress silently
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                        // Get total number of trees from config
                        rf::integer_t total_trees = config_.ntree;
                        
                        // Use regular tqdm (not tqdm.auto) to avoid notebook widget issues
                        // tqdm.auto can cause cleanup problems in Jupyter
                        try {
                            // Try to use regular tqdm (works in both terminal and Jupyter)
                            // Use iterable with total explicitly set for proper formatting
                            // Create a range object to iterate over (but we'll update manually)
                            py::object range_obj = py::module_::import("builtins").attr("range")(py::cast(total_trees));
                            progress_bar_ = tqdm_module.attr("tqdm")(
                                range_obj,
                                py::arg("total") = py::cast(total_trees),  // Explicitly set total
                                py::arg("desc") = progress_desc_,
                                py::arg("unit") = "tree",
                                py::arg("unit_scale") = false,  // Don't scale units (keep "tree" not "Ktree")
                                py::arg("leave") = false,
                                py::arg("dynamic_ncols") = true,
                                py::arg("miniters") = 1,
                                py::arg("disable") = false,
                                py::arg("bar_format") = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
                            );
                        } catch (...) {
                            // If tqdm creation fails, disable progress
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                            // Force initial display of progress bar
                            try {
                                progress_bar_.attr("refresh")();
                            } catch (...) {
                                // Ignore refresh errors
                            }
                            
                            // Throttle progress updates to avoid Jupyter IOPub rate limit
                            // Reset refresh position tracker for this training session
                            last_progress_refresh_pos_ = 0;
                            int refresh_interval = config_.batch_size > 0 ? config_.batch_size : 5; // Refresh every batch
                            
                            // Ultra-safe callback - minimal Python interaction, no refresh unless critical
                            rf_->set_progress_callback([this, refresh_interval](rf::integer_t current, rf::integer_t total) {
                                // Wrap everything in try-catch to prevent any crashes
                                try {
                                    // Check if progress bar exists and is valid
                                    if (progress_bar_.is_none()) {
                                        return;  // Progress bar cleared, skip
                                    }
                                    
                                    // Very minimal update - just update the counter, no refresh
                                    if (current > last_progress_refresh_pos_) {
                                        try {
                                            // Get current position safely
                                            int current_pos = progress_bar_.attr("n").cast<int>();
                                            if (current > current_pos) {
                                                int update_amount = current - current_pos;
                                                progress_bar_.attr("update")(py::cast(update_amount));
                                            }
                                        } catch (...) {
                                            // Ignore update errors - don't crash
                                        }
                                        
                                        // Only refresh very infrequently to avoid Jupyter crashes
                                        if (current >= total || (current - last_progress_refresh_pos_ >= refresh_interval)) {
                                            try {
                                                progress_bar_.attr("refresh")();
                                                last_progress_refresh_pos_ = current;
                                            } catch (...) {
                                                // Ignore refresh errors - critical to not crash here
                                            }
                                        }
                                    }
                                    
                                    // Handle completion
                                    if (current >= total) {
                                        try {
                                            // Final update
                                            int final_pos = progress_bar_.attr("n").cast<int>();
                                            if (final_pos < total) {
                                                progress_bar_.attr("update")(py::cast(total - final_pos));
                                            }
                                            // Close and clear
                                            if (py::hasattr(progress_bar_, "close")) {
                                                progress_bar_.attr("close")();
                                            }
                                            progress_bar_ = py::none();
                                        } catch (...) {
                                            // Ignore all errors during cleanup
                                            progress_bar_ = py::none();
                                        }
                                    }
                                } catch (...) {
                                    // Ultimate safety net - catch absolutely everything
                                    // Don't do anything, just return
                                }
                            });
                        }
                    }
                } catch (...) {
                    // If tqdm fails, disable progress silently
                    show_progress_ = false;
                }
            }
            
            // Print auto-selected batch size if using GPU with auto-scaling (batch_size=0)
            if (config_.use_gpu && config_.batch_size == 0 && config_.ntree >= 10) {
                try {
                    rf::integer_t recommended_batch = rf::cuda::get_recommended_batch_size(config_.ntree);
                    recommended_batch = std::min(recommended_batch, config_.ntree);
                    py::module_ builtins = py::module_::import("builtins");
                    builtins.attr("print")(py::str("🔧 Auto-scaling: Selected batch size = {} trees (out of {} total)").format(recommended_batch, config_.ntree));
                } catch (...) {
                    // Ignore errors - the C++ code will print it anyway via std::cout
                }
            }
            
            // Call fit_unsupervised directly (no y parameter needed)
            rf_->fit_unsupervised(static_cast<rf::real_t*>(X_buf.ptr), 
                                 sample_weight_buf.size > 0 ? static_cast<rf::real_t*>(sample_weight_buf.ptr) : nullptr);
        }
        
        // ====================================================================
        // SPARSE FIT - accepts scipy.sparse.csr_matrix for unsupervised learning
        // ====================================================================
        void fit_sparse(py::object X_sparse, py::array_t<rf::real_t> sample_weight = py::array_t<rf::real_t>()) {
            // Convert to CSR if not already
            py::object X_csr = ensure_csr(X_sparse);
            
            // Convert scipy CSR to RFX SparseMatrixCSR
            rf::SparseMatrixCSR sparse_matrix = scipy_csr_to_rfx(X_csr);
            
            if (!sparse_matrix.is_valid()) {
                throw std::runtime_error("Invalid sparse matrix structure");
            }
            
            // Get dimensions from sparse matrix
            rf::integer_t actual_nsample = sparse_matrix.nrows;
            rf::integer_t actual_mdim = sparse_matrix.ncols;
            
            // Update config with sparse dimensions
            config_.nsample = actual_nsample;
            config_.mdim = actual_mdim;
            config_.nclass = 2;  // Unsupervised uses 2 synthetic classes
            config_.use_sparse = true;
            
            // Recalculate mtry (unsupervised uses sqrt(mdim))
            if (config_.mtry <= 0 || config_.mtry > actual_mdim) {
                config_.mtry = std::max(1, static_cast<rf::integer_t>(std::sqrt(static_cast<float>(actual_mdim))));
            }
            
            // Always recreate RandomForest for sparse to ensure correct config
            if (rf_ != nullptr) {
                delete rf_;
                rf_ = nullptr;
            }
            rf_ = new rf::RandomForest(config_);
            
            // Set up progress bar for sparse training (same as dense)
            if (show_progress_) {
                try {
                    py::object tqdm_module;
                    try {
                        tqdm_module = py::module_::import("tqdm");
                    } catch (...) {
                        show_progress_ = false;
                    }
                    
                    if (show_progress_) {
                        rf::integer_t total_trees = config_.ntree;
                        try {
                            py::object range_obj = py::module_::import("builtins").attr("range")(py::cast(total_trees));
                            progress_bar_ = tqdm_module.attr("tqdm")(
                                range_obj,
                                py::arg("total") = py::cast(total_trees),
                                py::arg("desc") = progress_desc_,
                                py::arg("unit") = "tree",
                                py::arg("unit_scale") = false,
                                py::arg("leave") = false,
                                py::arg("dynamic_ncols") = true,
                                py::arg("miniters") = 1,
                                py::arg("disable") = false,
                                py::arg("bar_format") = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} {unit} [{elapsed}<{remaining}, {rate_fmt}]"
                            );
                        } catch (...) {
                            show_progress_ = false;
                        }
                        
                        if (show_progress_) {
                            try {
                                progress_bar_.attr("refresh")();
                            } catch (...) {}
                            
                            last_progress_refresh_pos_ = 0;
                            int refresh_interval = config_.batch_size > 0 ? config_.batch_size : 5;
                            
                            rf_->set_progress_callback([this, refresh_interval](rf::integer_t current, rf::integer_t total) {
                                try {
                                    if (progress_bar_.is_none()) return;
                                    if (current > last_progress_refresh_pos_) {
                                        try {
                                            int current_pos = progress_bar_.attr("n").cast<int>();
                                            if (current > current_pos) {
                                                int update_amount = current - current_pos;
                                                progress_bar_.attr("update")(py::cast(update_amount));
                                            }
                                        } catch (...) {}
                                        
                                        if (current >= total || (current - last_progress_refresh_pos_ >= refresh_interval)) {
                                            try {
                                                progress_bar_.attr("refresh")();
                                                last_progress_refresh_pos_ = current;
                                            } catch (...) {}
                                        }
                                    }
                                    if (current >= total) {
                                        try {
                                            int final_pos = progress_bar_.attr("n").cast<int>();
                                            if (final_pos < total) {
                                                progress_bar_.attr("update")(py::cast(total - final_pos));
                                            }
                                            if (py::hasattr(progress_bar_, "close")) {
                                                progress_bar_.attr("close")();
                                            }
                                            progress_bar_ = py::none();
                                        } catch (...) {
                                            progress_bar_ = py::none();
                                        }
                                    }
                                } catch (...) {}
                            });
                        }
                    }
                } catch (...) {
                    show_progress_ = false;
                }
            }
            
            // Get sample weights
            const rf::real_t* weight_ptr = nullptr;
            if (sample_weight.size() > 0) {
                auto w_buf = sample_weight.request();
                weight_ptr = static_cast<const rf::real_t*>(w_buf.ptr);
            }
            
            // Call sparse fit
            rf_->fit_unsupervised_sparse(sparse_matrix, weight_ptr);
            
            // Clean up progress bar
            if (show_progress_ && !progress_bar_.is_none()) {
                try {
                    progress_bar_ = py::none();
                } catch (...) {}
            }
        }
        
        // Delegate all other methods to the underlying RandomForest
        py::array_t<rf::integer_t> predict(py::array_t<rf::real_t> X) {
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            py::array_t<rf::integer_t> predictions(X_buf.shape[0]);
            rf_->predict(static_cast<rf::real_t*>(X_buf.ptr), X_buf.shape[0], predictions.mutable_data());
            return predictions;
        }
        
        // Predict class probabilities (P(real), P(synthetic)) for unsupervised mode
        // This is useful for imputation validation and anomaly detection
        py::array_t<rf::real_t> predict_proba(py::array_t<rf::real_t> X) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling predict_proba()");
            py::array_t<rf::real_t, py::array::c_style | py::array::forcecast> X_contig(X);
            auto X_buf = X_contig.request();
            // Unsupervised mode has 2 classes: 0=synthetic, 1=real
            py::array_t<rf::real_t> probabilities({X_buf.shape[0], static_cast<py::ssize_t>(rf_->get_n_classes())});
            rf_->predict_proba(static_cast<rf::real_t*>(X_buf.ptr), X_buf.shape[0], probabilities.mutable_data());
            return probabilities;
        }
        
        // Get prototypes - most representative/central samples (highest mean proximity to all samples)
        // For unsupervised: returns overall central samples (cluster centroids in proximity space)
        // Perfect for finding archetypal examples or cluster representatives
        // Returns list: [(sample_idx, prototype_score), ...]
        py::list get_prototypes(rf::integer_t n_prototypes = 5) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_prototypes()");
            
            rf::integer_t nsamples = rf_->get_n_samples();
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            
            if (!proximity_ptr) throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True");
            
            std::vector<std::pair<rf::real_t, rf::integer_t>> scores;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                rf::real_t sum = 0.0f;
                for (rf::integer_t j = 0; j < nsamples; ++j) {
                    if (i != j) sum += static_cast<rf::real_t>(proximity_ptr[i * nsamples + j]);
                }
                rf::real_t mean_prox = (nsamples > 1) ? sum / (nsamples - 1) : 0.0f;
                scores.push_back({mean_prox, i});
            }
            
            rf::integer_t n_proto = std::min(n_prototypes, nsamples);
            std::partial_sort(scores.begin(), scores.begin() + n_proto, scores.end(),
                             [](const auto& a, const auto& b) { return a.first > b.first; });
            
            py::list result;
            for (rf::integer_t i = 0; i < n_proto; ++i) {
                result.append(py::make_tuple(scores[i].second, scores[i].first));
            }
            return result;
        }
        
        rf::real_t get_oob_error() const { return rf_->get_oob_error(); }
        
        // Save model to file
        void save(const std::string& filepath) const {
            if (rf_ == nullptr) {
                throw std::runtime_error("Model must be fitted before saving");
            }
            rf_->save(filepath);
        }
        
        py::array_t<rf::real_t> feature_importances_() {
            rf::integer_t mdim = rf_->get_n_features();
            py::array_t<rf::real_t> importances(mdim);
            const rf::real_t* imp_ptr = rf_->get_feature_importances();
            // Check if pointer is valid before copying
            // If compute_importance=False, feature_importances_ may be empty
            if (imp_ptr != nullptr) {
                std::copy(imp_ptr, imp_ptr + mdim, importances.mutable_data());
            } else {
                // If importance was not computed, return zeros
                std::fill(importances.mutable_data(), importances.mutable_data() + mdim, 0.0f);
            }
            return importances;
        }
        
        rf::TaskType get_task_type() const { return rf_->get_task_type(); }
        rf::integer_t get_n_samples() const { return rf_->get_n_samples(); }
        rf::integer_t get_n_features() const { return rf_->get_n_features(); }
        rf::integer_t get_n_classes() const { return rf_->get_n_classes(); }
        
        // Add local importance and proximity matrix methods
        py::array_t<rf::real_t> get_local_importance() {
            // Get the local importance pointer from C++ and convert to numpy array
            const rf::real_t* local_imp_ptr = rf_->get_qimpm();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (local_imp_ptr) {
                // Copy from C++ array to Python array
                std::copy(local_imp_ptr, local_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Fallback: initialize with zeros if not computed
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get local proximity importance (per-sample, per-feature)
        // Shape: (n_samples, n_features)
        // Value at [i, k] = % of trees where permuting feature k changed terminal node for OOB sample i
        py::array_t<rf::real_t> get_local_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result({nsamples, mdim});
            auto buf = result.request();
            
            if (prox_imp_ptr) {
                std::copy(prox_imp_ptr, prox_imp_ptr + nsamples * mdim, static_cast<rf::real_t*>(buf.ptr));
            } else {
                // Not computed - return zeros
                std::fill(static_cast<rf::real_t*>(buf.ptr), 
                         static_cast<rf::real_t*>(buf.ptr) + nsamples * mdim, 0.0f);
            }
            return result;
        }
        
        // Get overall proximity importance (aggregated across samples)
        // Shape: (n_features,)
        // Value at [k] = mean % of trees where permuting feature k changed terminal node
        py::array_t<rf::real_t> get_overall_proximity_importance() {
            const rf::real_t* prox_imp_ptr = rf_->get_proximity_importance();
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            
            py::array_t<rf::real_t> result(mdim);
            auto buf = result.request();
            rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
            
            if (prox_imp_ptr && nsamples > 0) {
                for (rf::integer_t k = 0; k < mdim; ++k) {
                    rf::real_t sum = 0.0f;
                    for (rf::integer_t i = 0; i < nsamples; ++i) {
                        sum += prox_imp_ptr[i * mdim + k];
                    }
                    dest[k] = sum / static_cast<rf::real_t>(nsamples);
                }
            } else {
                std::fill(dest, dest + mdim, 0.0f);
            }
            return result;
        }
        
        // Legacy alias for backward compatibility
        py::array_t<rf::real_t> get_proximity_importance() {
            return get_local_proximity_importance();
        }
        
        // Get top-K similar samples with feature explanations
        // Uses leaf assignments ONLY for similarity (Breiman RF proximity)
        // Uses proximity importance for explanations (which features affect similarity)
        py::tuple get_top_k_similar_with_explanations(
            rf::integer_t sample_idx, rf::integer_t k = 5, rf::integer_t n_explanations = 3
        ) {
            if (!rf_) throw std::runtime_error("Model must be fitted first");
            
            // REQUIRE leaf assignments - no fallback to proximity matrix
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error(
                    "Leaf assignments required. Set compute_leaf_assignments=True when creating the model. "
                    "This method uses exact RF proximity from leaf co-occurrence."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::integer_t mdim = rf_->get_n_features();
            rf::integer_t ntree = rf_->get_n_trees();
            if (sample_idx < 0 || sample_idx >= nsamples) throw std::runtime_error("sample_idx out of range");
            k = std::min(k, nsamples - 1);
            n_explanations = std::min(n_explanations, mdim);
            
            // Compute similarity from leaf assignments (exact RF proximity)
            std::vector<rf::real_t> similarities_raw(nsamples, 0.0f);
            for (rf::integer_t t = 0; t < ntree; ++t) {
                int16_t query_leaf = leaf_ptr[t * nsamples + sample_idx];
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    if (i != sample_idx && leaf_ptr[t * nsamples + i] == query_leaf) {
                        similarities_raw[i] += 1.0f;
                    }
                }
            }
            // Normalize by ntree (fraction of trees where samples co-occur)
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                similarities_raw[i] /= static_cast<rf::real_t>(ntree);
            }
            
            // Create normalized version (max similarity = 1.0)
            // This ensures comparable results between dense and sparse modes
            std::vector<rf::real_t> similarities_norm = similarities_raw;
            rf::real_t max_sim = 0.0f;
            for (rf::integer_t i = 0; i < nsamples; ++i) {
                if (i != sample_idx && similarities_raw[i] > max_sim) {
                    max_sim = similarities_raw[i];
                }
            }
            if (max_sim > 0.0f) {
                for (rf::integer_t i = 0; i < nsamples; ++i) {
                    similarities_norm[i] /= max_sim;
                }
            }
            
            // Get top-K by similarity (use normalized for ranking)
            std::vector<std::tuple<rf::real_t, rf::real_t, rf::integer_t>> sims;
            for (rf::integer_t j = 0; j < nsamples; ++j) {
                if (j != sample_idx) {
                    sims.push_back({similarities_norm[j], similarities_raw[j], j});
                }
            }
            std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), 
                [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
            
            py::array_t<rf::integer_t> idx_arr(k);
            py::array_t<rf::real_t> sim_raw_arr(k);
            py::array_t<rf::real_t> sim_norm_arr(k);
            auto idx_buf = idx_arr.request();
            auto sim_raw_buf = sim_raw_arr.request();
            auto sim_norm_buf = sim_norm_arr.request();
            for (rf::integer_t i = 0; i < k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = std::get<2>(sims[i]);
                static_cast<rf::real_t*>(sim_raw_buf.ptr)[i] = std::get<1>(sims[i]);
                static_cast<rf::real_t*>(sim_norm_buf.ptr)[i] = std::get<0>(sims[i]);
            }
            
            // Get explanations from proximity importance (Breiman/Cutler: which features affect similarity)
            const rf::real_t* pi = rf_->get_proximity_importance();
            py::array_t<rf::integer_t> feat_arr(n_explanations); py::array_t<rf::real_t> val_arr(n_explanations);
            auto feat_buf = feat_arr.request(); auto val_buf = val_arr.request();
            if (pi) {
                std::vector<std::pair<rf::real_t, rf::integer_t>> fi;
                for (rf::integer_t f = 0; f < mdim; ++f) fi.push_back({pi[sample_idx * mdim + f], f});
                std::partial_sort(fi.begin(), fi.begin() + n_explanations, fi.end(), [](auto& a, auto& b) { return a.first > b.first; });
                for (rf::integer_t i = 0; i < n_explanations; ++i) {
                    static_cast<rf::integer_t*>(feat_buf.ptr)[i] = fi[i].second;
                    static_cast<rf::real_t*>(val_buf.ptr)[i] = fi[i].first;
                }
            } else { 
                std::fill_n(static_cast<rf::integer_t*>(feat_buf.ptr), n_explanations, 0); 
                std::fill_n(static_cast<rf::real_t*>(val_buf.ptr), n_explanations, 0.0f); 
            }
            // Return: (indices, raw_similarities, normalized_similarities, feature_indices, feature_scores)
            return py::make_tuple(idx_arr, sim_raw_arr, sim_norm_arr, feat_arr, val_arr);
        }
        
        py::array_t<rf::real_t> get_proximity_matrix() {
            // Get the proximity matrix pointer and convert to numpy array
            const rf::dp_t* proximity_ptr = rf_->get_proximity_matrix();
            rf::integer_t nsamples = rf_->get_n_samples();
            
            py::array_t<rf::real_t> result({nsamples, nsamples});
            auto buf = result.request();
            
            if (proximity_ptr) {
                // Convert from dp_t to real_t
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                // Use element-by-element copy with bounds checking to avoid memory corruption
                rf::real_t* dest = static_cast<rf::real_t*>(buf.ptr);
                const rf::dp_t* src = proximity_ptr;
                size_t n_elements = static_cast<size_t>(nsamples) * nsamples;
                
                // Copy data immediately while holding reference to model
                // This ensures the proximity matrix vector is not moved or cleared during copy
                for (size_t i = 0; i < n_elements; ++i) {
                    dest[i] = static_cast<rf::real_t>(src[i]);
                }
            } else {
                // Low-rank mode: full matrix not available
                if (rf_->get_compute_proximity()) {
                    std::string error_msg = 
                        "Proximity matrix not available in full form. "
                        "Low-rank mode is active (use_qlora=True). "
                        "Full matrix reconstruction would require O(n²) memory and can CRASH your system! "
                        "Example: 100k samples = ~80 GB (likely to crash!). "
                        "Proximity is stored in low-rank factors (A and B). "
                        "Use get_lowrank_factors() or compute_mds_3d_from_factors() instead. "
                        "Or disable use_qlora=True for smaller datasets that can fit full matrix.";
                    
                    if (nsamples > 50000) {
                        error_msg += " ERROR: Dataset too large (" + std::to_string(nsamples) + 
                                    " samples) - reconstruction aborted to prevent system crash.";
                    }
                    
                    throw std::runtime_error(error_msg);
                } else {
                    // Fallback: initialize with identity matrix if not computed
                    std::fill(static_cast<rf::real_t*>(buf.ptr), 
                             static_cast<rf::real_t*>(buf.ptr) + nsamples * nsamples, 0.0f);
                    // Set diagonal to 1.0
                    for (rf::integer_t i = 0; i < nsamples; i++) {
                        static_cast<rf::real_t*>(buf.ptr)[i * nsamples + i] = 1.0f;
                    }
                }
            }
            
            return result;
        }
        
        py::tuple get_lowrank_factors() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_lowrank_factors()");
            rf::integer_t nsamples = rf_->get_n_samples();
            rf::dp_t* A_host = nullptr;
            rf::dp_t* B_host = nullptr;
            rf::integer_t rank = 0;
            
            bool success = rf_->get_lowrank_factors(&A_host, &B_host, &rank);
            
            if (!success) {
                throw std::runtime_error(
                    "Low-rank factors not available. "
                    "Either low-rank mode is not active (use_qlora=True required), "
                    "or factors have not been computed yet."
                );
            }
            
            // Create numpy arrays from host memory
            py::array_t<rf::dp_t> A_array(std::vector<py::ssize_t>{nsamples, rank});
            py::array_t<rf::dp_t> B_array(std::vector<py::ssize_t>{nsamples, rank});
            
            auto A_buf = A_array.request();
            auto B_buf = B_array.request();
            
            if (!A_buf.ptr || !B_host) {
                delete[] A_host;
                delete[] B_host;
                throw std::runtime_error("Failed to allocate memory for low-rank factors");
            }
            
            // Copy data
            size_t factor_size = static_cast<size_t>(nsamples) * rank;
            std::copy(A_host, A_host + factor_size, static_cast<rf::dp_t*>(A_buf.ptr));
            std::copy(B_host, B_host + factor_size, static_cast<rf::dp_t*>(B_buf.ptr));
            
            // Free host memory
            delete[] A_host;
            delete[] B_host;
            
            return py::make_tuple(A_array, B_array, rank);
        }
        
        py::array_t<rf::integer_t> get_oob_counts() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_oob_counts()");
            rf::integer_t nsamples = rf_->get_n_samples();
            const rf::integer_t* nout_ptr = rf_->get_nout_ptr();
            
            if (!nout_ptr) {
                throw std::runtime_error("OOB counts not available.");
            }
            
            py::array_t<rf::integer_t> result(nsamples);
            auto result_buf = result.request();
            rf::integer_t* result_ptr = static_cast<rf::integer_t*>(result_buf.ptr);
            std::copy(nout_ptr, nout_ptr + nsamples, result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_3d_cpu() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_3d_cpu()");
            
            // Get proximity matrix
            py::array_t<rf::dp_t> prox_array = get_proximity_matrix();
            if (prox_array.size() == 0) {
                throw std::runtime_error("Proximity matrix not available. Set compute_proximity=True when fitting.");
            }
            
            auto prox_buf = prox_array.request();
            rf::dp_t* prox_ptr = static_cast<rf::dp_t*>(prox_buf.ptr);
            rf::integer_t n_samples = static_cast<rf::integer_t>(std::sqrt(prox_array.size()));
            
            // Call C++ CPU MDS implementation (uses LAPACK)
            // Pass OOB counts for RF-GAP normalization only if RF-GAP is actually enabled
            // Note: If RF-GAP was computed as full matrix, it's already normalized by |Si| in cpu_proximity_rfgap
            // But if proximity matrix was reconstructed from low-rank factors (QLoRA), normalization is required here
            // Get OOB counts for RF-GAP normalization if available
            const rf::integer_t* oob_counts = rf_->get_nout_ptr();
            // Only normalize by OOB counts if RF-GAP is actually enabled
            // Standard proximity matrices are already normalized and don't need RF-GAP normalization
            bool use_rfgap = rf_->get_use_rfgap();
            std::vector<double> coords_3d = rf::compute_mds_3d_cpu(prox_ptr, n_samples, true, oob_counts, use_rfgap);
            
            // Convert to numpy array
            py::array_t<double> result(std::vector<py::ssize_t>{n_samples, 3});
            auto result_buf = result.request();
            double* result_ptr = static_cast<double*>(result_buf.ptr);
            std::copy(coords_3d.begin(), coords_3d.end(), result_ptr);
            
            return result;
        }
        
        py::array_t<double> compute_mds_from_factors(rf::integer_t k = 3) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling compute_mds_from_factors()");
            
            if (k < 1) {
                throw std::runtime_error("MDS dimension k must be >= 1");
            }
            
            std::vector<double> coords = rf_->compute_mds_from_factors(k);
            
            if (coords.empty()) {
                throw std::runtime_error(
                    "MDS computation failed. "
                    "Low-rank factors may not be available (use_qlora=True required), "
                    "or computation failed."
                );
            }
            
            rf::integer_t nsamples = rf_->get_n_samples();
            py::array_t<double> coords_array(std::vector<py::ssize_t>{nsamples, static_cast<py::ssize_t>(k)});
            auto coords_buf = coords_array.request();
            
            std::copy(coords.begin(), coords.end(), static_cast<double*>(coords_buf.ptr));
            
            // Check for duplicate coordinates (indicates insufficient tree coverage)
            py::module_ np = py::module_::import("numpy");
            py::module_ warnings = py::module_::import("warnings");
            py::array_t<double> rounded = np.attr("round")(coords_array, 6);
            py::array_t<double> unique_coords = np.attr("unique")(rounded, py::arg("axis")=0);
            py::ssize_t n_unique = unique_coords.request().shape[0];
            py::ssize_t n_duplicates = nsamples - n_unique;
            if (n_duplicates > 0) {
                double pct_duplicates = (static_cast<double>(n_duplicates) / nsamples) * 100.0;
                
                // Calculate recommended tree count based on sample size
                int recommended_trees;
                std::string size_category;
                if (nsamples < 500) {
                    recommended_trees = nsamples * 8;
                    size_category = "8-10× samples";
                } else if (nsamples < 2000) {
                    recommended_trees = nsamples * 6;
                    size_category = "6-8× samples";
                } else if (nsamples < 10000) {
                    recommended_trees = nsamples * 5;
                    size_category = "5-7× samples";
                } else {
                    recommended_trees = nsamples * 4;
                    size_category = "3-5× samples";
                }
                
                std::string warn_msg = "MDS coordinates have " + std::to_string(n_duplicates) + 
                    " duplicate points (" + std::to_string(static_cast<int>(pct_duplicates)) + "% of " + 
                    std::to_string(nsamples) + " samples). Only " + std::to_string(n_unique) + 
                    " unique positions will be visible. This typically indicates insufficient tree coverage " +
                    "for proximity computation. Recommend ntree ≥ " + std::to_string(recommended_trees) + 
                    " (" + size_category + ") for complete MDS coverage.";
                warnings.attr("warn")(warn_msg, py::module_::import("builtins").attr("UserWarning"));
            }
            
            return coords_array;
        }
        
        py::array_t<double> compute_mds_3d_from_factors() {
            return compute_mds_from_factors(3);
        }
        
        // Get leaf assignments (terminal node IDs) for all samples in all trees
        py::array_t<int16_t> get_leaf_assignments() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_leaf_assignments()");
            
            const int16_t* leaf_ptr = rf_->get_leaf_assignments();
            if (!leaf_ptr) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            rf::integer_t ntree = rf_->get_n_trees();
            rf::integer_t nsample = rf_->get_n_samples();
            
            py::array_t<int16_t> result({ntree, nsample});
            auto buf = result.request();
            int16_t* result_ptr = static_cast<int16_t*>(buf.ptr);
            std::copy(leaf_ptr, leaf_ptr + ntree * nsample, result_ptr);
            
            return result;
        }
        
        // Get top-K similar samples using leaf assignments (exact proximity)
        // Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity
        py::tuple get_top_k_similar(rf::integer_t query_idx, int k = 10, bool exclude_self = true) {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling get_top_k_similar()");
            
            // Lazy build inverted index on first call for optimal performance
            if (!rf_->has_inverted_leaf_index() && rf_->has_leaf_assignments()) {
                rf_->build_inverted_leaf_index();
            }
            
            auto results = rf_->get_top_k_similar(query_idx, k, exclude_self);
            
            if (results.empty()) {
                throw std::runtime_error("Leaf assignments not available. Set compute_leaf_assignments=True when creating the model.");
            }
            
            // Create output arrays
            rf::integer_t actual_k = static_cast<rf::integer_t>(results.size());
            py::array_t<rf::integer_t> top_k_indices(actual_k);
            py::array_t<rf::real_t> top_k_scores(actual_k);
            auto idx_buf = top_k_indices.request();
            auto score_buf = top_k_scores.request();
            
            for (rf::integer_t i = 0; i < actual_k; ++i) {
                static_cast<rf::integer_t*>(idx_buf.ptr)[i] = results[i].first;
                static_cast<rf::real_t*>(score_buf.ptr)[i] = results[i].second;
            }
            
            return py::make_tuple(top_k_indices, top_k_scores);
        }
        
        // Build inverted leaf index for fast top-K lookups (usually called automatically)
        void build_inverted_leaf_index() {
            if (!rf_) throw std::runtime_error("Model must be fitted before calling build_inverted_leaf_index()");
            rf_->build_inverted_leaf_index();
        }
        
        // Check if inverted leaf index is built
        bool has_inverted_leaf_index() const {
            if (!rf_) return false;
            return rf_->has_inverted_leaf_index();
        }
    };

    // Register RandomForestUnsupervised class
    // Note: compute_importance and compute_local_importance are NOT available for unsupervised mode
    // because permutation importance requires a target variable. Use proximity-based analysis instead.
    py::class_<RandomForestUnsupervised>(m, "RandomForestUnsupervised")
        .def(py::init<int, int, int, int, int, int, int, int, int, int, bool, bool, bool,
                      bool, bool, std::string, bool, float, int, int, float, bool, std::string, float, int, int, int, bool, std::string, int, bool, bool, bool, bool, int>(),
    py::arg("nsample") = 1000,
    py::arg("ntree") = 100,
    py::arg("mdim") = 10,
    py::arg("maxcat") = 10,
    py::arg("mtry") = 0,
    py::arg("maxnode") = 0,
    py::arg("minndsize") = 1,
    py::arg("ncsplit") = 25,
    py::arg("ncmax") = 25,
    py::arg("iseed") = 12345,
             py::arg("compute_proximity") = true,  // Proximity is the main output for unsupervised
             py::arg("compute_importance") = true,  // Overall importance (which features have real structure)
             py::arg("compute_local_importance") = false,  // Per-sample importance
    py::arg("use_gpu") = false,
    py::arg("use_qlora") = false,
    py::arg("quant_mode") = "nf4",
    py::arg("use_sparse") = false,
    py::arg("sparsity_threshold") = 1e-6f,
    py::arg("batch_size") = 0,
    py::arg("nodesize") = 5,
    py::arg("cutoff") = 0.01f,
             py::arg("show_progress") = true,
             py::arg("progress_desc") = "Training Random Forest Unsupervised",
             py::arg("synthetic_ratio") = 1.0f,  // Ratio of synthetic to real samples (default 1:1)
             py::arg("rank") = 100,  // Low-rank proximity matrix rank
             py::arg("power_iterations") = 30,  // Power iteration count for low-rank SVD
             py::arg("n_threads_cpu") = 0,  // Number of CPU threads (0 = auto-detect)
             py::arg("use_rfgap") = false,  // Use RF-GAP proximity
             py::arg("importance_method") = "local_imp",  // Kept for API consistency
             py::arg("clique_M") = -1,  // Kept for API consistency
             py::arg("use_casewise") = false,  // Case-wise vs non-case-wise calculations
             py::arg("compute_proximity_importance") = false,  // Terminal node change % for interpretability
             py::arg("compute_leaf_assignments") = false,  // Store leaf IDs for on-demand proximity & outlier detection
             py::arg("use_histogram") = true,  // Use histogram-based split finding (faster for large datasets >= 1000 samples)
             py::arg("n_bins") = 256)  // Number of bins for histogram (max 256)
        // NOTE: Outlier detection is ON-DEMAND only via compute_outlier_scores(mode='greedy'|'full')
        // Requires compute_leaf_assignments=True. Uses leaf assignments, NOT full proximity matrix.
        // Auto-detecting fit: accepts both dense numpy arrays and scipy sparse matrices
        .def("fit", [](RandomForestUnsupervised& self, py::object X, py::array_t<rf::real_t> sample_weight) {
            if (is_scipy_sparse(X)) {
                // Auto-detected sparse input - enable sparse mode and call fit_sparse
                self.set_use_sparse(true);
                return self.fit_sparse(X, sample_weight);
            } else {
                // Dense input - call regular fit
                py::array_t<rf::real_t> X_dense = X.cast<py::array_t<rf::real_t>>();
                return self.fit(X_dense, sample_weight);
            }
        }, py::arg("X"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit the unsupervised random forest model. Auto-detects scipy.sparse matrices and uses sparse kernels automatically.")
        .def("fit_sparse", [](RandomForestUnsupervised& self, py::object X_sparse) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, py::array_t<rf::real_t>());
        }, py::arg("X"),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix). "
           "For unsupervised learning on sparse data (e.g., user-item matrices).")
        .def("fit_sparse", [](RandomForestUnsupervised& self, py::object X_sparse, py::array_t<rf::real_t> sample_weight) {
            self.set_use_sparse(true);  // Auto-enable sparse mode
            return self.fit_sparse(X_sparse, sample_weight);
        }, py::arg("X"), py::arg("sample_weight") = py::array_t<rf::real_t>(),
           "Fit using sparse matrix input (scipy.sparse.csr_matrix) with sample weights.")
        .def("predict", &RandomForestUnsupervised::predict)
        .def("predict_proba", &RandomForestUnsupervised::predict_proba,
             "Predict class probabilities for unsupervised mode.\n\n"
             "In unsupervised mode, returns P(synthetic) and P(real) for each sample.\n"
             "Useful for imputation validation (poorly imputed data looks more 'synthetic')\n"
             "and anomaly detection (anomalies look more 'synthetic').\n\n"
             "Args:\n"
             "    X: Feature matrix (n_samples, n_features)\n\n"
             "Returns:\n"
             "    2D array of shape (n_samples, 2) with [P(synthetic), P(real)]")
        .def("get_oob_error", &RandomForestUnsupervised::get_oob_error)
        .def("save", &RandomForestUnsupervised::save, py::arg("filepath"),
             "Save trained model to file (binary format). Can be loaded with rfx.load().")
        .def("feature_importances_", &RandomForestUnsupervised::feature_importances_,
             "Get overall feature importance. With synthetic data generation, this measures "
             "which features help distinguish real from synthetic (i.e., have real structure vs noise).")
        .def("get_task_type", &RandomForestUnsupervised::get_task_type)
        .def("get_n_samples", &RandomForestUnsupervised::get_n_samples)
        .def("get_n_features", &RandomForestUnsupervised::get_n_features)
        .def("get_n_classes", &RandomForestUnsupervised::get_n_classes)
        .def("get_prototypes", &RandomForestUnsupervised::get_prototypes, py::arg("n_prototypes") = 5,
             "Get prototypes - most representative/central samples in proximity space.\n\n"
             "Finds samples with highest mean proximity to all other samples.\n"
             "These are cluster centroids in proximity space - the most 'typical' examples.\n"
             "Perfect for finding archetypal items in recommender systems.\n\n"
             "Args:\n"
             "    n_prototypes: Number of prototypes to return (default 5)\n\n"
             "Returns:\n"
             "    list: [(sample_idx, prototype_score), ...]")
        .def("get_local_importance", &RandomForestUnsupervised::get_local_importance,
             "Get local importance matrix (n_samples, n_features). With synthetic data generation, "
             "this measures which features make each sample look 'real' (structured) vs 'random'. "
             "Useful for anomaly detection and sample-level structure analysis.")
        .def("get_local_proximity_importance", &RandomForestUnsupervised::get_local_proximity_importance,
             "Get local proximity importance matrix (n_samples, n_features). "
             "Value at [i, k] = fraction of trees where permuting feature k changed terminal node for OOB sample i. "
             "Unlike local_importance, this WORKS for unsupervised - no target variable needed! "
             "Perfect for recommender system interpretability: explains WHY samples have similar proximity. "
             "Requires compute_proximity_importance=True when fitting.")
        .def("get_overall_proximity_importance", &RandomForestUnsupervised::get_overall_proximity_importance,
             "Get overall proximity importance vector (n_features,). "
             "Value at [k] = mean fraction of trees where permuting feature k changed terminal node. "
             "Aggregated across all samples. Requires compute_proximity_importance=True when fitting.")
        .def("get_top_k_similar_with_explanations", &RandomForestUnsupervised::get_top_k_similar_with_explanations,
             py::arg("sample_idx"), py::arg("k") = 5, py::arg("n_explanations") = 3,
             "Get top-K most similar samples with feature explanations for a given sample.\n\n"
             "Perfect for recommender systems: find similar items/users and explain WHY.\n\n"
             "Args:\n"
             "    sample_idx: Index of the query sample (item/user)\n"
             "    k: Number of similar samples to return (default 5)\n"
             "    n_explanations: Number of top features to explain similarity (default 3)\n\n"
             "Returns:\n"
             "    Tuple of (similar_indices, similarities, explanation_features, explanation_values)")
        .def("get_proximity_importance", &RandomForestUnsupervised::get_proximity_importance,
             "Legacy alias for get_local_proximity_importance(). Use get_local_proximity_importance() instead.")
        .def("get_proximity_matrix", &RandomForestUnsupervised::get_proximity_matrix)
        .def("get_lowrank_factors", &RandomForestUnsupervised::get_lowrank_factors)
        .def("get_oob_counts", &RandomForestUnsupervised::get_oob_counts,
             "Get the OOB counts for each sample (number of trees where sample was OOB).")
        .def("compute_mds_from_factors", &RandomForestUnsupervised::compute_mds_from_factors,
             py::arg("k") = 3,
             "Compute k-dimensional MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, k) with MDS coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        .def("compute_mds_3d_from_factors", &RandomForestUnsupervised::compute_mds_3d_from_factors,
             "Compute 3D MDS coordinates from low-rank factors (GPU ONLY, memory efficient). "
             "Returns numpy array of shape (n_samples, 3) with [x, y, z] coordinates. "
             "Requires use_gpu=True and use_qlora=True. For CPU, use compute_mds_3d_cpu() with full proximity matrix.")
        .def("compute_proximity_matrix", &RandomForestUnsupervised::get_proximity_matrix)  // Alias for compatibility
        .def("get_leaf_assignments", &RandomForestUnsupervised::get_leaf_assignments,
             "Get leaf (terminal node) assignments for all samples in all trees.\n\n"
             "Returns:\n"
             "    2D numpy array of shape (n_trees, n_samples) with terminal node IDs.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("get_top_k_similar", &RandomForestUnsupervised::get_top_k_similar,
             py::arg("query_idx"), py::arg("k") = 10, py::arg("exclude_self") = true,
             "Get top-K most similar TRAINING samples using exact leaf-based proximity.\n\n"
             "Uses inverted leaf index automatically for O(ntree * avg_samples_per_leaf) complexity.\n\n"
             "Args:\n"
             "    query_idx: Index of the query sample (must be in training set)\n"
             "    k: Number of similar samples to return (default 10)\n"
             "    exclude_self: Exclude the query sample from results (default True)\n\n"
             "Returns:\n"
             "    Tuple of (indices, similarity_scores)\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("build_inverted_leaf_index", &RandomForestUnsupervised::build_inverted_leaf_index,
             "Pre-build inverted leaf index for faster top-K lookups.\n\n"
             "Usually called automatically by get_top_k_similar() on first use.\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("has_inverted_leaf_index", &RandomForestUnsupervised::has_inverted_leaf_index,
             "Check if inverted leaf index has been built.")
        // ====================================================================
        // BREIMAN-CUTLER OUTLIER DETECTION (Unsupervised)
        // ====================================================================
        .def("compute_outlier_scores", 
             [](RandomForestUnsupervised& self, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::real_t> scores = rf->compute_outlier_scores(mode, n_anchors);
                 py::array_t<rf::real_t> result(scores.size());
                 auto buf = result.request();
                 std::copy(scores.begin(), scores.end(), static_cast<rf::real_t*>(buf.ptr));
                 return result;
             },
             py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Compute Breiman-Cutler outlier scores for all training samples.\n\n"
             "Uses the original formula: raw = N / sum(prox^2), normalized by MAD.\n"
             "Scores > 10 typically indicate outliers.\n\n"
             "Args:\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    1D array of normalized outlier scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("compute_outliers", 
             [](RandomForestUnsupervised& self, rf::integer_t k, const std::string& mode, rf::integer_t n_anchors) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 auto outliers = rf->compute_outliers(k, mode, n_anchors);
                 py::array_t<rf::integer_t> indices(k);
                 py::array_t<rf::real_t> scores(k);
                 auto idx_buf = indices.request();
                 auto score_buf = scores.request();
                 for (size_t i = 0; i < outliers.size(); ++i) {
                     static_cast<rf::integer_t*>(idx_buf.ptr)[i] = outliers[i].first;
                     static_cast<rf::real_t*>(score_buf.ptr)[i] = outliers[i].second;
                 }
                 return py::make_tuple(indices, scores);
             },
             py::arg("k") = 10, py::arg("mode") = "full", py::arg("n_anchors") = 100,
             "Get top-K outliers (convenience wrapper).\n\n"
             "Args:\n"
             "    k: Number of top outliers to return (default 10)\n"
             "    mode: 'full' (exact) or 'greedy' (approximate using anchors)\n"
             "    n_anchors: Number of anchor points for greedy mode (default 100)\n\n"
             "Returns:\n"
             "    Tuple of (indices, scores) - top-K outlier sample indices and their scores.\n\n"
             "Requires compute_leaf_assignments=True when creating the model.")
        .def("select_outlier_anchors",
             [](RandomForestUnsupervised& self, rf::integer_t n_anchors, const std::string& method) {
                 rf::RandomForest* rf = self.get_rf_ptr();
                 if (!rf) throw std::runtime_error("Model must be fitted first");
                 std::vector<rf::integer_t> anchors = rf->select_outlier_anchors(n_anchors, method);
                 py::array_t<rf::integer_t> result(anchors.size());
                 auto buf = result.request();
                 std::copy(anchors.begin(), anchors.end(), static_cast<rf::integer_t*>(buf.ptr));
                 return result;
             },
             py::arg("n_anchors") = 100, py::arg("method") = "random",
             "Select anchor points for greedy outlier detection.\n\n"
             "Args:\n"
             "    n_anchors: Number of anchors to select (default 100)\n"
             "    method: 'random' (fast) or 'diverse' (better coverage)\n\n"
             "Returns:\n"
             "    1D array of anchor sample indices.")
        .def("cleanup", [](RandomForestUnsupervised& self) {
            // Explicit cleanup of GPU memory
            // Safe to call multiple times, idempotent (cuda_cleanup() is idempotent)
            rf::cuda::cuda_cleanup();
        }, "Explicitly clean up GPU memory. Safe to call multiple times. Useful for Jupyter notebook memory management.");

    // rfviz visualization function binding - generates 2x2 grid HTML with linked brushing JavaScript
    m.def("rfviz", [](py::object rf_model, py::array_t<rf::real_t> X, py::array_t<rf::real_t> y,
                      py::object feature_names = py::none(), py::object n_clusters = py::cast(3),
                      py::object title = py::none(), py::object output_file = py::cast("rfviz.html"),
                      py::object show_in_browser = py::cast(true), py::object save_html = py::cast(true),
                      int mds_k = 3) -> py::object {
        // Use Python exec to generate 2x2 grid, then inject JavaScript
        try {
            py::object locals = py::dict();
            locals["rf_model"] = rf_model;
            locals["X"] = X;
            locals["y"] = y;
            locals["feature_names"] = feature_names;
            // Handle None for n_clusters (auto-select)
            if (n_clusters.is_none()) {
                locals["n_clusters"] = py::none();
            } else {
                locals["n_clusters"] = py::cast<int>(n_clusters);
            }
            locals["title"] = title;
            locals["output_file"] = output_file;
            locals["save_html"] = save_html;
            locals["mds_k"] = mds_k;
            
            // Execute Python code to create the 2x2 grid figure
            std::string python_code = R"PYTHON(
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

# Get model data
local_imp = rf_model.get_local_importance()

# Try to get proximity matrix (may fail for low-rank mode)
proximity = None
try:
    proximity = rf_model.get_proximity_matrix()
except:
    # Low-rank mode: proximity not available as full matrix
    # Will use GPU low-rank MDS instead
    proximity = None

# Get predictions
try:
    oob_pred = rf_model.get_oob_predictions().astype(int)
except:
    oob_pred = rf_model.predict(X).astype(int)

# Create feature names if not provided
if feature_names is None:
    feature_names = [f'Feature {i}' for i in range(X.shape[1])]

# Normalize local importance
local_imp_normalized = (local_imp - np.min(local_imp, axis=0, keepdims=True)) / (
    np.max(local_imp, axis=0, keepdims=True) - np.min(local_imp, axis=0, keepdims=True) + 1e-10)

# Get unique classes
unique_classes = np.unique(y)
n_classes = len(unique_classes)
n_samples = X.shape[0]

# Create vote matrix
vote_matrix = np.zeros((n_samples, n_classes))
for i in range(n_samples):
    pred_class = oob_pred[i]
    vote_matrix[i, pred_class] = 1.0

# Create 2x2 subplot layout
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Input Features Parallel Coordinates',
                   'Local Importance Parallel Coordinates',
                   '3D MDS Proximity Plot',
                   'Class Votes Heatmap (RAFT-style)'),
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter3d"}, {"type": "heatmap"}]],
    horizontal_spacing=0.12,
    vertical_spacing=0.22
)

# Color mapping
unique_labels = np.unique(y)
colors = {}
for i, label in enumerate(unique_labels):
    colors[label] = f'hsl({i * 360 // len(unique_labels)}, 70%, 50%)'

# 1. Top-left: Parallel coordinates of input features
for i in range(n_samples):
    fig.add_trace(
        go.Scatter(
            x=feature_names,
            y=X[i, :],
            mode='lines+markers',
            line=dict(color=colors[y[i]], width=1),
            marker=dict(size=3),
            showlegend=False,
            hovertext=f'Sample {i}',
            hovertemplate=f'<b>Sample {i}</b><br>' + 
                        '<br>'.join([f'{dim}: {val:.3f}' for dim, val in zip(feature_names, X[i, :])]) +
                        '<extra></extra>',
            selected=dict(marker=dict(size=8, color='red', opacity=1.0)),
            unselected=dict(marker=dict(size=3, opacity=0.2)),
            customdata=[[i]]
        ),
        row=1, col=1
    )

# 2. Top-right: Parallel coordinates of local importance
# Use feature_names for importance dimensions instead of generic "Imp 0", "Imp 1", etc.
imp_dimensions = feature_names  # Already defined with actual feature names
for i in range(local_imp.shape[0]):
    fig.add_trace(
        go.Scatter(
            x=imp_dimensions,
            y=local_imp_normalized[i, :],
            mode='lines+markers',
            line=dict(color=colors[y[i]], width=1),
            marker=dict(size=3),
            showlegend=False,
            hovertext=f'Sample {i}',
            hovertemplate=f'<b>Sample {i}</b><br>' + 
                        '<br>'.join([f'{feature_names[j]}: {val:.3f}' for j, val in enumerate(local_imp_normalized[i, :])]) +
                        '<extra></extra>',
            selected=dict(marker=dict(size=8, color='red', opacity=1.0)),
            unselected=dict(marker=dict(size=3, opacity=0.2)),
            customdata=[[i]]
        ),
        row=1, col=2
    )

# 3. Bottom-left: 3D MDS proximity plot
# Try GPU low-rank MDS first (if available), then fall back to CPU MDS or sampling
# Use mds_k dimensions (default: 3 for 3D visualization)
n_samples = X.shape[0]
coords_kd = None

# Try GPU low-rank MDS first (memory efficient for large datasets)
gpu_mds_error = None
try:
    coords_kd_gpu = rf_model.compute_mds_from_factors(k=mds_k)
    if coords_kd_gpu is not None and coords_kd_gpu.shape[0] == n_samples:
        coords_kd = coords_kd_gpu
        print(f"   ✓ Using GPU low-rank MDS (memory efficient, k={mds_k} dimensions)")
except Exception as e:
    # GPU low-rank MDS not available, fall back to CPU methods
    gpu_mds_error = str(e)
    print(f"   GPU low-rank MDS not available: {e}")
    print(f"   Falling back to CPU MDS...")

if coords_kd is None:
    # Try CPU MDS (C++ implementation) - currently only supports 3D
    if mds_k == 3:
        try:
            coords_3d = rf_model.compute_mds_3d_cpu()
            coords_kd = coords_3d
            print(f"   ✓ Using CPU MDS (C++ implementation)")
        except Exception as e2:
            # CPU MDS failed - raise error instead of falling back to sklearn
            gpu_err_str = gpu_mds_error if gpu_mds_error else "unknown error"
            raise RuntimeError(f"Cannot compute MDS: GPU low-rank MDS failed ({gpu_err_str}) and CPU MDS failed ({e2}). "
                              "Ensure proximity matrix is available (set compute_proximity=True) or use GPU low-rank MDS (use_gpu=True, use_qlora=True).")
    else:
        # For k != 3, CPU MDS doesn't support it yet
        raise RuntimeError(f"Cannot compute MDS: GPU low-rank MDS failed ({e}) and CPU MDS only supports k=3 (requested k={mds_k}). "
                          "Use GPU low-rank MDS (use_gpu=True, use_qlora=True) for arbitrary k, or set mds_k=3 for CPU MDS.")

# Extract dimensions for 3D visualization
# For k > 3: Use first 3 for x,y,z, encode additional dimensions via color/size/opacity
# For k < 3: Pad with zeros
if coords_kd.shape[1] >= 3:
    coords_3d = coords_kd[:, :3]
    # Store extra dimensions for visual encoding
    extra_dims = coords_kd[:, 3:] if coords_kd.shape[1] > 3 else None
    if extra_dims is not None:
        print(f"   Note: Using first 3 dimensions for x,y,z, encoding {extra_dims.shape[1]} extra dimensions via color/size/opacity")
elif coords_kd.shape[1] == 2:
    # Pad with zeros for 2D -> 3D
    coords_3d = np.column_stack([coords_kd, np.zeros(n_samples)])
    extra_dims = None
    print(f"   Note: Using 2D MDS, padded to 3D for visualization")
elif coords_kd.shape[1] == 1:
    # Pad with zeros for 1D -> 3D
    coords_3d = np.column_stack([coords_kd, np.zeros(n_samples), np.zeros(n_samples)])
    extra_dims = None
    print(f"   Note: Using 1D MDS, padded to 3D for visualization")
else:
    raise RuntimeError(f"Invalid MDS dimensions: {coords_kd.shape[1]}")

# Check for NaN/inf values in MDS coordinates and warn if any are found
valid_mask = np.all(np.isfinite(coords_3d), axis=1)
n_invalid = n_samples - np.sum(valid_mask)
if n_invalid > 0:
    print(f"   Warning: {n_invalid} samples have invalid MDS coordinates (NaN/inf) and will not be displayed")

# Note: Duplicate coordinate warning is now handled in compute_mds_from_factors()

# Automatic cluster selection based on WCSS percentage decrease (elbow method)
auto_select_clusters = (n_clusters is None or n_clusters <= 0)
if auto_select_clusters:
    print(f"   Automatically selecting optimal number of clusters...")
    max_k = min(15, n_samples // 10, len(unique_classes) * 3)  # Reasonable upper bound
    min_k = max(2, len(unique_classes))  # At least as many clusters as classes
    
    if max_k < min_k:
        n_clusters = min_k
        print(f"   Using {n_clusters} clusters (minimum required)")
    else:
        # Calculate WCSS for different k values
        wcss = []
        k_range = range(min_k, max_k + 1)
        
        for k in k_range:
            kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            kmeans_test.fit(coords_3d)
            wcss.append(kmeans_test.inertia_)
        
        # Find elbow point based on percentage decrease threshold
        # Look for the point where percentage decrease drops below threshold
        threshold_percent = 5.0  # 5% decrease threshold
        optimal_k = min_k
        
        for i in range(1, len(wcss)):
            if wcss[i-1] > 0:
                percent_decrease = ((wcss[i-1] - wcss[i]) / wcss[i-1]) * 100.0
                if percent_decrease < threshold_percent:
                    optimal_k = k_range[i-1]
                    break
            optimal_k = k_range[i]  # Use last k if no elbow found
        
        n_clusters = optimal_k
        print(f"   ✓ Optimal clusters: {n_clusters} (elbow method, threshold: {threshold_percent}% decrease)")
        print(f"   WCSS values: {[f'{w:.2f}' for w in wcss]}")

# Prepare extra dimension encodings if available
dim4_color = None
dim5_size = None
dim6_opacity = None
if extra_dims is not None and extra_dims.shape[1] > 0:
    # Normalize extra dimensions for visual encoding
    if extra_dims.shape[1] >= 1:
        # Dimension 4: encode as color (normalize to 0-1 range)
        dim4 = extra_dims[:, 0]
        dim4_min, dim4_max = dim4.min(), dim4.max()
        if dim4_max > dim4_min:
            dim4_color = (dim4 - dim4_min) / (dim4_max - dim4_min)
        else:
            dim4_color = np.ones(n_samples) * 0.5
    
    if extra_dims.shape[1] >= 2:
        # Dimension 5: encode as marker size (normalize to reasonable range)
        dim5 = extra_dims[:, 1]
        dim5_min, dim5_max = dim5.min(), dim5.max()
        if dim5_max > dim5_min:
            dim5_size = 3 + 7 * (dim5 - dim5_min) / (dim5_max - dim5_min)  # Size range: 3-10
        else:
            dim5_size = np.ones(n_samples) * 6.5
    
    if extra_dims.shape[1] >= 3:
        # Dimension 6: encode as opacity (normalize to 0.3-1.0 range)
        dim6 = extra_dims[:, 2]
        dim6_min, dim6_max = dim6.min(), dim6.max()
        if dim6_max > dim6_min:
            dim6_opacity = 0.3 + 0.7 * (dim6 - dim6_min) / (dim6_max - dim6_min)  # Opacity range: 0.3-1.0
        else:
            dim6_opacity = np.ones(n_samples) * 0.65

# Apply KMeans clustering with selected/auto-determined number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(coords_3d)

for cluster_id in range(n_clusters):
    mask = cluster_labels == cluster_id
    cluster_coords = coords_3d[mask]
    cluster_y = y[mask]
    cluster_indices = np.where(mask)[0]
    
    # Get extra dimension values for this cluster
    cluster_dim4 = dim4_color[mask] if dim4_color is not None else None
    cluster_dim5 = dim5_size[mask] if dim5_size is not None else None
    cluster_dim6 = dim6_opacity[mask] if dim6_opacity is not None else None
    
    for label in unique_labels:
        label_mask = cluster_y == label
        if np.any(label_mask):
            label_coords = cluster_coords[label_mask]
            label_indices = cluster_indices[label_mask]
            
            # Build hover text with extra dimensions
            hover_dims = '<b>Coordinates:</b> (%{x:.2f}, %{y:.2f}, %{z:.2f})'
            if extra_dims is not None and extra_dims.shape[1] > 0:
                hover_dims += '<br>'
                for dim_idx in range(min(extra_dims.shape[1], 3)):
                    dim_name = f'Dim {dim_idx + 4}'
                    # customdata[0] is index, customdata[1+] are extra dimensions
                    hover_dims += f'<br><b>{dim_name}:</b> %{{customdata[{dim_idx + 1}]:.2f}}'
            
            # Build marker dict with extra dimension encodings
            marker_dict = dict(
                line=dict(color='black', width=1)
            )
            
            # Color: use dimension 4 if available, otherwise use cluster_id
            if cluster_dim4 is not None:
                marker_dict['color'] = cluster_dim4[label_mask]
                marker_dict['colorscale'] = 'Viridis'
                marker_dict['cmin'] = 0
                marker_dict['cmax'] = 1
                # Only add colorbar to first trace to avoid duplicates
                if cluster_id == 0 and label == unique_labels[0]:
                    marker_dict['colorbar'] = dict(title="Dim 4", x=1.02, len=0.5)
            else:
                marker_dict['color'] = cluster_id
                marker_dict['colorscale'] = 'Viridis'
            
            # Size: use dimension 5 if available, otherwise default size
            if cluster_dim5 is not None:
                marker_dict['size'] = cluster_dim5[label_mask]
            else:
                marker_dict['size'] = 5
            
            # Opacity: use dimension 6 if available, otherwise default
            if cluster_dim6 is not None:
                marker_dict['opacity'] = cluster_dim6[label_mask]
            else:
                marker_dict['opacity'] = 0.7
            
            # Prepare customdata for hover (extra dimensions)
            customdata_list = []
            for idx in label_indices:
                customdata_row = [idx]
                if extra_dims is not None:
                    for dim_idx in range(min(extra_dims.shape[1], 3)):
                        customdata_row.append(extra_dims[idx, dim_idx])
                customdata_list.append(customdata_row)
            
            fig.add_trace(
                go.Scatter3d(
                    x=label_coords[:, 0],
                    y=label_coords[:, 1],
                    z=label_coords[:, 2],
                    mode='markers',
                    marker=marker_dict,
                    name=f'Cluster {cluster_id}, Label {label}',
                    showlegend=True,
                    text=[f'Cluster {cluster_id}' for _ in label_indices],
                    hovertemplate='<b>Cluster:</b> %{text}<br>' +
                                f'<b>Label:</b> {label}<br>' +
                                hover_dims + '<extra></extra>',
                    customdata=customdata_list
                ),
                row=2, col=1
            )

# 4. Bottom-right: Class Votes Heatmap
class_names = [f'Class {i}' for i in unique_classes]
fig.add_trace(
    go.Heatmap(
        z=vote_matrix.T,
        x=[f'S{i}' for i in range(n_samples)],
        y=class_names,
        colorscale='Blues',
        colorbar=dict(title="Vote Score", x=1.15),
        hovertemplate='<b>Sample:</b> %{x}<br>' +
                    '<b>Class:</b> %{y}<br>' +
                    '<b>Vote Score:</b> %{z:.2f}<extra></extra>',
        showscale=True
    ),
    row=2, col=2
)

# Update layout
title_str = title if title is not None else "Random Forest Visualization Dashboard (RAFT-style)"
fig.update_layout(
    title=dict(text=title_str, font=dict(size=24, color='black')),
    height=1400,
    width=1800,
    showlegend=True,
    hovermode='closest',
    dragmode='select',
    clickmode='event+select',
    selectdirection='d',
    selectionrevision=0  # Initialize with no selections
)

# Update axes
fig.update_xaxes(title_text="Feature", row=1, col=1, tickangle=-45)
fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
fig.update_xaxes(title_text="Local Importance Feature", row=1, col=2, tickangle=-45)
fig.update_yaxes(title_text="Local Importance Value", row=1, col=2)
fig.update_xaxes(title_text="Sample Index", row=2, col=2, tickangle=-90, nticks=min(20, n_samples))
fig.update_yaxes(title_text="Class", row=2, col=2)

fig.update_scenes(
    xaxis_title='MDS Dimension 1',
    yaxis_title='MDS Dimension 2',
    zaxis_title='MDS Dimension 3',
    aspectmode='cube',
    dragmode='orbit',  # Default: rotation (Shift+drag also rotates)
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    ),
    xaxis=dict(showspikes=False),
    yaxis=dict(showspikes=False),
    zaxis=dict(showspikes=False),
    row=2, col=1
)
# Note: Selections for 3D plots are controlled via JavaScript and layout.selectionrevision
# They cannot be initialized through Python Scene properties
)PYTHON";
            
            py::dict globals_dict;
            py::exec(python_code, globals_dict, locals);
            py::object fig = locals["fig"];
            
            // Generate HTML and inject JavaScript
            std::string output_file_str = py::cast<std::string>(output_file);
            // Generate indices file name from output file name
            std::string indices_file_str = output_file_str;
            size_t html_pos = indices_file_str.find(".html");
            if (html_pos != std::string::npos) {
                indices_file_str.replace(html_pos, 5, "_selected_indices.json");
            } else {
                indices_file_str += "_selected_indices.json";
            }
            
            if (py::cast<bool>(save_html)) {
                py::object html_str = fig.attr("to_html")(py::arg("include_plotlyjs") = py::cast(true));
                
                // JavaScript for linked brushing and reset
                std::string linked_brushing_js = R"(
        <script>
        // Linked brushing with reset functionality (R or Escape)
        (function() {
            let gd = null;
            let isUpdating = false;
            
            function initLinkedBrushing() {
                if (typeof Plotly === 'undefined') {
                    setTimeout(initLinkedBrushing, 100);
                return;
                }
                
                gd = document.getElementsByClassName('plotly-graph-div')[0];
                if (!gd || !gd.data) {
                    setTimeout(initLinkedBrushing, 100);
                    return;
                }
                
                function extractSampleIdx(custom) {
                    if (custom === undefined || custom === null) return null;
                    if (typeof custom === 'number') return custom;
                    if (Array.isArray(custom) && custom.length > 0) {
                        const first = custom[0];
                        if (typeof first === 'number') return first;
                        if (Array.isArray(first) && first.length > 0 && typeof first[0] === 'number') {
                            return first[0];
                        }
                    }
                    return null;
                }
                
                function getSampleIdxFromTrace(trace) {
                    if (!trace.customdata || !Array.isArray(trace.customdata) || trace.customdata.length === 0) return null;
                    return extractSampleIdx(trace.customdata[0]);
                }
                
                function getSelectedIndices(eventData) {
                    const selectedIndices = new Set();
                    if (eventData && eventData.points) {
                        eventData.points.forEach(point => {
                            const sampleIdx = extractSampleIdx(point.customdata);
                            if (sampleIdx !== null) {
                                selectedIndices.add(sampleIdx);
                            } else {
                                const curveNumber = point.curveNumber;
                                if (curveNumber !== undefined && gd.data[curveNumber]) {
                                    const traceSampleIdx = getSampleIdxFromTrace(gd.data[curveNumber]);
                                    if (traceSampleIdx !== null) {
                                        selectedIndices.add(traceSampleIdx);
                                    }
                                }
                            }
                        });
                    }
                    return Array.from(selectedIndices);
                }
                
                function clearAllSelections() {
                    // Force clear even if updating - user explicitly requested clear via keyboard/button
                    // But check if already in the middle of clearing to avoid race conditions
                    const wasUpdating = isUpdating;
                    isUpdating = true;
                    
                    try {
                        if (!gd || !gd.data) {
                            isUpdating = wasUpdating;
                                    return;
                        }
                        
                        console.log('🗑️ Clearing all selections...');
                        
                        // Separate 2D and 3D traces - they require different clearing methods
                        const scatter2DTraces = [];
                        const scatter3DTraces = [];
                        const markerUpdates = [];
                        const markerTraceIndices = [];
                        
                        gd.data.forEach((trace, traceIdx) => {
                            if (trace.type === 'scatter') {
                                scatter2DTraces.push(traceIdx);
                            } else if (trace.type === 'scatter3d') {
                                scatter3DTraces.push(traceIdx);
                                // Prepare marker reset for 3D traces - get actual point count
                                const nPoints = trace.x ? trace.x.length : (trace.y ? trace.y.length : (trace.z ? trace.z.length : (trace.customdata ? trace.customdata.length : 0)));
                                if (nPoints > 0) {
                                    markerTraceIndices.push(traceIdx);
                                    // Reset ALL markers to default state - this is critical for clearing visual selection
                                    markerUpdates.push({
                                        'marker.size': Array(nPoints).fill(5),
                                        'marker.opacity': Array(nPoints).fill(0.7),
                                        'marker.line.width': Array(nPoints).fill(1),
                                        'marker.line.color': Array(nPoints).fill('black')
                                    });
                                }
                            }
                        });
                        
                        console.log('Found', scatter2DTraces.length, '2D traces and', scatter3DTraces.length, '3D traces to clear');
                        
                        const promises = [];
                        
                        // Step 1: Clear selectedpoints for 2D traces (parallel coordinates)
                        if (scatter2DTraces.length > 0) {
                            promises.push(Plotly.restyle(gd, {selectedpoints: null}, scatter2DTraces).catch((err) => {
                                console.warn('Error clearing 2D selectedpoints:', err);
                            }));
                        }
                        
                        // Step 2: Try to clear selectedpoints for 3D traces too (even though not officially supported,
                        // Plotly might store it internally from applySelectionToAll)
                        if (scatter3DTraces.length > 0) {
                            const clearSelectedPoints = scatter3DTraces.map(() => null);
                            promises.push(Plotly.restyle(gd, {selectedpoints: clearSelectedPoints}, scatter3DTraces).catch(() => {
                                // Ignore errors - selectedpoints might not be valid for 3D, that's okay
                            }));
                        }
                        
                        // Step 3: Reset 3D marker properties (this clears visual highlighting immediately)
                        if (markerTraceIndices.length > 0) {
                            // Convert markerUpdates array format to Plotly.restyle format
                            // Plotly.restyle expects: { 'marker.size': [array1, array2, ...], 'marker.opacity': [array1, array2, ...] }
                            const restyleUpdates = {};
                            ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                            });
                            
                            promises.push(Plotly.restyle(gd, restyleUpdates, markerTraceIndices).catch((err) => {
                                console.warn('Error resetting 3D markers:', err);
                            }));
                        }
                        
                        // Step 4: Clear layout selections (critical for 3D plots and lasso selections)
                        // This must clear ALL axis selections for ALL subplots to handle lasso selections
                        const layoutUpdate = {
                            'selectionrevision': Date.now()
                        };
                        
                        // Clear scene selections (for 3D MDS plot) - must be done via relayout
                        if (scatter3DTraces.length > 0) {
                            // Clear all possible selection properties for 3D scene
                            layoutUpdate['scene.selections'] = [];
                            layoutUpdate['scene.xaxis.selections'] = [];
                            layoutUpdate['scene.yaxis.selections'] = [];
                            layoutUpdate['scene.zaxis.selections'] = [];
                            layoutUpdate['scene.xaxis.selectionrevision'] = Date.now();
                            layoutUpdate['scene.yaxis.selectionrevision'] = Date.now();
                            layoutUpdate['scene.zaxis.selectionrevision'] = Date.now();
                            layoutUpdate['scene.selectionrevision'] = Date.now();
                        }
                        
                        // Clear selections from ALL 2D axes for ALL subplots (for lasso selections on parallel coordinates)
                        // Need to handle xaxis, yaxis, xaxis2, yaxis2, etc. for multi-subplot layouts
                        Object.keys(gd.layout).forEach(key => {
                            if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
                                layoutUpdate[key + '.selections'] = [];
                                layoutUpdate[key + '.selectionrevision'] = Date.now();
                            }
                        });
                        
                        // Also explicitly clear common subplot axes (in case Object.keys doesn't catch them all)
                        for (let i = 1; i <= 10; i++) {
                            const suffix = i === 1 ? '' : i.toString();
                            layoutUpdate['xaxis' + suffix + '.selections'] = [];
                            layoutUpdate['yaxis' + suffix + '.selections'] = [];
                            layoutUpdate['xaxis' + suffix + '.selectionrevision'] = Date.now();
                            layoutUpdate['yaxis' + suffix + '.selectionrevision'] = Date.now();
                        }
                        
                        // Clear selections from the full layout object
                        if (gd.layout.selections) {
                            layoutUpdate.selections = [];
                        }
                        
                        // Step 5: Execute layout update AND selectedpoints clear TOGETHER in parallel
                        // This ensures both are cleared simultaneously
                        promises.push(Plotly.relayout(gd, layoutUpdate).catch((err) => {
                            console.warn('Error in relayout:', err);
                        }));
                        
                        // Step 6: Wait for all updates, then force a redraw AND clear selectedpoints again
                        Promise.all(promises).then(() => {
                            // After layout clears, also ensure selectedpoints are cleared for ALL 2D traces
                            // Do this again after layout update to ensure synchronization
                            const finalPromises = [];
                            if (scatter2DTraces.length > 0) {
                                finalPromises.push(Plotly.restyle(gd, {selectedpoints: null}, scatter2DTraces));
                            }
                            
                            // Reset 3D markers AGAIN after layout update to ensure they're cleared
                            // This is necessary because layout updates can sometimes restore marker states
                            if (markerTraceIndices.length > 0) {
                                // Convert markerUpdates array format to Plotly.restyle format
                                const restyleUpdates = {};
                                ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                    restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                                });
                                finalPromises.push(Plotly.restyle(gd, restyleUpdates, markerTraceIndices));
                            }
                            
                            // Force redraw to ensure all changes are applied and visual state is synced
                            finalPromises.push(Plotly.redraw(gd));
                            
                            Promise.all(finalPromises).then(() => {
                                // Double-check: clear any remaining selections by resetting selectionrevision again
                                Plotly.relayout(gd, {
                                    'selectionrevision': Date.now(),
                                    'scene.selectionrevision': Date.now()
                                }).then(() => {
                                    // FINAL: Reset 3D markers ONE MORE TIME after all relayouts are done
                                    // This ensures markers are cleared even if relayout restored them
                                    if (markerTraceIndices.length > 0) {
                                        // Convert markerUpdates array format to Plotly.restyle format
                                        const restyleUpdates = {};
                                        ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                            restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                                        });
                                        Plotly.restyle(gd, restyleUpdates, markerTraceIndices).then(() => {
                                            Plotly.redraw(gd).then(() => {
                                                isUpdating = false;
                                                console.log('✅ All selections cleared successfully (including lasso and 3D MDS)');
                                            }).catch(() => {
                                                isUpdating = false;
                                                console.log('✅ All selections cleared successfully (including lasso and 3D MDS)');
                                            });
                                        }).catch(() => {
                                            isUpdating = false;
                                            console.log('✅ All selections cleared successfully (including lasso and 3D MDS)');
                                        });
                                    } else {
                                        Plotly.redraw(gd).then(() => {
                                            isUpdating = false;
                                            console.log('✅ All selections cleared successfully (including lasso and 3D MDS)');
                                        }).catch(() => {
                                            isUpdating = false;
                                            console.log('✅ All selections cleared successfully (including lasso and 3D MDS)');
                                        });
                                    }
                                }).catch((err) => {
                                    console.error('Error in final relayout:', err);
                                    // Still reset markers even if relayout failed
                                    if (markerTraceIndices.length > 0) {
                                        const restyleUpdates = {};
                                        ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                            restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                                        });
                                        Plotly.restyle(gd, restyleUpdates, markerTraceIndices).catch(() => {});
                                    }
                                    isUpdating = false;
                                });
                            }).catch((err) => {
                                console.error('Error in final redraw/restyle:', err);
                                // Still reset markers even if redraw failed
                                if (markerTraceIndices.length > 0) {
                                    const restyleUpdates = {};
                                    ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                        restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                                    });
                                    Plotly.restyle(gd, restyleUpdates, markerTraceIndices).catch(() => {});
                                }
                                isUpdating = false;
                            });
                        }).catch((err) => {
                            console.error('Error clearing selections:', err);
                            // Still try to reset markers even if layout update failed
                            if (markerTraceIndices.length > 0) {
                                const restyleUpdates = {};
                                ['marker.size', 'marker.opacity', 'marker.line.width', 'marker.line.color'].forEach(prop => {
                                    restyleUpdates[prop] = markerUpdates.map(update => update[prop]);
                                });
                                Plotly.restyle(gd, restyleUpdates, markerTraceIndices).catch(() => {});
                            }
                            if (scatter2DTraces.length > 0) {
                                Plotly.restyle(gd, {selectedpoints: null}, scatter2DTraces).catch(() => {});
                            }
                            Plotly.redraw(gd).catch(() => {});
                            isUpdating = false;
                        });
                        
                        // Safety timeout: reset isUpdating flag after 2 seconds if something goes wrong
                        setTimeout(() => {
                            if (isUpdating) {
                                console.warn('⚠️ Clear operation timed out, resetting isUpdating flag');
                                isUpdating = false;
                            }
                        }, 2000);
                    } catch (e) {
                        console.error('Error in clearAllSelections:', e);
                        isUpdating = false;
                    }
                }
                
                function applySelectionToAll(selectedIndices) {
                    if (isUpdating) return;
                    isUpdating = true;
                    
                    try {
                        if (!gd || !gd.data) {
                            isUpdating = false;
            return;
                        }
                        
                        const selectedpointsArray = [];
                        const markerUpdates = [];
                        const traceIndices = [];
                        
                        gd.data.forEach((trace, traceIdx) => {
                            if (trace.type === 'scatter' || trace.type === 'scatter3d') {
                                if (trace.customdata && Array.isArray(trace.customdata)) {
                                    traceIndices.push(traceIdx);
                                    
                                    if (trace.type === 'scatter3d') {
                                        // For 3D scatter: find which points to highlight
                                        const pointIndices = [];
                                        const nPoints = trace.x ? trace.x.length : (trace.customdata ? trace.customdata.length : 0);
                                        const sizeArray = Array(nPoints).fill(5);  // Default size
                                        const opacityArray = Array(nPoints).fill(0.7);  // Default opacity
                                        const lineWidthArray = Array(nPoints).fill(1);  // Default line width
                                        
                                        for (let pointIdx = 0; pointIdx < trace.customdata.length; pointIdx++) {
                                            const sampleIdx = extractSampleIdx(trace.customdata[pointIdx]);
                                            if (sampleIdx !== null && selectedIndices.includes(sampleIdx)) {
                                                pointIndices.push(pointIdx);
                                                // Highlight selected points: larger, more opaque, red border
                                                sizeArray[pointIdx] = 10;
                                                opacityArray[pointIdx] = 1.0;
                                                lineWidthArray[pointIdx] = 3;
                                            }
                                        }
                                        
                                        selectedpointsArray.push(pointIndices.length > 0 ? pointIndices : null);
                                        // Update marker properties for visual highlighting
                                        markerUpdates.push({
                                            'marker.size': sizeArray,
                                            'marker.opacity': opacityArray,
                                            'marker.line.width': lineWidthArray,
                                            'marker.line.color': pointIndices.length > 0 ? 
                                                sizeArray.map((s, i) => s > 5 ? 'red' : 'black') : 
                                                Array(nPoints).fill('black')
                                        });
                                    } else {
                                        // For 2D scatter (parallel coordinates)
                                        const sampleIdx = getSampleIdxFromTrace(trace);
                                        if (sampleIdx !== null && selectedIndices.includes(sampleIdx)) {
                                            const numPoints = trace.x ? trace.x.length : (trace.y ? trace.y.length : trace.customdata.length);
                                            if (numPoints > 0) {
                                                const allPointIndices = Array.from({length: numPoints}, (_, i) => i);
                                                selectedpointsArray.push(allPointIndices);
                                                markerUpdates.push({});  // Use default marker styling for 2D
                                            } else {
                                                selectedpointsArray.push([0]);
                                                markerUpdates.push({});
                                            }
                                        } else {
                                            selectedpointsArray.push(null);
                                            markerUpdates.push({});
                                        }
                                    }
                                }
                            }
                        });
                        
                        if (traceIndices.length > 0) {
                            // Apply both selectedpoints and marker updates
                            const updates = {selectedpoints: selectedpointsArray};
                            
                            // Merge marker updates for 3D plots
                            const has3DUpdates = markerUpdates.some(m => Object.keys(m).length > 0);
                            if (has3DUpdates) {
                                markerUpdates.forEach((markerUpdate, idx) => {
                                    if (Object.keys(markerUpdate).length > 0) {
                                        Object.keys(markerUpdate).forEach(key => {
                                            if (!updates[key]) updates[key] = [];
                                            updates[key][idx] = markerUpdate[key];
                                        });
                                    }
                                });
                            }
                            
                            Plotly.restyle(gd, updates, traceIndices).then(() => {
                                isUpdating = false;
                                console.log('✓ Selection applied to', traceIndices.length, 'traces for', selectedIndices.length, 'samples');
                            }).catch((err) => {
                                console.error('Error applying selection:', err);
                                isUpdating = false;
                            });
                        } else {
                            isUpdating = false;
                        }
                    } catch (e) {
                        console.error('Error in applySelectionToAll:', e);
                        isUpdating = false;
                    }
                }
                
                gd.on('plotly_selected', function(eventData) {
                    if (isUpdating || !eventData) return;
                    const selectedIndices = getSelectedIndices(eventData);
                    if (selectedIndices.length > 0) {
                        applySelectionToAll(selectedIndices);
                    } else {
                        clearAllSelections();
                    }
                });
                
                gd.on('plotly_click', function(eventData) {
                    if (isUpdating || !eventData) return;
                    const clickedIndices = getSelectedIndices(eventData);
                    if (clickedIndices.length === 0) return;
                    
                    const currentSelected = new Set();
                    gd.data.forEach(trace => {
                        if (trace.selectedpoints !== null && trace.selectedpoints !== undefined) {
                            if (Array.isArray(trace.selectedpoints) && trace.selectedpoints.length > 0) {
                                const sampleIdx = getSampleIdxFromTrace(trace);
                                if (sampleIdx !== null) currentSelected.add(sampleIdx);
                            }
                        }
                    });
                    
                    clickedIndices.forEach(idx => {
                        if (currentSelected.has(idx)) {
                            currentSelected.delete(idx);
                        } else {
                            currentSelected.add(idx);
                        }
                    });
                    
                    applySelectionToAll(Array.from(currentSelected));
                });
                
                // Improved keyboard shortcuts for reset (R or Escape) - works in iframe context
                function handleKeyDown(event) {
                    // Check if it's R or Escape key
                    if ((event.key === 'r' || event.key === 'R' || event.key === 'Escape') && 
                        !event.ctrlKey && !event.metaKey && !event.altKey) {
                        // Don't trigger if typing in input fields
                        const target = event.target || event.srcElement;
                        if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
                        return;
                        }
                        
                        event.preventDefault();
                        event.stopPropagation();
                        event.stopImmediatePropagation();
                        console.log('⌨️ Keyboard shortcut detected:', event.key, 'from:', target ? target.tagName : 'unknown');
                        clearAllSelections();
                        return false;
                    }
                }
                
                // Attach to multiple targets for maximum compatibility
                // Use capture phase to catch events early
                if (document) {
                    document.addEventListener('keydown', handleKeyDown, true);
                }
                if (window) {
                    window.addEventListener('keydown', handleKeyDown, true);
                }
                if (document.body) {
                    document.body.addEventListener('keydown', handleKeyDown, true);
                }
                
                // Also attach to parent window if in iframe (common in Jupyter)
                try {
                    if (window.parent && window.parent !== window) {
                        window.parent.addEventListener('keydown', handleKeyDown, true);
                    }
                } catch (e) {
                    // Cross-origin iframe, can't access parent
                }
                
                // Also attach to the plotly graph div and 3D scene for direct interaction
                if (gd) {
                    // Make plot div focusable so it can receive keyboard events
                    if (gd.setAttribute) {
                        gd.setAttribute('tabindex', '0');
                    }
                    
                    gd.on('plotly_doubleclick', function(eventData) {
                        console.log('🖱️ Double-click detected');
                        clearAllSelections();
                    });
                    
                    // Ensure Escape key works even when focus is on the plot
                    gd.addEventListener('keydown', function(event) {
                        if (event.key === 'Escape' || event.key === 'r' || event.key === 'R') {
                            event.preventDefault();
                            event.stopPropagation();
                            event.stopImmediatePropagation();
                            console.log('⌨️ Keyboard shortcut detected on plot:', event.key);
                            clearAllSelections();
                            return false;
                        }
                    }, true);
                    
                    // Ensure plot can receive focus for keyboard events
                    gd.addEventListener('click', function() {
                        if (gd.focus) {
                            gd.focus();
                        }
                        // Also ensure keyboard handlers are still active
                        console.log('📌 Plot clicked, keyboard handlers should be active');
                    }, false);
                    
                    // Also attach keyboard handlers to 3D scene element specifically
                    setTimeout(function() {
                        const scenes = gd.querySelectorAll('.scene');
                        scenes.forEach(function(scene) {
                            if (scene.setAttribute) {
                                scene.setAttribute('tabindex', '0');
                            }
                            scene.addEventListener('keydown', function(event) {
                                if (event.key === 'Escape' || event.key === 'r' || event.key === 'R') {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    event.stopImmediatePropagation();
                                    console.log('⌨️ Keyboard shortcut detected on 3D scene:', event.key);
                                    clearAllSelections();
                                    return false;
                                }
                            }, true);
                            
                            // Make scene focusable on click
                            scene.addEventListener('click', function() {
                                if (scene.focus) {
                                    scene.focus();
                                }
                                console.log('📌 3D scene clicked, keyboard handlers should be active');
                            }, false);
                        });
                        console.log('✅ Attached keyboard handlers to', scenes.length, '3D scene(s)');
                    }, 500);  // Wait for scene to render
                }
                
                console.log('✅ Keyboard shortcuts enabled: Press R or Escape to clear all selections');
                console.log('💡 Tip: Click on the plot to focus it, then press R or Escape');
                console.log('🎯 3D MDS Plot: Ctrl+Alt+mouse drag to lasso select');
                
                // Minimal initialization - Python already sets selectionrevision=0
                // Don't clear after render - let it initialize naturally
                gd.on('plotly_afterplot', function() {
                    // Only clear if there are actual scene selections (not marker highlighting)
                    if (gd.layout && gd.layout.scene) {
                        const hasSelections = gd.layout.scene.selections && gd.layout.scene.selections.length > 0;
                        if (hasSelections) {
                            Plotly.relayout(gd, {
                                'scene.selections': [],
                                'scene.selectionrevision': Date.now()
                            }).catch(() => {});
                        }
                    }
                });
                
                console.log('✓ Linked brushing enabled: selections sync across all plots');
                console.log('✓ Press R or Escape to clear all selections (or double-click plot)');
                console.log('✓ Scroll wheel: Zoom in/out on 3D plot (works anywhere in plot area)');
                
                // Ensure scroll wheel works for zooming on 3D scene - works on entire plot area
                if (gd) {
                    // Enable wheel zoom on the entire graph div (not just on points)
                    const graphDiv = gd;
                    
                    // Add wheel event listener to the graph div itself
                    graphDiv.addEventListener('wheel', function(event) {
                        // Check if over the 3D scene
                        const rect = graphDiv.getBoundingClientRect();
                        const x = event.clientX - rect.left;
                        const y = event.clientY - rect.top;
                        
                        // Find which subplot is being hovered over
                        const scenes = graphDiv.querySelectorAll('.scene');
                        if (scenes.length > 0) {
                            // Likely over a 3D scene - allow zoom
                            event.preventDefault();
                            
                            // Get current camera settings
                            const update = {};
                            const camera = gd.layout.scene ? gd.layout.scene.camera : null;
                            
                            if (camera) {
                                const zoomFactor = event.deltaY > 0 ? 1.1 : 0.9; // Zoom out on scroll down, in on scroll up
                                const eye = camera.eye || {x: 1.5, y: 1.5, z: 1.5};
                                
                                update['scene.camera.eye.x'] = eye.x * zoomFactor;
                                update['scene.camera.eye.y'] = eye.y * zoomFactor;
                                update['scene.camera.eye.z'] = eye.z * zoomFactor;
                                
                                Plotly.relayout(gd, update);
                            }
                        }
                    }, { passive: false });
                    
                    // Also ensure Plotly's built-in wheel handler works
                    if (gd.layout && gd.layout.scene) {
                        Plotly.relayout(gd, {
                            'scene.dragmode': 'orbit',  // Default rotation mode
                            'scene.camera.eye.x': 1.5,
                            'scene.camera.eye.y': 1.5,
                            'scene.camera.eye.z': 1.5
                        }).then(() => {
                            console.log('✓ 3D scene configured: scroll wheel zoom enabled');
                            console.log('✓ 3D plot controls: Click points to select, drag to rotate');
                        });
                    }
                }
                
                // Add toggle button for selection mode (lasso-like selection)
                let selectionModeActive = false;
                let lassoPath = [];
                let isDrawingLasso = false;
                
                // Lasso mode button removed - functionality still works with Ctrl+Alt+Drag
                
                // Custom lasso selection for 3D plot - requires Ctrl+Alt+drag
                function startLasso(event) {
                    // Only activate if Ctrl+Alt are both pressed
                    if (!(event.ctrlKey && event.altKey)) return;
                    isDrawingLasso = true;
                    lassoPath = [];
                    const rect = gd.getBoundingClientRect();
                    lassoPath.push({x: event.clientX - rect.left, y: event.clientY - rect.top});
                }
                
                function updateLasso(event) {
                    // Only continue if Ctrl+Alt are both pressed
                    if (!isDrawingLasso || !(event.ctrlKey && event.altKey)) {
                        if (isDrawingLasso) {
                            // Keys released, end lasso
                            endLasso(event);
                        }
                        return;
                    }
                    const rect = gd.getBoundingClientRect();
                    lassoPath.push({x: event.clientX - rect.left, y: event.clientY - rect.top});
                }
                
                function endLasso(event) {
                    if (!isDrawingLasso) return;
                    isDrawingLasso = false;
                    const rect = gd.getBoundingClientRect();
                    lassoPath.push({x: event.clientX - rect.left, y: event.clientY - rect.top});
                    
                    // Find points within lasso path (point-in-polygon test)
                    const selectedIndices = new Set();
                    
                    // Get 3D scene traces
                    gd.data.forEach(trace => {
                        if (trace.type === 'scatter3d') {
                            // Project 3D points to 2D screen coordinates
                            const scene = gd._fullLayout.scene;
                            if (!scene) return;
                            
                            const x = trace.x || [];
                            const y = trace.y || [];
                            const z = trace.z || [];
                            const customdata = trace.customdata || [];
                            
                            for (let i = 0; i < x.length; i++) {
                                // Convert 3D to 2D using Plotly's projection
                                const point3d = [x[i], y[i], z[i]];
                                const projected = Plotly.Fx.inbox(point3d, scene);
                                
                                if (projected && isPointInPolygon(projected[0], projected[1], lassoPath)) {
                                    const sampleIdx = extractSampleIdx(customdata[i]);
                                    if (sampleIdx !== null) {
                                        selectedIndices.add(sampleIdx);
                                    }
                                }
                            }
                        }
                    });
                    
                    if (selectedIndices.size > 0) {
                        applySelectionToAll(Array.from(selectedIndices));
                    }
                    
                    lassoPath = [];
                }
                
                // Point-in-polygon test (ray casting algorithm)
                function isPointInPolygon(x, y, polygon) {
                    let inside = false;
                    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
                        const xi = polygon[i].x, yi = polygon[i].y;
                        const xj = polygon[j].x, yj = polygon[j].y;
                        const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                        if (intersect) inside = !inside;
                    }
                    return inside;
                }
                
                // Enable Ctrl+Alt+drag lasso functionality for 3D plot
                if (gd) {
                    gd.addEventListener('mousedown', startLasso, true);  // Capture phase
                    gd.addEventListener('mousemove', updateLasso, true);
                    gd.addEventListener('mouseup', endLasso, true);
                    gd.addEventListener('mouseleave', endLasso, true);  // Handle mouse leaving plot
                    gd.addEventListener('keyup', function(event) {
                        // If Ctrl or Alt key released while drawing, end lasso
                        if (isDrawingLasso && (event.key === 'Control' || event.key === 'Alt' || event.key === 'Meta')) {
                            endLasso(event);
                        }
                    }, true);
                }
                
                console.log('✓ Ctrl+Alt+mouse drag: Lasso selection on 3D MDS plot');
                
                // Add explicit wheel event handler for 3D plot if needed
                if (gd) {
                    gd.on('plotly_wheel', function(eventData) {
                        // Wheel events should automatically zoom in Plotly 3D scenes
                        // This ensures they're not blocked
                        if (eventData && eventData.points && eventData.points.length > 0) {
                            const point = eventData.points[0];
                            if (point.data && point.data.type === 'scatter3d') {
                                // Zoom is handled automatically by Plotly
                                return true;
                            }
                        }
                    });
                }
                
                // Add save button
                function getCurrentSelectedIndices() {
                    const selected = new Set();
                    if (!gd || !gd.data) return selected;
                    
                    gd.data.forEach(trace => {
                        if (trace.selectedpoints !== null && trace.selectedpoints !== undefined) {
                            if (Array.isArray(trace.selectedpoints) && trace.selectedpoints.length > 0) {
                                const sampleIdx = getSampleIdxFromTrace(trace);
                                if (sampleIdx !== null) selected.add(sampleIdx);
                            }
                        }
                    });
                    return Array.from(selected).sort((a, b) => a - b);
                }
                
                function saveSelectedIndices() {
                    const selectedIndices = getCurrentSelectedIndices();
                    if (selectedIndices.length === 0) {
                        alert('No points selected. Please select points in the plot first.');
                        return;
                    }
                    
                    // Indices file name (set from C++ code)
                    const indicesFile = ')" + indices_file_str + R"(';
                    
                    // Create JSON data
                    const jsonData = JSON.stringify(selectedIndices, null, 2);
                    
                    // Create download link with full path to save in notebook directory
                    const blob = new Blob([jsonData], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const filename = indicesFile.split('/').pop().split('\\\\').pop(); // Handle both / and \\
                    a.download = filename;
                    
                    // Try to set download attribute to ensure it saves to the right location
                    // Note: Browser security may still send to Downloads folder, but filename will be correct
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    console.log('✓ Saved ' + selectedIndices.length + ' selected indices to ' + filename);
                    alert('✅ Saved ' + selectedIndices.length + ' selected indices to ' + filename + '\\n\\n💡 The file should be in your Downloads folder or notebook directory');
                    alert('💡 Re-run the auto-load cell in your notebook to load the DataFrame');
                }
                
                // Create a container for buttons (works in Jupyter embedded HTML)
                // Set plotly container to relative positioning so absolute children work
                gd.style.position = 'relative';
                
                // Create clear button
                const clearButton = document.createElement('button');
                clearButton.id = 'rfviz-clear-button';
                clearButton.innerHTML = 'Clear Selection';
                clearButton.style.cssText = 'position: absolute; top: 70px; right: 20px; z-index: 10000; ' +
                    'background-color: #f44336; color: white; border: none; padding: 12px 24px; ' +
                    'font-size: 14px; font-weight: bold; border-radius: 5px; cursor: pointer; ' +
                    'box-shadow: 0 4px 6px rgba(0,0,0,0.3);';
                clearButton.onclick = clearAllSelections;
                clearButton.title = 'Clear all selections (keyboard: R or Escape)';
                gd.appendChild(clearButton);
                
                // Create save button
                const saveButton = document.createElement('button');
                saveButton.id = 'rfviz-save-button';
                saveButton.innerHTML = 'Save Selected Subset';
                saveButton.style.cssText = 'position: absolute; top: 20px; right: 20px; z-index: 10000; ' +
                    'background-color: #4CAF50; color: white; border: none; padding: 12px 24px; ' +
                    'font-size: 14px; font-weight: bold; border-radius: 5px; cursor: pointer; ' +
                    'box-shadow: 0 4px 6px rgba(0,0,0,0.3);';
                saveButton.onclick = saveSelectedIndices;
                saveButton.title = 'Save currently selected sample indices to JSON file';
                gd.appendChild(saveButton);
                
                // Create instructions overlay for 3D MDS plot
                const instructionsDiv = document.createElement('div');
                instructionsDiv.id = 'rfviz-3d-instructions';
                instructionsDiv.innerHTML = '<div style="background: rgba(0,0,0,0.8); color: white; padding: 15px; border-radius: 8px; font-size: 12px; line-height: 1.6; max-width: 280px;">' +
                    '<strong style="color: #4CAF50; font-size: 14px;">3D MDS Plot Instructions:</strong><br>' +
                    '- <strong>Rotate:</strong> Drag (or Shift+drag)<br>' +
                    '- <strong>Zoom:</strong> Scroll wheel (works anywhere)<br>' +
                    '- <strong>Click Select:</strong> Click individual points<br>' +
                    '- <strong>Pan:</strong> Ctrl + drag<br>' +
                    '- <strong>Clear:</strong> Press R or Escape<br>' +
                    '<br><em style="color: #FFC107;">Selections sync across all 4 plots!</em>' +
                    '</div>';
                instructionsDiv.style.cssText = 'position: absolute; bottom: 20px; left: 20px; z-index: 10000; ' +
                    'cursor: pointer; transition: opacity 0.3s;';
                
                // Make instructions toggleable
                let instructionsVisible = true;
                instructionsDiv.onclick = function() {
                    instructionsVisible = !instructionsVisible;
                    instructionsDiv.style.opacity = instructionsVisible ? '1' : '0.3';
                };
                
                gd.appendChild(instructionsDiv);
                
                // Auto-hide after 10 seconds
                setTimeout(function() {
                    if (instructionsVisible) {
                        instructionsDiv.style.opacity = '0.3';
                    }
                }, 10000);
            }
            
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    setTimeout(initLinkedBrushing, 300);
                });
            } else {
                setTimeout(initLinkedBrushing, 300);
            }
        })();
        </script>
    )";
                
                // Inject JavaScript before closing body tag
                std::string html_str_cpp = py::cast<std::string>(html_str);
                size_t body_pos = html_str_cpp.find("</body>");
                if (body_pos != std::string::npos) {
                    html_str_cpp.insert(body_pos, linked_brushing_js);
                } else {
                    html_str_cpp += linked_brushing_js;
                }
                
                // Write to file
                py::object open_func = py::module_::import("builtins").attr("open");
                py::object file = open_func(py::cast(output_file_str), py::cast("w"), py::arg("encoding") = py::cast("utf-8"));
                file.attr("write")(py::cast(html_str_cpp));
                file.attr("close")();
            }
            
            // Show in browser if requested
            if (py::cast<bool>(show_in_browser)) {
                try {
                    py::object webbrowser = py::module_::import("webbrowser");
                    py::object os_module = py::module_::import("os");
                    py::object os_path = os_module.attr("path");
                    py::object abs_path = os_path.attr("abspath")(py::cast(output_file_str));
                    py::object file_url = py::cast("file://" + py::cast<std::string>(abs_path));
                    webbrowser.attr("open")(file_url);
                } catch (...) {
                    // Ignore browser opening errors
                }
            }
            
            return fig;
        } catch (py::error_already_set& e) {
            throw std::runtime_error("Error generating rfviz HTML: " + std::string(e.what()));
        }
    }, py::arg("rf_model"), py::arg("X"), py::arg("y"),
       py::arg("feature_names") = py::none(), py::arg("n_clusters") = py::cast(3),
       py::arg("title") = py::none(), py::arg("output_file") = py::cast("rfviz.html"),
       py::arg("show_in_browser") = py::cast(true), py::arg("save_html") = py::cast(true),
       py::arg("mds_k") = 3,
       "Create interactive Random Forest visualization with 2x2 grid and linked brushing");

    // GPU memory management utilities for Jupyter notebook safety
    // Check if CUDA is available
    m.def("cuda_is_available", []() -> bool {
        // Wrap in try-catch to prevent crashes in Jupyter
        try {
            return rf::cuda::cuda_is_available();
        } catch (...) {
            // If any exception occurs, return false safely
            return false;
        }
    }, "Check if CUDA/GPU is available. Returns True if CUDA-capable GPU is detected.");

#ifdef CUDA_FOUND
    // Check if enough GPU memory available
    m.def("check_gpu_memory", [](size_t size_mb) -> bool {
        // Ultra-safe: Only check via CudaConfig (no direct CUDA runtime calls)
        // This avoids triggering CUDA initialization that can crash Jupyter kernels
        try {
            // Check if CudaConfig has been initialized (means CUDA context exists)
            if (!rf::cuda::CudaConfig::instance().is_initialized()) {
                return false;  // Not initialized yet - safe to return false
            }
            
            // If initialized, it's safe to query memory (context exists)
            size_t available_mb = rf::cuda::CudaConfig::instance().get_available_memory() / (1024 * 1024);
            return available_mb >= size_mb;
        } catch (...) {
            // Catch everything - don't crash
            return false;
        }
    }, py::arg("size_mb"), "Check if enough GPU memory is available (in MB). Returns False if CUDA unavailable or insufficient memory. Safe to call anytime.");

    // Print GPU memory status
    m.def("print_gpu_memory_status", []() {
        // Ultra-safe: Only check via CudaConfig (no direct CUDA runtime calls)
        // This avoids triggering CUDA initialization that can crash Jupyter kernels
        try {
            // Check if CudaConfig has been initialized (means CUDA context exists)
            bool config_initialized = rf::cuda::CudaConfig::instance().is_initialized();
            
            if (!config_initialized) {
                // Suppress messages when context is not initialized - it's expected after cleanup
                // Users can check GPU memory manually if needed, but messages are not spammed
                return;
            }
            
            // If initialized, it's safe to query memory (context exists)
            size_t total_mb = rf::cuda::CudaConfig::instance().get_total_memory() / (1024 * 1024);
            size_t free_mb = rf::cuda::CudaConfig::instance().get_available_memory() / (1024 * 1024);
            size_t used_mb = total_mb - free_mb;
            
            // Use Python print instead of std::cout to avoid stream conflicts
            py::module_ builtins = py::module_::import("builtins");
            builtins.attr("print")("\n🖥️  GPU MEMORY INFORMATION");
            builtins.attr("print")("==================================================");
            builtins.attr("print")("📊 GPU Memory:");
            builtins.attr("print")(py::str("   Total: {} MB").format(total_mb));
            builtins.attr("print")(py::str("   Available: {} MB").format(free_mb));
            builtins.attr("print")(py::str("   Used: {} MB").format(used_mb));
            builtins.attr("print")();
            } catch (...) {
            // Catch everything - don't crash, just print a safe message
            // Use Python print instead of std::cout to avoid stream conflicts
            py::module_ builtins = py::module_::import("builtins");
            builtins.attr("print")("GPU memory status unavailable (CUDA context not initialized or error occurred)");
        }
    }, "Print current GPU memory usage status (total, used, free in MB). Safe to call anytime.");
#endif

    // Explicit GPU cleanup function
    m.def("gpu_cleanup", []() {
        rf::cuda::cuda_cleanup();
    }, "Explicitly clean up GPU memory. Safe to call multiple times. Can be used in context managers for Jupyter notebook safety.");

    // Clear GPU cache
    m.def("clear_gpu_cache", []() {
        // Clear any cached GPU data and free unused memory
        rf::cuda::cuda_cleanup();
        
        // Note: cudaDeviceReset() too aggressive for notebooks, so just cleanup global state
    }, "Clear GPU cache and free unused memory. Useful between notebook cells to prevent memory accumulation.");

    // Reset CUDA device (for stuck contexts)
    m.def("reset_cuda_device", []() {
            rf::cuda::cuda_reset_device();
    }, "Forcefully reset CUDA device to free stuck contexts. Use when 'CUDA device busy or unavailable' errors occur.");

    // Load Iris dataset (UCI ML repository)
    m.def("load_iris", []() -> py::tuple {
        rf::Dataset dataset = rf::DataLoader::load_iris();
        py::array_t<rf::real_t> X({dataset.n_samples, dataset.n_features});
        auto X_buf = X.mutable_unchecked<2>();
        for (rf::integer_t i = 0; i < dataset.n_samples; ++i) {
            for (rf::integer_t j = 0; j < dataset.n_features; ++j) {
                X_buf(i, j) = dataset.X[i * dataset.n_features + j];
            }
        }
        py::array_t<rf::integer_t> y({dataset.n_samples});
        auto y_buf = y.mutable_unchecked<1>();
        for (rf::integer_t i = 0; i < dataset.n_samples; ++i) {
            y_buf(i) = dataset.y_int[i];
        }
        return py::make_tuple(X, y);
    }, "Load Iris dataset (UCI ML repository)",
       "Returns (X, y) tuple where X is (n_samples, n_features) array and y is (n_samples,) array of class labels");

    // Load Wine dataset (UCI ML repository)
    m.def("load_wine", []() -> py::tuple {
        rf::Dataset dataset = rf::DataLoader::load_wine();
        py::array_t<rf::real_t> X({dataset.n_samples, dataset.n_features});
        auto X_buf = X.mutable_unchecked<2>();
        for (rf::integer_t i = 0; i < dataset.n_samples; ++i) {
            for (rf::integer_t j = 0; j < dataset.n_features; ++j) {
                X_buf(i, j) = dataset.X[i * dataset.n_features + j];
            }
        }
        py::array_t<rf::integer_t> y({dataset.n_samples});
        auto y_buf = y.mutable_unchecked<1>();
        for (rf::integer_t i = 0; i < dataset.n_samples; ++i) {
            y_buf(i) = dataset.y_int[i];
        }
        return py::make_tuple(X, y);
    }, "Load Wine dataset (UCI ML repository)",
       "Returns (X, y) tuple where X is (n_samples, n_features) array and y is (n_samples,) array of class labels");

    // Confusion matrix helper
    m.def("confusion_matrix", [](py::array_t<rf::integer_t> y_true, py::array_t<rf::integer_t> y_pred) -> py::array_t<rf::integer_t> {
        auto y_true_buf = y_true.unchecked<1>();
        auto y_pred_buf = y_pred.unchecked<1>();
        rf::integer_t n_samples = y_true.shape(0);
        
        // Find number of classes
        rf::integer_t n_classes = 0;
        for (rf::integer_t i = 0; i < n_samples; ++i) {
            if (y_true_buf(i) >= n_classes) n_classes = y_true_buf(i) + 1;
            if (y_pred_buf(i) >= n_classes) n_classes = y_pred_buf(i) + 1;
        }
        
        // Create confusion matrix
        py::array_t<rf::integer_t> conf_mat({n_classes, n_classes});
        auto conf_buf = conf_mat.mutable_unchecked<2>();
        for (rf::integer_t i = 0; i < n_classes; ++i) {
            for (rf::integer_t j = 0; j < n_classes; ++j) {
                conf_buf(i, j) = 0;
            }
        }
        
        // Fill confusion matrix
        for (rf::integer_t i = 0; i < n_samples; ++i) {
            rf::integer_t true_class = y_true_buf(i);
            rf::integer_t pred_class = y_pred_buf(i);
            if (true_class >= 0 && pred_class >= 0) {
                conf_buf(true_class, pred_class)++;
            }
        }
        
        return conf_mat;
    }, "Compute confusion matrix",
       py::arg("y_true"), py::arg("y_pred"));

    // Classification report is now implemented in pure Python below to avoid
    // circular import crashes in Jupyter notebooks. See the py::exec section.

    // Version info
    m.attr("__version__") = "1.0.0";
#ifdef CUDA_FOUND
    m.attr("__cuda_enabled__") = true;
#else
    m.attr("__cuda_enabled__") = false;
#endif
    m.attr("__quantization_enabled__") = true;
    
    // Import and expose notebook helper functions + enable auto-storage
    // Following XGBoost pattern: embed helpers directly and auto-enable model preservation
    // Wrap in try-catch to prevent module initialization crashes
    try {
        // Pass module object to exec context to avoid circular import
        py::dict exec_globals = py::globals();
        exec_globals["_rfx_module"] = m;
        py::exec(R"(
import sys
import os
from pathlib import Path
import importlib.util

# EMBEDDED NOTEBOOK HELPERS - Auto-storage enabled automatically
# This follows XGBoost's pattern: models are preserved automatically across cells

# Module-level storage to prevent garbage collection crashes (XGBoost-style)
_MODEL_STORAGE = []

def store_model(model):
    """Store a model in module-level storage to prevent garbage collection crashes."""
    if model not in _MODEL_STORAGE:
        _MODEL_STORAGE.append(model)
    return model

def clear_models():
    """Clear all stored models (use with caution - may trigger crashes)."""
    _MODEL_STORAGE.clear()

# Pure Python classification_report to avoid circular import crash in Jupyter
def classification_report(y_true, y_pred):
    """
    Generate classification report with precision, recall, F1-score per class.
    
    This is a pure Python implementation that avoids the circular import issue
    that causes kernel crashes in Jupyter notebooks.
    
    Args:
        y_true: Ground truth labels (array-like of integers)
        y_pred: Predicted labels (array-like of integers)
    
    Returns:
        str: Formatted classification report
    """
    import numpy as np
    
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    
    # Compute confusion matrix
    cm = _rfx_module.confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    lines = []
    lines.append("\nClassification Report:")
    lines.append("=" * 58)
    lines.append(f"{'Class':>10s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s} {'Support':>12s}")
    lines.append("-" * 58)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        lines.append(f"{i:>10d} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f} {support:>12d}")
    
    lines.append("")
    return "\n".join(lines)

# Expose classification_report in the module
_rfx_module.classification_report = classification_report

# Helper function to convert column names to indices
def _convert_categorical_features(categorical_features, X):
    """
    Convert categorical_features from column names (strings) to indices (integers).
    
    Args:
        categorical_features: None, list of column names (str), or list of indices (int)
        X: Input data (numpy array or pandas DataFrame)
    
    Returns:
        None or list of integer indices
    """
    if categorical_features is None:
        return None
    
    # Check if X is a pandas DataFrame with column names
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            # X has column names - convert string names to indices
            result = []
            for feat in categorical_features:
                if isinstance(feat, str):
                    # String column name - find its index
                    if feat in X.columns:
                        result.append(X.columns.get_loc(feat))
                    else:
                        raise ValueError(f"Column '{feat}' not found in DataFrame")
                elif isinstance(feat, int):
                    # Already an integer index
                    result.append(feat)
                else:
                    raise ValueError(f"categorical_features must contain strings (column names) or integers (indices), got {type(feat)}")
            return result
        else:
            # X is a numpy array - categorical_features must be indices
            result = []
            for feat in categorical_features:
                if isinstance(feat, int):
                    result.append(feat)
                elif isinstance(feat, str):
                    raise ValueError(f"categorical_features contains string '{feat}', but X is not a pandas DataFrame. Use integer indices instead.")
                else:
                    raise ValueError(f"categorical_features must contain integers (indices), got {type(feat)}")
            return result
    except ImportError:
        # pandas not available - categorical_features must be indices
        result = []
        for feat in categorical_features:
            if isinstance(feat, int):
                result.append(feat)
            else:
                raise ValueError(f"categorical_features must contain integers (indices). To use column names, install pandas.")
        return result

# Expose helper in the module
_rfx_module._convert_categorical_features = _convert_categorical_features

# Auto-storage: Monkey-patch constructors to auto-store models (XGBoost-style)
def _enable_auto_storage():
    """Enable automatic model storage - models preserved across cells automatically."""
    # Use module object passed from C++ to avoid circular import deadlock
    rf = _rfx_module
    
    # Store original constructors
    original_classifier_init = rf.RandomForestClassifier.__init__
    original_regressor_init = rf.RandomForestRegressor.__init__
    original_unsupervised_init = rf.RandomForestUnsupervised.__init__
    
    # Monkey-patch constructors to auto-store models immediately after creation
    def auto_store_init_classifier(self, *args, **kwargs):
        # Translate old parameter names to new ones (backward compatibility)
        if 'gpu_growth_mode' in kwargs:
            del kwargs['gpu_growth_mode']  # Old parameter removed
        if 'gpu_parallel_mode0' in kwargs:
            # gpu_parallel_mode0 is now automatically determined from batch_size, so ignore this parameter
            del kwargs['gpu_parallel_mode0']
        result = original_classifier_init(self, *args, **kwargs)
        store_model(self)  # Auto-store on creation (XGBoost pattern)
        return result
    
    def auto_store_init_regressor(self, *args, **kwargs):
        # Translate old parameter names to new ones (backward compatibility)
        if 'gpu_growth_mode' in kwargs:
            del kwargs['gpu_growth_mode']  # Old parameter removed
        if 'gpu_parallel_mode0' in kwargs:
            # gpu_parallel_mode0 is now automatically determined from batch_size, so ignore this parameter
            del kwargs['gpu_parallel_mode0']
        result = original_regressor_init(self, *args, **kwargs)
        store_model(self)  # Auto-store on creation
        return result
    
    def auto_store_init_unsupervised(self, *args, **kwargs):
        # Translate old parameter names to new ones (backward compatibility)
        if 'gpu_growth_mode' in kwargs:
            del kwargs['gpu_growth_mode']  # Old parameter removed
        if 'gpu_parallel_mode0' in kwargs:
            # gpu_parallel_mode0 is now automatically determined from batch_size, so ignore this parameter
            del kwargs['gpu_parallel_mode0']
        result = original_unsupervised_init(self, *args, **kwargs)
        store_model(self)  # Auto-store on creation
        return result
    
    # Monkey-patch fit methods to ensure storage after fitting (double-safety)
    original_classifier_fit = rf.RandomForestClassifier.fit
    original_regressor_fit = rf.RandomForestRegressor.fit
    original_unsupervised_fit = rf.RandomForestUnsupervised.fit
    
    def auto_store_fit_classifier(self, *args, **kwargs):
        result = original_classifier_fit(self, *args, **kwargs)
        store_model(self)  # Ensure stored after fit
        return result
    
    def auto_store_fit_regressor(self, *args, **kwargs):
        result = original_regressor_fit(self, *args, **kwargs)
        store_model(self)  # Ensure stored after fit
        return result
    
    def auto_store_fit_unsupervised(self, *args, **kwargs):
        result = original_unsupervised_fit(self, *args, **kwargs)
        store_model(self)  # Ensure stored after fit
        return result
    
    # Apply patches
    rf.RandomForestClassifier.__init__ = auto_store_init_classifier
    rf.RandomForestRegressor.__init__ = auto_store_init_regressor
    rf.RandomForestUnsupervised.__init__ = auto_store_init_unsupervised
    
    rf.RandomForestClassifier.fit = auto_store_fit_classifier
    rf.RandomForestRegressor.fit = auto_store_fit_regressor
    rf.RandomForestUnsupervised.fit = auto_store_fit_unsupervised

# Enable auto-storage immediately (XGBoost-style: automatic model preservation)
_enable_auto_storage()

# Try to load full notebook helpers if available (for display_rfviz_html, etc.)
try:
    import RFX
    rfx_file = RFX.__file__
    rfx_dir = os.path.dirname(rfx_file)
    helpers_path = os.path.join(rfx_dir, 'rfx_notebook_helpers.py')
    
    # Try multiple locations
    if not os.path.exists(helpers_path):
        parent_dir = os.path.dirname(rfx_dir)
        helpers_path = os.path.join(parent_dir, 'rfx_notebook_helpers.py')
    
    if not os.path.exists(helpers_path):
        python_dir = os.path.join(os.path.dirname(rfx_dir), 'python')
        helpers_path = os.path.join(python_dir, 'rfx_notebook_helpers.py')
    
    if not os.path.exists(helpers_path):
        for path in sys.path:
            test_path = os.path.join(path, 'rfx_notebook_helpers.py')
            if os.path.exists(test_path):
                helpers_path = test_path
                break
    
    # If found, import it for additional functions
    if os.path.exists(helpers_path):
        spec = importlib.util.spec_from_file_location("rfx_notebook_helpers", helpers_path)
        nb_helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nb_helpers)
        
        # Expose additional functions if available
        try:
            display_rfviz_html = nb_helpers.display_rfviz_html
            load_rfviz_selection = nb_helpers.load_rfviz_selection
        except AttributeError:
            # Functions not available, create dummy ones
            def display_rfviz_html(*args, **kwargs):
                raise ImportError("display_rfviz_html requires IPython/pandas. Install for notebook support.")
            def load_rfviz_selection(*args, **kwargs):
                raise ImportError("load_rfviz_selection requires IPython/pandas. Install for notebook support.")
    else:
        # Create dummy functions
        def display_rfviz_html(*args, **kwargs):
            raise ImportError("rfx_notebook_helpers.py not found. Install IPython/pandas for notebook support.")
        def load_rfviz_selection(*args, **kwargs):
            raise ImportError("rfx_notebook_helpers.py not found. Install IPython/pandas for notebook support.")
except Exception as e:
    # Create dummy functions that raise helpful errors
    def display_rfviz_html(*args, **kwargs):
        raise ImportError(f"RFX.notebook_helpers not available: {e}. Install IPython/pandas for notebook support.")
    def load_rfviz_selection(*args, **kwargs):
        raise ImportError(f"RFX.notebook_helpers not available: {e}. Install IPython/pandas for notebook support.")

# Auto-storage is now enabled! Models will be automatically preserved across cells (XGBoost-style)
)", exec_globals);
    } catch (py::error_already_set& e) {
        // Don't let Python errors crash module initialization
        // Auto-storage is nice-to-have, but module must load even if it fails
        // Silently continue - models will still work, just without auto-storage
        // Error is automatically cleared when exception is caught
    } catch (const std::exception& e) {
        // C++ exceptions during Python exec - also safe to ignore
        // Module will still work without auto-storage
    } catch (...) {
        // Any other errors - ignore to prevent module initialization crash
    }
    
    // ============================================================================
    // IMPUTATION FUNCTIONS - DISABLED (using Python-only rfx_impute.py instead)
    // ============================================================================
    // C++ bindings removed. Use Python module rfx_impute.py for imputation:
    //   from rfx_impute import rfx_impute_rough, rfx_impute_rand, rfx_impute_proximity
    // ============================================================================
        
        // Make functions available on module by directly exposing them
        // The Python exec already created them in the module dict, so just expose them
        try {
            py::object display_func = m.attr("display_rfviz_html");
            py::object load_func = m.attr("load_rfviz_selection");
            
            m.attr("display_rfviz_html") = display_func;
            m.attr("load_rfviz_selection") = load_func;
        } catch (...) {
        // Functions not available, create dummy functions
        m.def("display_rfviz_html", [](const std::string&, int = 1900, int = 1500) -> bool {
            throw std::runtime_error("display_rfviz_html not available. Install IPython/pandas for notebook support.");
        });
        m.def("load_rfviz_selection", [](const std::string&) -> py::object {
            throw std::runtime_error("load_rfviz_selection not available. Install IPython/pandas for notebook support.");
        });
    }
}



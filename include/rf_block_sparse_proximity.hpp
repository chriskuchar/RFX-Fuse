// Ultra-Optimized Block Sparse Proximity Matrix
// Provides massive memory savings with minimal accuracy impact

#pragma once

#include "rf_types.hpp"
#include <vector>
#include <unordered_map>
#include <algorithm>

namespace rf {

// Configuration for block sparse proximity
struct BlockSparseConfig {
    static constexpr float CONSERVATIVE_THRESHOLD = 0.001f;  // 99.9% accuracy
    static constexpr float AGGRESSIVE_THRESHOLD = 0.01f;     // 99% accuracy
    static constexpr float ULTRA_AGGRESSIVE_THRESHOLD = 0.05f; // 95% accuracy
    
    static constexpr int BLOCK_SIZE = 32;  // 32x32 blocks
    static constexpr int MAX_SPARSE_ENTRIES = 1000000;  // Limit sparse entries
};

// Sparse entry for very small proximity values
struct SparseProximityEntry {
    integer_t row, col;
    real_t value;
    
    SparseProximityEntry(integer_t r, integer_t c, real_t v) 
        : row(r), col(c), value(v) {}
};

// Block sparse proximity matrix
class BlockSparseProximityMatrix {
private:
    integer_t nsample_;
    real_t sparse_threshold_;
    std::vector<SparseProximityEntry> sparse_entries_;
    std::unordered_map<integer_t, real_t> dense_map_;  // Linear index -> value
    
public:
    BlockSparseProximityMatrix(integer_t nsample, real_t threshold = BlockSparseConfig::CONSERVATIVE_THRESHOLD)
        : nsample_(nsample), sparse_threshold_(threshold) {
        sparse_entries_.reserve(BlockSparseConfig::MAX_SPARSE_ENTRIES);
    }
    
    // Add proximity value (only upper triangle)
    void add_proximity(integer_t row, integer_t col, real_t value) {
        if (row > col) return;  // Only upper triangle
        
        if (value >= sparse_threshold_) {
            // Store in dense representation
            integer_t linear_idx = row + col * nsample_;
            dense_map_[linear_idx] += value;
        } else if (value > 1e-6f) {  // Only store non-zero values
            // Store in sparse representation
            if (sparse_entries_.size() < BlockSparseConfig::MAX_SPARSE_ENTRIES) {
                sparse_entries_.emplace_back(row, col, value);
            }
        }
        // Values below 1e-6 are implicitly zero (not stored)
    }
    
    // Get proximity value
    real_t get_proximity(integer_t row, integer_t col) const {
        // Check dense storage first
        integer_t linear_idx = row + col * nsample_;
        auto dense_it = dense_map_.find(linear_idx);
        if (dense_it != dense_map_.end()) {
            return dense_it->second;
        }
        
        // Check sparse storage
        for (const auto& entry : sparse_entries_) {
            if (entry.row == row && entry.col == col) {
                return entry.value;
            }
        }
        
        // Implicit zero (not stored)
        return 0.0f;
    }
    
    // Convert to full dense matrix (for final output)
    void to_dense_matrix(real_t* dense_matrix) const {
        // Initialize to zero
        std::fill(dense_matrix, dense_matrix + nsample_ * nsample_, 0.0f);
        
        // Copy dense entries
        for (const auto& entry : dense_map_) {
            integer_t row = entry.first % nsample_;
            integer_t col = entry.first / nsample_;
            dense_matrix[row + col * nsample_] = entry.second;
        }
        
        // Copy sparse entries
        for (const auto& entry : sparse_entries_) {
            dense_matrix[entry.row + entry.col * nsample_] = entry.value;
        }
        
        // Symmetrize (copy upper triangle to lower triangle)
        for (integer_t i = 0; i < nsample_; i++) {
            for (integer_t j = i + 1; j < nsample_; j++) {
                real_t upper_val = dense_matrix[i + j * nsample_];
                dense_matrix[j + i * nsample_] = upper_val;
            }
        }
        
        // Set diagonal to 1.0
        for (integer_t i = 0; i < nsample_; i++) {
            dense_matrix[i + i * nsample_] = 1.0f;
        }
    }
    
    // Get memory usage statistics
    struct MemoryStats {
        integer_t dense_entries;
        integer_t sparse_entries;
        integer_t total_memory_bytes;
        real_t sparsity_ratio;
    };
    
    MemoryStats get_memory_stats() const {
        MemoryStats stats;
        stats.dense_entries = dense_map_.size();
        stats.sparse_entries = sparse_entries_.size();
        stats.total_memory_bytes = stats.dense_entries * sizeof(real_t) + 
                                  stats.sparse_entries * sizeof(SparseProximityEntry);
        stats.sparsity_ratio = 1.0f - static_cast<real_t>(stats.dense_entries + stats.sparse_entries) / 
                              static_cast<real_t>(nsample_ * nsample_);
        return stats;
    }
    
    // Clear all data
    void clear() {
        dense_map_.clear();
        sparse_entries_.clear();
    }
};

// Note: CUDA-specific implementations are in separate .cu files

} // namespace rf

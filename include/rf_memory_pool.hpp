#ifndef RF_MEMORY_POOL_HPP
#define RF_MEMORY_POOL_HPP

#include "rf_types.hpp"
#include <memory>
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cstdlib>  // For std::free
#include <cstddef>  // For std::aligned_alloc

namespace rf {

// High-performance memory pool for Random Forest operations
// Reduces allocation overhead and improves cache locality
class MemoryPool {
public:
    explicit MemoryPool(size_t initial_size = 1024 * 1024); // 1MB initial
    ~MemoryPool();

    // Allocate aligned memory block
    void* allocate(size_t size, size_t alignment = 64);
    
    // Deallocate memory block (returns to pool)
    void deallocate(void* ptr, size_t size);
    
    // Reset pool (free all allocations)
    void reset();
    
    // Get memory usage statistics
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_current_usage() const { return current_usage_; }
    size_t get_peak_usage() const { return peak_usage_; }

private:
    struct Block {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;
    size_t total_allocated_;
    size_t current_usage_;
    size_t peak_usage_;
    mutable std::mutex mutex_;
    
    void expand_pool(size_t min_size);
    Block* find_free_block(size_t size);
};

// Thread-local memory pool for per-thread allocations
class ThreadLocalMemoryPool {
public:
    static MemoryPool& get_pool();
    static void cleanup_all_pools();
    
private:
    static thread_local std::unique_ptr<MemoryPool> pool_;
    static std::mutex cleanup_mutex_;
    static std::vector<MemoryPool*> all_pools_;
};

// RAII wrapper for memory pool allocations
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    PoolAllocator() = default;
    explicit PoolAllocator(MemoryPool& pool) : pool_(&pool) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}

    pointer allocate(size_type n) {
        if (pool_) {
            return static_cast<pointer>(pool_->allocate(n * sizeof(T), alignof(T)));
        } else {
            return static_cast<pointer>(std::aligned_alloc(alignof(T), n * sizeof(T)));
        }
    }

    void deallocate(pointer p, size_type n) {
        if (pool_) {
            pool_->deallocate(p, n * sizeof(T));
        } else {
            std::free(p);
        }
    }

    template<typename U>
    bool operator==(const PoolAllocator<U>& other) const {
        return pool_ == other.pool_;
    }

    template<typename U>
    bool operator!=(const PoolAllocator<U>& other) const {
        return !(*this == other);
    }

private:
    MemoryPool* pool_ = nullptr;
};

// High-performance vector using memory pool
template<typename T>
using PoolVector = std::vector<T, PoolAllocator<T>>;

// Memory-efficient array with custom allocator
template<typename T>
class FastArray {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;

    FastArray() = default;
    explicit FastArray(size_type size, MemoryPool* pool = nullptr);
    FastArray(size_type size, const T& value, MemoryPool* pool = nullptr);
    FastArray(const FastArray& other);
    FastArray(FastArray&& other) noexcept;
    ~FastArray();

    FastArray& operator=(const FastArray& other);
    FastArray& operator=(FastArray&& other) noexcept;

    // Element access
    reference operator[](size_type i) { return data_[i]; }
    const_reference operator[](size_type i) const { return data_[i]; }
    reference at(size_type i) { return data_[i]; }
    const_reference at(size_type i) const { return data_[i]; }
    reference front() { return data_[0]; }
    const_reference front() const { return data_[0]; }
    reference back() { return data_[size_ - 1]; }
    const_reference back() const { return data_[size_ - 1]; }

    // Iterators
    iterator begin() { return data_; }
    const_iterator begin() const { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator end() const { return data_ + size_; }

    // Capacity
    bool empty() const { return size_ == 0; }
    size_type size() const { return size_; }
    size_type capacity() const { return capacity_; }

    // Modifiers
    void resize(size_type new_size);
    void resize(size_type new_size, const T& value);
    void reserve(size_type new_capacity);
    void clear();
    void shrink_to_fit();

    // Data access
    pointer data() { return data_; }
    const_pointer data() const { return data_; }

private:
    T* data_ = nullptr;
    size_type size_ = 0;
    size_type capacity_ = 0;
    MemoryPool* pool_ = nullptr;
    
    void allocate(size_type size);
    void deallocate();
};

// Specialized fast arrays for common types
using FastIntArray = FastArray<integer_t>;
using FastRealArray = FastArray<real_t>;
using FastDoubleArray = FastArray<dp_t>;

// Memory-efficient matrix with column-major layout
template<typename T>
class FastMatrix {
public:
    using value_type = T;
    using size_type = size_t;

    FastMatrix() = default;
    FastMatrix(size_type rows, size_type cols, MemoryPool* pool = nullptr);
    FastMatrix(size_type rows, size_type cols, const T& value, MemoryPool* pool = nullptr);
    ~FastMatrix();

    // Element access (column-major indexing)
    T& operator()(size_type row, size_type col) { 
        return data_[row + col * rows_]; 
    }
    const T& operator()(size_type row, size_type col) const { 
        return data_[row + col * rows_]; 
    }

    // Row/column access
    T* row(size_type row) { return data_ + row; }
    const T* row(size_type row) const { return data_ + row; }
    T* column(size_type col) { return data_ + col * rows_; }
    const T* column(size_type col) const { return data_ + col * rows_; }

    // Dimensions
    size_type rows() const { return rows_; }
    size_type cols() const { return cols_; }
    size_type size() const { return rows_ * cols_; }

    // Data access
    T* data() { return data_; }
    const T* data() const { return data_; }

    // Resize
    void resize(size_type rows, size_type cols);

private:
    T* data_ = nullptr;
    size_type rows_ = 0;
    size_type cols_ = 0;
    MemoryPool* pool_ = nullptr;
    
    void allocate(size_type size);
    void deallocate();
};

// Specialized fast matrices
using FastIntMatrix = FastMatrix<integer_t>;
using FastRealMatrix = FastMatrix<real_t>;
using FastDoubleMatrix = FastMatrix<dp_t>;

// Global memory pool instance
extern MemoryPool g_memory_pool;

// Convenience functions for common allocations
template<typename T>
FastArray<T> make_fast_array(size_t size) {
    return FastArray<T>(size, &g_memory_pool);
}

template<typename T>
FastMatrix<T> make_fast_matrix(size_t rows, size_t cols) {
    return FastMatrix<T>(rows, cols, &g_memory_pool);
}

} // namespace rf

#endif // RF_MEMORY_POOL_HPP

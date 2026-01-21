#include "rf_memory_pool.hpp"
#include <cstdlib>
#include <algorithm>
#include <cstring>

namespace rf {

// Global memory pool instance
MemoryPool g_memory_pool;

// ============================================================================
// MemoryPool Implementation
// ============================================================================

MemoryPool::MemoryPool(size_t initial_size) 
    : total_allocated_(0), current_usage_(0), peak_usage_(0) {
    expand_pool(initial_size);
}

MemoryPool::~MemoryPool() {
    for (auto& block : blocks_) {
        if (block.ptr) {
            std::free(block.ptr);
        }
    }
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find existing free block
    Block* block = find_free_block(size);
    if (block) {
        block->in_use = true;
        current_usage_ += size;
        peak_usage_ = std::max(peak_usage_, current_usage_);
        return block->ptr;
    }
    
    // Need to expand pool
    expand_pool(size);
    block = find_free_block(size);
    if (block) {
        block->in_use = true;
        current_usage_ += size;
        peak_usage_ = std::max(peak_usage_, current_usage_);
        return block->ptr;
    }
    
    // Fallback to system allocation
    void* ptr = std::aligned_alloc(alignment, size);
    if (ptr) {
        total_allocated_ += size;
        current_usage_ += size;
        peak_usage_ = std::max(peak_usage_, current_usage_);
    }
    return ptr;
}

void MemoryPool::deallocate(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Find block and mark as free
    for (auto& block : blocks_) {
        if (block.ptr == ptr) {
            block.in_use = false;
            current_usage_ -= size;
            return;
        }
    }
    
    // Not found in pool, free with system
    std::free(ptr);
    current_usage_ -= size;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& block : blocks_) {
        block.in_use = false;
    }
    current_usage_ = 0;
}

void MemoryPool::expand_pool(size_t min_size) {
    size_t expand_size = std::max(min_size, total_allocated_ / 2);
    expand_size = std::max(expand_size, size_t(1024 * 1024)); // At least 1MB
    
    void* ptr = std::aligned_alloc(64, expand_size);
    if (ptr) {
        blocks_.push_back({ptr, expand_size, false});
        total_allocated_ += expand_size;
    }
}

MemoryPool::Block* MemoryPool::find_free_block(size_t size) {
    for (auto& block : blocks_) {
        if (!block.in_use && block.size >= size) {
            return &block;
        }
    }
    return nullptr;
}

// ============================================================================
// ThreadLocalMemoryPool Implementation
// ============================================================================

thread_local std::unique_ptr<MemoryPool> ThreadLocalMemoryPool::pool_;
std::mutex ThreadLocalMemoryPool::cleanup_mutex_;
std::vector<MemoryPool*> ThreadLocalMemoryPool::all_pools_;

MemoryPool& ThreadLocalMemoryPool::get_pool() {
    if (!pool_) {
        pool_ = std::make_unique<MemoryPool>();
        std::lock_guard<std::mutex> lock(cleanup_mutex_);
        all_pools_.push_back(pool_.get());
    }
    return *pool_;
}

void ThreadLocalMemoryPool::cleanup_all_pools() {
    std::lock_guard<std::mutex> lock(cleanup_mutex_);
    for (auto* pool : all_pools_) {
        pool->reset();
    }
}

// ============================================================================
// FastArray Implementation
// ============================================================================

template<typename T>
FastArray<T>::FastArray(size_type size, MemoryPool* pool) 
    : pool_(pool) {
    if (size > 0) {
        allocate(size);
        size_ = size;
    }
}

template<typename T>
FastArray<T>::FastArray(size_type size, const T& value, MemoryPool* pool) 
    : pool_(pool) {
    if (size > 0) {
        allocate(size);
        size_ = size;
        std::fill(data_, data_ + size_, value);
    }
}

template<typename T>
FastArray<T>::FastArray(const FastArray& other) 
    : pool_(other.pool_) {
    if (other.size_ > 0) {
        allocate(other.size_);
        size_ = other.size_;
        std::copy(other.data_, other.data_ + other.size_, data_);
    }
}

template<typename T>
FastArray<T>::FastArray(FastArray&& other) noexcept 
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_), pool_(other.pool_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

template<typename T>
FastArray<T>::~FastArray() {
    deallocate();
}

template<typename T>
FastArray<T>& FastArray<T>::operator=(const FastArray& other) {
    if (this != &other) {
        deallocate();
        pool_ = other.pool_;
        if (other.size_ > 0) {
            allocate(other.size_);
            size_ = other.size_;
            std::copy(other.data_, other.data_ + other.size_, data_);
        }
    }
    return *this;
}

template<typename T>
FastArray<T>& FastArray<T>::operator=(FastArray&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        pool_ = other.pool_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    return *this;
}

template<typename T>
void FastArray<T>::resize(size_type new_size) {
    if (new_size <= capacity_) {
        size_ = new_size;
    } else {
        T* old_data = data_;
        size_type old_size = size_;
        allocate(new_size);
        if (old_data) {
            std::copy(old_data, old_data + std::min(old_size, new_size), data_);
            if (pool_) {
                pool_->deallocate(old_data, old_size * sizeof(T));
            } else {
                std::free(old_data);
            }
        }
        size_ = new_size;
    }
}

template<typename T>
void FastArray<T>::resize(size_type new_size, const T& value) {
    size_type old_size = size_;
    resize(new_size);
    if (new_size > old_size) {
        std::fill(data_ + old_size, data_ + new_size, value);
    }
}

template<typename T>
void FastArray<T>::reserve(size_type new_capacity) {
    if (new_capacity > capacity_) {
        T* old_data = data_;
        size_type old_size = size_;
        allocate(new_capacity);
        if (old_data) {
            std::copy(old_data, old_data + old_size, data_);
            if (pool_) {
                pool_->deallocate(old_data, old_size * sizeof(T));
            } else {
                std::free(old_data);
            }
        }
    }
}

template<typename T>
void FastArray<T>::clear() {
    size_ = 0;
}

template<typename T>
void FastArray<T>::shrink_to_fit() {
    if (size_ < capacity_) {
        T* old_data = data_;
        size_type old_size = size_;
        allocate(size_);
        if (old_data) {
            std::copy(old_data, old_data + old_size, data_);
            if (pool_) {
                pool_->deallocate(old_data, old_size * sizeof(T));
            } else {
                std::free(old_data);
            }
        }
    }
}

template<typename T>
void FastArray<T>::allocate(size_type size) {
    if (pool_) {
        data_ = static_cast<T*>(pool_->allocate(size * sizeof(T), alignof(T)));
    } else {
        data_ = static_cast<T*>(std::aligned_alloc(alignof(T), size * sizeof(T)));
    }
    capacity_ = size;
}

template<typename T>
void FastArray<T>::deallocate() {
    if (data_) {
        if (pool_) {
            pool_->deallocate(data_, capacity_ * sizeof(T));
        } else {
            std::free(data_);
        }
        data_ = nullptr;
        capacity_ = 0;
    }
}

// ============================================================================
// FastMatrix Implementation
// ============================================================================

template<typename T>
FastMatrix<T>::FastMatrix(size_type rows, size_type cols, MemoryPool* pool) 
    : rows_(rows), cols_(cols), pool_(pool) {
    if (rows > 0 && cols > 0) {
        allocate(rows * cols);
    }
}

template<typename T>
FastMatrix<T>::FastMatrix(size_type rows, size_type cols, const T& value, MemoryPool* pool) 
    : rows_(rows), cols_(cols), pool_(pool) {
    if (rows > 0 && cols > 0) {
        allocate(rows * cols);
        std::fill(data_, data_ + rows * cols, value);
    }
}

template<typename T>
FastMatrix<T>::~FastMatrix() {
    if (data_) {
        if (pool_) {
            pool_->deallocate(data_, rows_ * cols_ * sizeof(T));
        } else {
            std::free(data_);
        }
    }
}

template<typename T>
void FastMatrix<T>::resize(size_type rows, size_type cols) {
    if (rows != rows_ || cols != cols_) {
        if (data_) {
            if (pool_) {
                pool_->deallocate(data_, rows_ * cols_ * sizeof(T));
            } else {
                std::free(data_);
            }
        }
        rows_ = rows;
        cols_ = cols;
        if (rows > 0 && cols > 0) {
            allocate(rows * cols);
        } else {
            data_ = nullptr;
        }
    }
}

template<typename T>
void FastMatrix<T>::allocate(size_type size) {
    if (pool_) {
        data_ = static_cast<T*>(pool_->allocate(size * sizeof(T), alignof(T)));
    } else {
        data_ = static_cast<T*>(std::aligned_alloc(alignof(T), size * sizeof(T)));
    }
}

template<typename T>
void FastMatrix<T>::deallocate() {
    if (data_) {
        if (pool_) {
            pool_->deallocate(data_, rows_ * cols_ * sizeof(T));
        } else {
            std::free(data_);
        }
        data_ = nullptr;
    }
}

// Explicit template instantiations for common types
template class FastArray<integer_t>;
template class FastArray<real_t>;
template class FastArray<dp_t>;

template class FastMatrix<integer_t>;
template class FastMatrix<real_t>;
template class FastMatrix<dp_t>;

} // namespace rf

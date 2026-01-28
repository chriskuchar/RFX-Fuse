#ifndef RF_PARALLEL_TREE_GROWING_HPP
#define RF_PARALLEL_TREE_GROWING_HPP

#include "rf_types.hpp"
#include "rf_memory_pool.hpp"
#include "rf_vectorized_ops.hpp"
#include "rf_config.hpp"
#include "rf_random_forest.hpp"
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include <condition_variable>
#include <chrono>
#include <unordered_map>
#include <string>
#include <memory>

namespace rf {

// High-performance parallel tree growing implementation
// Uses work-stealing thread pool for optimal load balancing

// ============================================================================
// Work-Stealing Thread Pool
// ============================================================================

class WorkStealingThreadPool {
public:
    explicit WorkStealingThreadPool(size_t num_threads = 0);
    ~WorkStealingThreadPool();

    // Submit work to the thread pool
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    // Wait for all tasks to complete
    void wait_for_all();

    // Get number of threads
    size_t get_num_threads() const { return threads_.size(); }

    // Shutdown the thread pool
    void shutdown();

private:
    std::vector<std::thread> threads_;
    std::vector<std::queue<std::function<void()>>> work_queues_;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes_;
    std::atomic<bool> shutdown_;
    std::atomic<size_t> next_thread_;
    
    void worker_thread(size_t thread_id);
    bool try_steal_work(size_t thread_id, std::function<void()>& task);
};

// ============================================================================
// Parallel Tree Growing Manager
// ============================================================================

class ParallelTreeGrowingManager {
public:
    explicit ParallelTreeGrowingManager(size_t num_threads = 0);
    ~ParallelTreeGrowingManager();

    // Grow multiple trees in parallel
    void grow_trees_parallel(
        const real_t* X,                    // Training data
        const integer_t* y,                 // Labels (classification)
        const real_t* y_reg,                // Targets (regression)
        const real_t* sample_weight,        // Sample weights
        integer_t n_samples,                // Number of samples
        integer_t n_features,               // Number of features
        integer_t n_classes,                // Number of classes
        integer_t n_trees,                  // Number of trees to grow
        integer_t mtry,                     // Features per split
        integer_t min_samples_split,        // Min samples for split
        integer_t max_depth,                // Maximum tree depth
        integer_t seed,                     // Random seed
        rf::TaskType task_type,                 // Classification/Regression/Unsupervised
        // Tree storage arrays
        real_t* tnodewt_all,               // Node weights
        real_t* xbestsplit_all,             // Split values
        integer_t* nodestatus_all,          // Node status
        integer_t* bestvar_all,             // Best variables
        integer_t* treemap_all,             // Tree structure
        integer_t* catgoleft_all,           // Categorical splits
        integer_t* nodeclass_all,           // Node classes
        integer_t* nnode_all,               // Number of nodes per tree
        // OOB tracking
        real_t* q_all,                      // OOB votes
        integer_t* nout_all,                // OOB counts
        // Proximity (if requested)
        dp_t* proximity_matrix,             // Proximity matrix
        bool compute_proximity,             // Compute proximity?
        // Importance (if requested)
        real_t* feature_importances,        // Feature importances
        bool compute_importance             // Compute importance?
    );

    // Get performance statistics
    struct PerformanceStats {
        double total_time_ms;
        double tree_growing_time_ms;
        double oob_time_ms;
        double proximity_time_ms;
        double importance_time_ms;
        size_t trees_per_second;
        double parallel_efficiency;
    };
    
    PerformanceStats get_performance_stats() const { return stats_; }

private:
    WorkStealingThreadPool thread_pool_;
    mutable std::mutex stats_mutex_;
    PerformanceStats stats_;
    
    // Thread-local storage for each worker
    struct ThreadLocalData {
        FastIntArray nin;                   // In-bag counts
        FastIntArray nout;                  // Out-of-bag counts
        FastIntArray jinbag;                // In-bag indices
        FastIntArray joobag;                // Out-of-bag indices
        FastIntArray amat;                  // Sorted indices
        FastIntArray ties;                  // Ties array
        FastIntArray isort;                 // Sort indices
        FastRealArray v;                    // Sort values
        FastRealArray win;                  // Sample weights
        FastIntArray cat;                   // Variable types
        FastIntArray asave;                 // Sorted data
        
        // Tree building workspace
        FastIntArray idmove;                 // Data movement
        FastIntArray igoleft;               // Left groups
        FastIntArray igl;                   // Group labels
        FastIntArray nodestart;             // Node starts
        FastIntArray nodestop;              // Node stops
        FastIntArray itemp;                 // Temp indices
        FastIntArray itmp;                  // Temp indices
        FastIntArray icat;                  // Categorical temp
        FastRealArray tcat;                 // Categorical values
        FastRealArray classpop;             // Class populations
        FastRealArray tclasspop;            // Temp class pop
        FastRealArray wr;                   // Right weights
        FastRealArray wl;                   // Left weights
        FastRealArray tclasscat;            // Class categories
        FastRealArray tmpclass;             // Temp classes
        
        // Tree output arrays
        FastRealArray tnodewt;              // Node weights
        FastRealArray xbestsplit;           // Split values
        FastIntArray nodestatus;            // Node status
        FastIntArray bestvar;               // Best variables
        FastIntArray treemap;               // Tree structure
        FastIntArray catgoleft;             // Categorical splits
        FastIntArray nodeclass;             // Node classes
        FastIntArray jtr;                   // Tree predictions
        FastIntArray nodextr;               // Node extra
        
        ThreadLocalData(integer_t nsample, integer_t mdim, integer_t nclass, 
                       integer_t maxnode, integer_t maxcat, integer_t ninbag);
    };
    
    // Thread-local storage management
    thread_local static std::unique_ptr<ThreadLocalData> thread_local_data_;
    
    // Tree growing task
    struct TreeGrowingTask {
        integer_t tree_id;
        integer_t seed;
        integer_t n_samples;
        integer_t n_features;
        integer_t n_classes;
        integer_t mtry;
        integer_t min_samples_split;
        integer_t max_depth;
        rf::TaskType task_type;
        
        // Data pointers
        const real_t* X;
        const integer_t* y;
        const real_t* y_reg;
        const real_t* sample_weight;
        
        // Output pointers
        real_t* tnodewt;
        real_t* xbestsplit;
        integer_t* nodestatus;
        integer_t* bestvar;
        integer_t* treemap;
        integer_t* catgoleft;
        integer_t* nodeclass;
        integer_t* nnode;
        real_t* q;
        integer_t* nout;
        dp_t* proximity_matrix;
        real_t* feature_importances;
        
        bool compute_proximity;
        bool compute_importance;
    };
    
    // Grow a single tree (thread-safe)
    void grow_single_tree(const TreeGrowingTask& task, ThreadLocalData& local_data);
    
    // Bootstrap sampling (thread-safe)
    void bootstrap_sample(ThreadLocalData& local_data, integer_t nsample, 
                         integer_t ninbag, integer_t seed);
    
    // Prepare data for tree growing
    void prepare_tree_data(ThreadLocalData& local_data, const real_t* X,
                          integer_t n_samples, integer_t n_features);
    
    // Test tree on OOB samples
    void test_tree_oob(ThreadLocalData& local_data, const TreeGrowingTask& task);
    
    // Update proximity matrix
    void update_proximity_matrix(ThreadLocalData& local_data, const TreeGrowingTask& task);
    
    // Update feature importance
    void update_feature_importance(ThreadLocalData& local_data, const TreeGrowingTask& task);
};

// ============================================================================
// NUMA-Aware Processing
// ============================================================================

class NUMAProcessor {
public:
    static void bind_thread_to_cpu(std::thread::id thread_id, int cpu_id);
    static void bind_thread_to_numa_node(std::thread::id thread_id, int numa_node);
    static int get_num_numa_nodes();
    static int get_num_cpus_per_numa_node();
    static void optimize_for_numa();
    
private:
    static std::mutex numa_mutex_;
    static std::unordered_map<std::thread::id, int> thread_cpu_bindings_;
};

// ============================================================================
// Performance Monitoring
// ============================================================================

class PerformanceMonitor {
public:
    struct TimingData {
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        double duration_ms;
    };
    
    void start_timer(const std::string& name);
    void end_timer(const std::string& name);
    double get_duration_ms(const std::string& name) const;
    
    void print_statistics() const;
    void reset();
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, TimingData> timings_;
};

// ============================================================================
// Convenience Functions
// ============================================================================

// Create optimized thread pool based on system capabilities
std::unique_ptr<WorkStealingThreadPool> create_optimized_thread_pool();

// Get optimal number of threads for current system
size_t get_optimal_thread_count();

// Benchmark parallel vs sequential performance
struct BenchmarkResults {
    double sequential_time_ms;
    double parallel_time_ms;
    double speedup;
    double parallel_efficiency;
};

BenchmarkResults benchmark_tree_growing(
    const real_t* X, const integer_t* y, integer_t n_samples, 
    integer_t n_features, integer_t n_trees);

} // namespace rf

#endif // RF_PARALLEL_TREE_GROWING_HPP

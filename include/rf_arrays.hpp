#ifndef RF_ARRAYS_HPP
#define RF_ARRAYS_HPP

#include "rf_types.hpp"
#include <vector>
#include <memory>

namespace rf {

// Global RF arrays container
// Matches prog_cuda_arrays.f90 structure
struct RFArrays {
    // Global parameters (accessible everywhere)
    integer_t mdim = 0;
    integer_t nsample = 0;
    integer_t nclass = 0;
    integer_t ntree = 0;
    integer_t mtry = 0;
    integer_t maxcat = 0;
    integer_t maxnode = 0;

    static constexpr integer_t iprox = 1;
    static constexpr integer_t minndsize = 1;
    static constexpr integer_t imp = 1;
    static constexpr integer_t impn = 1;
    integer_t mimp = 0;  // = imp*(mdim-1) + 1

    // Main data arrays
    std::vector<real_t> x;              // (mdim, nsample) - column-major
    std::vector<integer_t> cl;          // (nsample)
    std::vector<real_t> weight;         // (nsample)
    std::vector<real_t> win;            // (nsample)
    std::vector<real_t> v;              // (nsample)
    std::vector<real_t> q;              // (nclass, nsample)
    std::vector<real_t> qimp;           // (nsample)
    std::vector<real_t> qimpm;          // (nsample, mdim)
    std::vector<real_t> wr;             // (nclass)
    std::vector<real_t> wl;             // (nclass)
    std::vector<real_t> tmpclass;       // (nclass)
    std::vector<real_t> tclasscat;      // (nclass, maxcat)
    std::vector<real_t> avimp;          // (mdim)
    std::vector<real_t> sqsd;           // (mdim)
    std::vector<real_t> xbestsplit;     // (maxnode)
    std::vector<real_t> tnodewt;        // (maxnode)
    std::vector<real_t> classpop;       // (nclass, maxnode)
    std::vector<real_t> tclasspop;      // (nclass)

    // Index and working arrays
    std::vector<integer_t> asave;       // (mdim, nsample)
    std::vector<integer_t> amat;        // (mdim, nsample)
    std::vector<integer_t> ties;        // (mdim, nsample)
    std::vector<integer_t> nout;        // (nsample)
    std::vector<integer_t> nin;         // (nsample)
    std::vector<integer_t> joob;        // (nsample)
    std::vector<integer_t> pjoob;       // (nsample)
    std::vector<integer_t> nodextr;     // (nsample)
    std::vector<integer_t> nodexvr;     // (nsample)
    std::vector<integer_t> ndbegin;     // (nsample) - proximity workspace
    std::vector<integer_t> jvr;         // (nsample)
    std::vector<integer_t> jtr;         // (nsample)
    std::vector<integer_t> jest;        // (nsample)
    std::vector<integer_t> isort;       // (nsample)
    std::vector<integer_t> itemp;       // (nsample)
    std::vector<integer_t> jinbag;      // (nsample)
    std::vector<integer_t> joobag;      // (nsample)
    std::vector<integer_t> idmove;      // (nsample)
    std::vector<integer_t> bestvar;     // (maxnode)
    std::vector<integer_t> nod;         // (maxnode)
    std::vector<integer_t> nodestatus;  // (maxnode)
    std::vector<integer_t> nodestart;   // (maxnode)
    std::vector<integer_t> nodestop;    // (maxnode)
    std::vector<integer_t> nodeclass;   // (maxnode)
    std::vector<integer_t> treemap;     // (2, maxnode)
    std::vector<integer_t> catgoleft;   // (maxcat, maxnode)

    std::vector<integer_t> nc;          // (nclass)
    std::vector<integer_t> ncn;         // (nsample)
    std::vector<integer_t> igoleft;     // (nsample)
    std::vector<integer_t> igl;         // (nsample)
    std::vector<integer_t> itmp;        // (nsample)
    std::vector<integer_t> nodexb;      // (nsample) - proximity workspace
    std::vector<integer_t> npcase;      // (nsample) - proximity workspace
    std::vector<integer_t> ncount;      // (nsample)
    std::vector<real_t> tcat;           // (nsample)
    std::vector<integer_t> icat;        // (nsample)

    // Proximity arrays (double precision)
    std::vector<dp_t> prox;             // (nsample, nsample)
    std::vector<dp_t> proxsym;          // (nsample, nsample)
    std::vector<dp_t> dwork;            // (nsample)

    // QLORA-Proximity configuration
    integer_t quant_mode = 1;           // 0=FP32, 1=FP16, 2=INT8, 3=NF4
    integer_t lora_rank = 0;            // LoRA rank (0=disabled)
    integer_t fusion_interval = 1;      // Trees between fusion (always 1 in ultra mode)
    bool use_qlora = true;              // Enable QLORA-Proximity
    bool use_sparse = false;            // Enable block-sparse storage
    bool upper_triangle = true;         // Store only upper triangle
    real_t sparsity_threshold = 1e-3f;  // Block sparsity threshold

    // Allocate all arrays based on RF parameters
    void allocate(integer_t ns, integer_t nt, integer_t md, integer_t n_classes, integer_t mc, integer_t mn);

    // Helper to access column-major 2D array: arr(i, j) in Fortran = arr[i + j*rows] in C
    template<typename T>
    static inline T& at_colmajor(std::vector<T>& vec, integer_t i, integer_t j, integer_t rows) {
        return vec[i + j * rows];
    }

    template<typename T>
    static inline const T& at_colmajor(const std::vector<T>& vec, integer_t i, integer_t j, integer_t rows) {
        return vec[i + j * rows];
    }

    // Zero all variable importance arrays (used at start of each tree)
    inline void zero_varimp() {
        std::fill(qimp.begin(), qimp.end(), 0.0f);
        std::fill(qimpm.begin(), qimpm.end(), 0.0f);
        std::fill(avimp.begin(), avimp.end(), 0.0f);
        std::fill(sqsd.begin(), sqsd.end(), 0.0f);
    }

    // Zero proximity arrays
    inline void zero_prox() {
        std::fill(prox.begin(), prox.end(), 0.0);
        std::fill(proxsym.begin(), proxsym.end(), 0.0);
    }
};

// Global arrays instance
extern RFArrays g_arrays;

} // namespace rf

#endif // RF_ARRAYS_HPP

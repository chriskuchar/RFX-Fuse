#include "rf_utils.hpp"

namespace rf {

// Global RNG instance
MT19937 g_rng;

// Cleanup function for global RNG state
void cleanup_global_rng() {
    // Reset the global RNG to a known state
    g_rng.sgrnd(0);
}

// Quicksort algorithm - exact port from util.f90 preserving original structure
void quicksort(real_t* v, integer_t* iperm, integer_t ii, integer_t jj, integer_t kk) {
    real_t vt, vtt;
    integer_t t, tt;
    integer_t iu[32], il[32];
    integer_t m, i, j, k, ij, l;

    // Convert to 0-based indexing (Fortran is 1-based)
    i = ii - 1;
    j = jj - 1;
    m = 0;

    while (true) {
        // Label 10
        if (i >= j) goto label_80;

        // Label 20
        k = i;
        ij = (j + i) / 2;
        t = iperm[ij];
        vt = v[ij];

        if (v[i] <= vt) goto label_30;
        iperm[ij] = iperm[i];
        iperm[i] = t;
        t = iperm[ij];
        v[ij] = v[i];
        v[i] = vt;
        vt = v[ij];

        // Label 30
        label_30:
        l = j;
        if (v[j] >= vt) goto label_50;
        iperm[ij] = iperm[j];
        iperm[j] = t;
        t = iperm[ij];
        v[ij] = v[j];
        v[j] = vt;
        vt = v[ij];

        if (v[i] <= vt) goto label_50;
        iperm[ij] = iperm[i];
        iperm[i] = t;
        t = iperm[ij];
        v[ij] = v[i];
        v[i] = vt;
        vt = v[ij];
        goto label_50;

        // Label 40
        label_40:
        iperm[l] = iperm[k];
        iperm[k] = tt;
        v[l] = v[k];
        v[k] = vtt;

        // Label 50
        label_50:
        l = l - 1;
        if (v[l] > vt) goto label_50;
        tt = iperm[l];
        vtt = v[l];

        // Label 60
        label_60:
        k = k + 1;
        if (v[k] < vt) goto label_60;
        if (k <= l) goto label_40;

        if (l - i <= j - k) goto label_70;
        il[m] = i;
        iu[m] = l;
        i = k;
        m = m + 1;
        goto label_90;

        // Label 70
        label_70:
        il[m] = k;
        iu[m] = j;
        j = l;
        m = m + 1;
        goto label_90;

        // Label 80
        label_80:
        m = m - 1;
        if (m < 0) return;
        i = il[m];
        j = iu[m];

        // Label 90
        label_90:
        if (j - i > 10) goto label_20;
        
        // Label 20
        label_20:
        if (i == ii - 1) continue;  // goto 10
        i = i - 1;

        // Label 100
        label_100:
        i = i + 1;
        if (i == j) goto label_80;
        t = iperm[i + 1];
        vt = v[i + 1];
        if (v[i] <= vt) goto label_100;
        k = i;

        // Label 110
        label_110:
        iperm[k + 1] = iperm[k];
        v[k + 1] = v[k];
        k = k - 1;
        if (vt < v[k]) goto label_110;
        iperm[k + 1] = t;
        v[k + 1] = vt;
        goto label_100;
    }
}

} // namespace rf

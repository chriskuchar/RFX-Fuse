#ifndef RF_UTILS_HPP
#define RF_UTILS_HPP

#include "rf_types.hpp"
#include <vector>

namespace rf {

// Zero integer vector (matches zerv from util.f90)
inline void zerv(integer_t* ix, integer_t m1) {
    for (integer_t i = 0; i < m1; i++) {
        ix[i] = 0;
    }
}

// Zero real vector (matches zervr from util.f90)
inline void zervr(real_t* rx, integer_t m1) {
    for (integer_t i = 0; i < m1; i++) {
        rx[i] = 0.0f;
    }
}

// Zero double precision vector (matches zervd from util.f90)
inline void zervd(dp_t* rx, integer_t m1) {
    for (integer_t i = 0; i < m1; i++) {
        rx[i] = 0.0;
    }
}

// Zero integer matrix (matches zerm from util.f90)
// Fortran column-major order: mx(m1, m2)
inline void zerm(integer_t* mx, integer_t m1, integer_t m2) {
    for (integer_t i = 0; i < m1 * m2; i++) {
        mx[i] = 0;
    }
}

// Zero real matrix (matches zermr from util.f90)
// Fortran column-major order: rx(m1, m2)
inline void zermr(real_t* rx, integer_t m1, integer_t m2) {
    for (integer_t i = 0; i < m1 * m2; i++) {
        rx[i] = 0.0f;
    }
}

// Zero double precision matrix (matches zermd from util.f90)
// Fortran column-major order: rx(m1, m2)
inline void zermd(dp_t* rx, integer_t m1, integer_t m2) {
    for (integer_t i = 0; i < m1 * m2; i++) {
        rx[i] = 0.0;
    }
}

// Random bit function (matches irbit from util.f90)
// Exact algorithm from original
inline integer_t irbit(integer_t& iseed) {
    constexpr integer_t IB1 = 1;
    constexpr integer_t IB2 = 2;
    constexpr integer_t IB5 = 16;
    constexpr integer_t IB18 = 131072;
    constexpr integer_t MASK = IB1 + IB2 + IB5;

    if ((iseed & IB18) != 0) {
        iseed = ((iseed ^ MASK) << 1) | IB1;
        return 1;
    } else {
        iseed = (iseed << 1) & (~IB1);
        return 0;
    }
}

// MT19937 Random Number Generator
// Fixed implementation based on working Fortran version
class MT19937 {
private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATA = 0x9908b0dfUL;
    static constexpr uint32_t UMASK = 0x80000000UL;  // Most significant w-r bits
    static constexpr uint32_t LMASK = 0x7fffffffUL;  // Least significant r bits
    static constexpr uint32_t TMASKB = 0x9d2c5680UL;
    static constexpr uint32_t TMASKC = 0xefc60000UL;

    int mti;
    uint32_t mt[N];
    uint32_t mag01[2];

public:
    MT19937() : mti(N + 1) {
        mag01[0] = 0x0UL;
        mag01[1] = MATA;
    }

    // Initialize generator
    void sgrnd(uint32_t seed) {
        mt[0] = seed & 0xffffffffUL;
        for (mti = 1; mti < N; mti++) {
            mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
            mt[mti] &= 0xffffffffUL;
        }
    }

    // Generate random double in [0,1]
    double grnd() {
        uint32_t y;
        int kk;

        if (mti >= N) {
            if (mti == N + 1) {
                sgrnd(5489UL);
            }

            for (kk = 0; kk < N - M; kk++) {
                y = (mt[kk] & UMASK) | (mt[kk+1] & LMASK);
                mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
            }
            for (kk = N - M; kk < N - 1; kk++) {
                y = (mt[kk] & UMASK) | (mt[kk+1] & LMASK);
                mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
            }
            y = (mt[N-1] & UMASK) | (mt[0] & LMASK);
            mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

            mti = 0;
        }

        y = mt[mti++];

        y ^= (y >> 11);
        y ^= (y << 7) & TMASKB;
        y ^= (y << 15) & TMASKC;
        y ^= (y >> 18);

        return ((double)y / (double)0xffffffffUL);
    }

    // Random uniform in [0,1] as float
    real_t randomu() {
        return static_cast<real_t>(grnd());
    }
};

// Global MT19937 instance
extern MT19937 g_rng;

// Cleanup function for global RNG state
void cleanup_global_rng();

// Quicksort algorithm (exact port from util.f90)
// Preserves original algorithm structure with goto statements converted to loops
void quicksort(real_t* v, integer_t* iperm, integer_t ii, integer_t jj, integer_t kk);

} // namespace rf

#endif // RF_UTILS_HPP

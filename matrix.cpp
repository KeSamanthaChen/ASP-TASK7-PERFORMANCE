#include "matrix.h"

#include <immintrin.h> // AVX, but include the SSE & SSE2

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <omp.h>

namespace chrono = std::chrono;

// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
inline void transpose(float *src, float *dst, const int N, const int M) {
    #pragma omp parallel for num_threads(4)
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}


void naive_matrix_multiply(float *a, float *b, float *c, int n) { //A*B
    std::vector<float> d(n * n, 0.0f);
    // std::vector<float> r(4, 0.0f);
    transpose(b, d.data(), n, n);

    #pragma omp parallel for num_threads(4)  
    for (int i = 0; i < n; i++) { // row
        for (int j = 0; j < n; j++) { // column
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < n; k+=4) { 
                __m128 fa = _mm_load_ps(&a[i*n+k]);
                __m128 fd = _mm_load_ps(&d[j*n+k]);
                sum = _mm_add_ps(sum, _mm_mul_ps(fa, fd));
            }
            // https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
            __m128 shuf   = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
            __m128 sums   = _mm_add_ps(sum, shuf);      // sums = [ D+C C+D | B+A A+B ]
            shuf          = _mm_movehl_ps(shuf, sums);      //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
            sums          = _mm_add_ss(sums, shuf);
            c[i*n+j]      = _mm_cvtss_f32(sums);
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void matrix_multiply(float *a, float *b, float *c, int n) {
    naive_matrix_multiply(a, b, c, n);
}

#ifdef __cplusplus
}
#endif
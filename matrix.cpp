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
inline void transpose4x4_SSE(float *A, float *B, const int n) {
    __m128 row1 = _mm_load_ps(&A[0*n]);
    __m128 row2 = _mm_load_ps(&A[1*n]);
    __m128 row3 = _mm_load_ps(&A[2*n]);
    __m128 row4 = _mm_load_ps(&A[3*n]);
     _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
     _mm_store_ps(&B[0*n], row1);
     _mm_store_ps(&B[1*n], row2);
     _mm_store_ps(&B[2*n], row3);
     _mm_store_ps(&B[3*n], row4);
}

inline void transpose_block_SSE4x4(float *A, float *B, const int n) {
    #pragma omp parallel for num_threads(4)
    for(int i=0; i<n; i+=4) {
        for(int j=0; j<n; j+=4) {
            transpose4x4_SSE(&A[i*n +j], &B[j*n + i], n);
        }   
    }
    // for(int i=0; i<n; i+=16) {
    //     for(int j=0; j<n; j+=16) {
    //         int max_i2 = i+16 < n ? i + 16 : n;
    //         int max_j2 = j+16 < n ? j + 16 : n;
    //         for(int i2=i; i2<max_i2; i2+=4) {
    //             for(int j2=j; j2<max_j2; j2+=4) { // calculate four by four
    //                 transpose4x4_SSE(&A[i2*n +j2], &B[j2*n + i2], n);
    //             }
    //         }
    //     }
    // }
}


void naive_matrix_multiply(float *a, float *b, float *c, int n) { //A*B
    std::vector<float> d(n * n, 0.0f);
    transpose_block_SSE4x4(b, d.data(), n);

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) { // row
        for (int j = 0; j < n; j++) { // column
            c[i * n + j] = 0.0f;
            for (int k = 0; k < n; k++) { // matrix multiple, store in row, cache locality
                c[i * n + j] += a[i * n + k] * d[j * n + k];
            }
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
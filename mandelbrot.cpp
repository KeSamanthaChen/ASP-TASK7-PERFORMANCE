#include "mandelbrot.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <omp.h>

__m128 constCompare = _mm_set_ps1(4.0f);
__m128 constMulti = _mm_set_ps1(2);

/**
 * @brief Calculates mandelbrot and stores the output in plot
 *
 * @param width  number of pixels available horizontally
 * @param height number of pixels available vertically
 * @param plot   output array
 */
void naive_mandelbrot(int width, int height, int* plot) {
    float dx = (X_END - X_START) / (width - 1);
    float dy = (Y_END - Y_START) / (height - 1);

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < height; i++) { // the same i, j in mandelbort_dirver compare, for i=0
        for (int j = 0; j < width; j+=4) {
            float x = X_START + j * dx;  // real value
            float x1 =  X_START + (j+1) * dx;// + dx;
            float x2 =  X_START + (j+2) * dx;// + dx + dx;
            float x3 =  X_START + (j+3) * dx;// + dx + dx + dx;
            float y = Y_END - i * dy;    // imaginary value

            __m128 x_simd = _mm_set_ps(x, x1, x2, x3);
            __m128 y_simd = _mm_set_ps1(y);

            // put the function here
            auto re = x_simd;
            auto im = y_simd;
            // auto re = x;
            // auto im = y;
            int current_cmp_val_int = 0;
            int flags[4] = {0};

            for (auto k = 0; k < LOOP; k++) {
                __m128 re2 = _mm_mul_ps(re, re);
                __m128 im2 = _mm_mul_ps(im, im);
                // float re2 = re * re;
                // float im2 = im * im;

                // verify if f(z) diverges to infinity
                __m128 cmp_val = _mm_cmpgt_ps(_mm_add_ps(re2, im2), constCompare); // if true -> -nan, if false -> 0
                // if (re2 + im2 > 4.0f) return i;
                // Set each bit of mask dst based on the most significant bit of the corresponding packed single-precision (32-bit) floating-point element in a.
                int cmp_val_int = _mm_movemask_ps(cmp_val); 
                
                // 1    1    0    0
                // 8188 8189 8190 8191
                if (cmp_val_int != current_cmp_val_int) {
                    if (cmp_val_int & 1 && !flags[0]) {
                        flags[0] = 1;
                        plot[i*width + j + 3] = k;
                        current_cmp_val_int += 1;
                    }
                    if (cmp_val_int & 2 && !flags[1]) {
                        flags[1] = 1;
                        plot[i*width + j + 2] = k;
                        current_cmp_val_int += 2;
                    }
                    if (cmp_val_int & 4 && !flags[2]) {
                        flags[2] = 1;
                        plot[i*width + j + 1] = k;
                        current_cmp_val_int += 4;
                    }
                    if (cmp_val_int & 8 && !flags[3]) {
                        flags[3] = 1;
                        plot[i*width + j] = k;
                        current_cmp_val_int += 8;
                    }
                    if (current_cmp_val_int == 15) break; // all result set
                }

                im = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(constMulti, re),im), y_simd);
                re = _mm_add_ps(_mm_sub_ps(re2, im2), x_simd);
                // im = 2 * re * im + y;
                // re = re2 - im2 + x;
            }
            if (current_cmp_val_int != 15) {
                int new_int = 15 - current_cmp_val_int;
                if (new_int & 1) plot[i*width + j + 3] = LOOP;
                if (new_int & 2) plot[i*width + j + 2] = LOOP;
                if (new_int & 4) plot[i*width + j + 1] = LOOP;
                if (new_int & 8) plot[i*width + j] = LOOP;
            }
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void mandelbrot(int width, int height, int* plot) {
    naive_mandelbrot(width, height, plot);
}

#ifdef __cplusplus
}
#endif

/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/// AVX512-optimized BLAS Level 1 operations for the GELSD solver.
/// Matches the scalar LstSqGelsdBlas.hpp exactly, with AVX512 SIMD acceleration.

#pragma once

#include <cstddef>
#include <immintrin.h>

#include "LstSqGelsdSimdAttr.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AVX512-optimized dot product (double)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline double dot_d_avx512(const double *x, const double *y, size_t n) {
                __m512d sum = _mm512_setzero_pd();
                size_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m512d vx = _mm512_loadu_pd(x + i);
                    __m512d vy = _mm512_loadu_pd(y + i);
                    sum = _mm512_add_pd(sum, _mm512_mul_pd(vx, vy));
                }
                double s = _mm512_reduce_add_pd(sum);
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AVX512-optimized scale (double)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void scal_d_avx512(double a, double *x, size_t n) {
                __m512d va = _mm512_set1_pd(a);
                size_t i = 0;
                for (; i + 7 < n; i += 8) {
                    _mm512_storeu_pd(x + i, _mm512_mul_pd(va, _mm512_loadu_pd(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

            // ============================================================
            //  AVX512-optimized dot product (float)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline float dot_f_avx512(const float *x, const float *y, size_t n) {
                __m512 sum = _mm512_setzero_ps();
                size_t i = 0;
                for (; i + 15 < n; i += 16) {
                    __m512 vx = _mm512_loadu_ps(x + i);
                    __m512 vy = _mm512_loadu_ps(y + i);
                    sum = _mm512_add_ps(sum, _mm512_mul_ps(vx, vy));
                }
                float s = _mm512_reduce_add_ps(sum);
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AVX512-optimized scale (float)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void scal_f_avx512(float a, float *x, size_t n) {
                __m512 va = _mm512_set1_ps(a);
                size_t i = 0;
                for (; i + 15 < n; i += 16) {
                    _mm512_storeu_ps(x + i, _mm512_mul_ps(va, _mm512_loadu_ps(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

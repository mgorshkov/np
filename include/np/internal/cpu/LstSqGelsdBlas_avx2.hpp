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

/// AVX2-optimized BLAS Level 1 operations for the GELSD solver.
/// Matches the scalar LstSqGelsdBlas.hpp exactly, with AVX2 SIMD acceleration.

#pragma once

#include <cstddef>
#include <immintrin.h>

#include "LstSqGelsdSimdAttr.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AVX2-optimized dot product (double)
            // ============================================================

            AVX2_TARGET_ATTR
            static inline double dot_d_avx2(const double *x, const double *y, size_t n) {
                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();
                size_t i = 0;
                for (; i + 15 < n; i += 16) {
                    sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(_mm256_loadu_pd(x + i + 0), _mm256_loadu_pd(y + i + 0)));
                    sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(_mm256_loadu_pd(x + i + 4), _mm256_loadu_pd(y + i + 4)));
                    sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(_mm256_loadu_pd(x + i + 8), _mm256_loadu_pd(y + i + 8)));
                    sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(_mm256_loadu_pd(x + i + 12), _mm256_loadu_pd(y + i + 12)));
                }
                for (; i + 3 < n; i += 4) {
                    sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i)));
                }
                // Combine and reduce once
                __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                double s = _mm256_extractf128_pd(sum, 0)[0] + _mm256_extractf128_pd(sum, 0)[1] + _mm256_extractf128_pd(sum, 1)[0] + _mm256_extractf128_pd(sum, 1)[1];
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AVX2-optimized scale (double)
            // ============================================================

            AVX2_TARGET_ATTR
            static inline void scal_d_avx2(double a, double *x, size_t n) {
                __m256d va = _mm256_set1_pd(a);
                size_t i = 0;
                for (; i + 3 < n; i += 4) {
                    _mm256_storeu_pd(x + i, _mm256_mul_pd(va, _mm256_loadu_pd(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

            // ============================================================
            //  AVX2-optimized dot product (float)
            // ============================================================

            AVX2_TARGET_ATTR
            static inline float dot_f_avx2(const float *x, const float *y, size_t n) {
                __m256 sum0 = _mm256_setzero_ps();
                __m256 sum1 = _mm256_setzero_ps();
                __m256 sum2 = _mm256_setzero_ps();
                __m256 sum3 = _mm256_setzero_ps();
                size_t i = 0;
                for (; i + 31 < n; i += 32) {
                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(x + i + 0), _mm256_loadu_ps(y + i + 0)));
                    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
                    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(x + i + 16), _mm256_loadu_ps(y + i + 16)));
                    sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_loadu_ps(x + i + 24), _mm256_loadu_ps(y + i + 24)));
                }
                for (; i + 7 < n; i += 8) {
                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
                }
                __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                // Correct reduction of 8 floats to scalar:
                // sum = [a,b,c,d, e,f,g,h]
                // hi  = [a+e, b+f, c+g, d+h]
                // shuf = [c+g, d+h, a+e, b+f]
                // sum128 = [a+e+c+g, b+f+d+h, ...]
                // shuf2 = [b+f+d+h, a+e+c+g, ...]
                // sum128 = [a+b+c+d+e+f+g+h, ...]
                __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
                __m128 shuf = _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 sum128 = _mm_add_ps(hi, shuf);
                shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
                sum128 = _mm_add_ps(sum128, shuf);
                float s = _mm_cvtss_f32(sum128);
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AVX2-optimized scale (float)
            // ============================================================

            AVX2_TARGET_ATTR
            static inline void scal_f_avx2(float a, float *x, size_t n) {
                __m256 va = _mm256_set1_ps(a);
                size_t i = 0;
                for (; i + 7 < n; i += 8) {
                    _mm256_storeu_ps(x + i, _mm256_mul_ps(va, _mm256_loadu_ps(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

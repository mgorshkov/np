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

/// AVX2-optimized Householder reflection operations for the GELSD solver.
/// Matches the scalar LstSqGelsdHouseholder.hpp exactly, with AVX2 SIMD acceleration.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <immintrin.h>

#include "LstSqGelsdBlas_avx2.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AVX2-optimized Householder generate (double)
            //  Matches scalar householder_generate exactly, using AVX2 dot
            // ============================================================

            /// Generate a Householder reflector H = I - tau * v * v^T
            /// such that H * x = [alpha, 0, ..., 0]^T.
            ///
            /// This is the LAPACK DLARFG equivalent, matching the scalar
            /// householder_generate() exactly.
            ///
            /// LAPACK DLARFG formula:
            ///   x0_in = x[0] (original)
            ///   sigma = sum(x[1..n-1]^2)
            ///   beta = -sign(x0_in) * sqrt(x0_in^2 + sigma)
            ///   tau = (beta - x0_in) / beta
            ///   x[0] = 1
            ///   x[1..n-1] = x[1..n-1] / (x0_in - beta)
            ///
            /// In our variable naming:
            ///   alpha = beta (the new diagonal element)
            ///   beta = alpha - x0 = beta_lapack - x0_in
            ///   tau = beta / alpha = (beta_lapack - x0_in) / beta_lapack
            ///   x[1..n-1] = x[1..n-1] / (x0 - alpha) = x[1..n-1] / (-beta)
            AVX2_TARGET_ATTR
            static inline double householder_generate_d_avx2(double *x, size_t n, double *alpha_out) {
                if (n == 0) return 0.0;
                if (n == 1) {
                    double alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0;
                    return 0.0;
                }
                double x0 = x[0];
                double sigma = dot_d_avx2(x + 1, x + 1, n - 1);

                // alpha = -sign(x0) * sqrt(x0^2 + sigma)  (LAPACK's beta)
                // LAPACK DLARFG: beta = -sign(x0_in) * sqrt(x0_in^2 + sigma)
                // If x0 >= 0: alpha = -sqrt(...) (negative)
                // If x0 < 0:  alpha = +sqrt(...) (positive)
                double norm2 = x0 * x0 + sigma;
                double alpha = std::sqrt(norm2);
                if (x0 >= 0.0) alpha = -alpha;
                if (alpha_out) *alpha_out = alpha;

                // beta = alpha - x0  (LAPACK's beta - x0_in)
                double beta = alpha - x0;

                // Check for zero norm (x is already on the axis)
                if (beta == 0.0 && sigma == 0.0) {
                    x[0] = 1.0;
                    return 0.0;
                }

                // Store reflector: v[0] = 1 (implicit)
                // v[1..n-1] = x[1..n-1] / (x0 - alpha) = -x[1..n-1] / beta
                // This matches LAPACK DLARFG: x[i] = x[i] / (x0_in - beta_lapack)
                double inv_scale = 1.0 / (x0 - alpha);// = -1/beta
                x[0] = 1.0;
                for (size_t i = 1; i < n; ++i)
                    x[i] *= inv_scale;

                return beta / alpha;
            }

            // ============================================================
            //  Householder apply left (double, row-major)
            //
            //  NOTE: Uses SCALAR implementation, NOT AVX2 SIMD.
            //
            //  For left application, A[:,j] elements are strided by lda in
            //  row-major storage. Using _mm256_set_pd to gather 4 strided
            //  elements requires 4 separate scalar loads, which is actually
            //  SLOWER than scalar code. The compiler auto-vectorizes the
            //  scalar loop better for this strided access pattern.
            //
            //  The right application (householder_apply_right_d_avx2) uses
            //  contiguous row access and DOES benefit from AVX2 SIMD.
            // ============================================================

            /// Apply Householder reflector from the left: A = (I - tau*v*v^T) * A
            /// A is m x n stored row-major with leading dimension lda.
            /// v has length m (v[0] = 1 implicit).
            ///
            /// Uses scalar implementation because strided column access in
            /// row-major layout cannot benefit from AVX2 gather instructions.
            AVX2_TARGET_ATTR
            static inline void householder_apply_left_d_avx2(double tau, const double *v,
                                                             double *A, size_t m, size_t n,
                                                             size_t lda) {
                if (tau == 0.0) return;
                for (size_t j = 0; j < n; ++j) {
                    // Compute s = v^T * A[:,j] = sum_i v[i] * A[i*lda + j]
                    double s_val = 0.0;
                    for (size_t i = 0; i < m; ++i)
                        s_val += v[i] * A[i * lda + j];
                    s_val *= tau;
                    // A[:,j] -= s_val * v
                    for (size_t i = 0; i < m; ++i)
                        A[i * lda + j] -= s_val * v[i];
                }
            }

            // ============================================================
            //  AVX2-optimized Householder apply right (double, row-major)
            //  Matches scalar householder_apply_right structure exactly,
            //  with AVX2 SIMD in the inner loops.
            // ============================================================

            /// Apply Householder reflector from the right: A = A * (I - tau*v*v^T)
            /// A is m x n stored row-major with leading dimension lda.
            /// v has length n (v[0] = 1 implicit).
            AVX2_TARGET_ATTR
            static inline void householder_apply_right_d_avx2(double tau, const double *v,
                                                              double *A, size_t m, size_t n,
                                                              size_t lda) {
                if (tau == 0.0) return;
                for (size_t i = 0; i < m; ++i) {
                    // Use 4 accumulators to hide latency
                    __m256d sum0 = _mm256_setzero_pd();
                    __m256d sum1 = _mm256_setzero_pd();
                    __m256d sum2 = _mm256_setzero_pd();
                    __m256d sum3 = _mm256_setzero_pd();
                    size_t j = 0;
                    for (; j + 15 < n; j += 16) {
                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(_mm256_loadu_pd(&A[i * lda + j + 0]), _mm256_loadu_pd(v + j + 0)));
                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(_mm256_loadu_pd(&A[i * lda + j + 4]), _mm256_loadu_pd(v + j + 4)));
                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(_mm256_loadu_pd(&A[i * lda + j + 8]), _mm256_loadu_pd(v + j + 8)));
                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(_mm256_loadu_pd(&A[i * lda + j + 12]), _mm256_loadu_pd(v + j + 12)));
                    }
                    for (; j + 3 < n; j += 4) {
                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(_mm256_loadu_pd(&A[i * lda + j]), _mm256_loadu_pd(v + j)));
                    }
                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                    double s_val = _mm256_extractf128_pd(sum, 0)[0] + _mm256_extractf128_pd(sum, 0)[1] + _mm256_extractf128_pd(sum, 1)[0] + _mm256_extractf128_pd(sum, 1)[1];
                    for (; j < n; ++j) s_val += A[i * lda + j] * v[j];
                    s_val *= tau;
                    __m256d sv = _mm256_set1_pd(s_val);
                    j = 0;
                    for (; j + 3 < n; j += 4) {
                        __m256d aj = _mm256_loadu_pd(&A[i * lda + j]);
                        __m256d vj = _mm256_loadu_pd(v + j);
                        _mm256_storeu_pd(&A[i * lda + j], _mm256_sub_pd(aj, _mm256_mul_pd(sv, vj)));
                    }
                    for (; j < n; ++j) A[i * lda + j] -= s_val * v[j];
                }
            }

            // ============================================================
            //  AVX2-optimized Householder generate (float)
            //  Matches scalar householder_generate exactly, using AVX2 dot
            // ============================================================

            AVX2_TARGET_ATTR
            static inline float householder_generate_f_avx2(float *x, size_t n, float *alpha_out) {
                if (n == 0) return 0.0f;
                if (n == 1) {
                    float alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0f;
                    return 0.0f;
                }
                float x0 = x[0];
                float sigma = dot_f_avx2(x + 1, x + 1, n - 1);

                // alpha = -sign(x0) * sqrt(x0^2 + sigma)  (LAPACK's beta)
                // LAPACK DLARFG: beta = -sign(x0_in) * sqrt(x0_in^2 + sigma)
                // If x0 >= 0: alpha = -sqrt(...) (negative)
                // If x0 < 0:  alpha = +sqrt(...) (positive)
                float norm2 = x0 * x0 + sigma;
                float alpha = std::sqrt(norm2);
                if (x0 >= 0.0f) alpha = -alpha;
                if (alpha_out) *alpha_out = alpha;

                // beta = alpha - x0  (LAPACK's beta - x0_in)
                float beta = alpha - x0;

                // Check for zero norm (x is already on the axis)
                if (beta == 0.0f && sigma == 0.0f) {
                    x[0] = 1.0f;
                    return 0.0f;
                }

                // Store reflector: v[0] = 1 (implicit)
                // v[1..n-1] = x[1..n-1] / (x0 - alpha) = -x[1..n-1] / beta
                // This matches LAPACK DLARFG: x[i] = x[i] / (x0_in - beta_lapack)
                float inv_scale = 1.0f / (x0 - alpha);// = -1/beta
                x[0] = 1.0f;
                for (size_t i = 1; i < n; ++i)
                    x[i] *= inv_scale;

                return beta / alpha;
            }

            // ============================================================
            //  Householder apply left (float, row-major)
            //
            //  NOTE: Uses SCALAR implementation, NOT AVX2 SIMD.
            //
            //  For left application, A[:,j] elements are strided by lda in
            //  row-major storage. Using _mm256_set_ps to gather strided
            //  elements requires multiple scalar loads, which is actually
            //  SLOWER than scalar code. The compiler auto-vectorizes the
            //  scalar loop better for this strided access pattern.
            //
            //  The right application (householder_apply_right_f_avx2) uses
            //  contiguous row access and DOES benefit from AVX2 SIMD.
            // ============================================================

            AVX2_TARGET_ATTR
            static inline void householder_apply_left_f_avx2(float tau, const float *v,
                                                             float *A, size_t m, size_t n,
                                                             size_t lda) {
                if (tau == 0.0f) return;
                for (size_t j = 0; j < n; ++j) {
                    float s_val = 0.0f;
                    for (size_t i = 0; i < m; ++i)
                        s_val += v[i] * A[i * lda + j];
                    s_val *= tau;
                    for (size_t i = 0; i < m; ++i)
                        A[i * lda + j] -= s_val * v[i];
                }
            }

            // ============================================================
            //  AVX2-optimized Householder apply right (float, row-major)
            // ============================================================

            AVX2_TARGET_ATTR
            static inline void householder_apply_right_f_avx2(float tau, const float *v,
                                                              float *A, size_t m, size_t n,
                                                              size_t lda) {
                if (tau == 0.0f) return;
                for (size_t i = 0; i < m; ++i) {
                    __m256 sum0 = _mm256_setzero_ps();
                    __m256 sum1 = _mm256_setzero_ps();
                    __m256 sum2 = _mm256_setzero_ps();
                    __m256 sum3 = _mm256_setzero_ps();
                    size_t j = 0;
                    for (; j + 31 < n; j += 32) {
                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(&A[i * lda + j + 0]), _mm256_loadu_ps(v + j + 0)));
                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(_mm256_loadu_ps(&A[i * lda + j + 8]), _mm256_loadu_ps(v + j + 8)));
                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(_mm256_loadu_ps(&A[i * lda + j + 16]), _mm256_loadu_ps(v + j + 16)));
                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(_mm256_loadu_ps(&A[i * lda + j + 24]), _mm256_loadu_ps(v + j + 24)));
                    }
                    for (; j + 7 < n; j += 8) {
                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(_mm256_loadu_ps(&A[i * lda + j]), _mm256_loadu_ps(v + j)));
                    }
                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                    __m128 hi = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
                    __m128 lo = _mm_movehl_ps(hi, hi);
                    __m128 sum128 = _mm_add_ps(hi, lo);
                    float s_val = _mm_cvtss_f32(sum128);
                    for (; j < n; ++j) s_val += A[i * lda + j] * v[j];
                    s_val *= tau;
                    __m256 sv = _mm256_set1_ps(s_val);
                    j = 0;
                    for (; j + 7 < n; j += 8) {
                        __m256 aj = _mm256_loadu_ps(&A[i * lda + j]);
                        __m256 vj = _mm256_loadu_ps(v + j);
                        _mm256_storeu_ps(&A[i * lda + j], _mm256_sub_ps(aj, _mm256_mul_ps(sv, vj)));
                    }
                    for (; j < n; ++j) A[i * lda + j] -= s_val * v[j];
                }
            }

            // ============================================================
            //  Compact WY helpers (double, AVX2)
            //  Matches scalar GEMM helpers exactly, using AVX2 dot
            // ============================================================

            /// Block size for compact WY transforms (matches scalar)
            // Use preprocessor macro to avoid redefinition when both scalar and AVX2 headers are included
#ifndef NP_HOUSEHOLDER_BLOCK_NB_DEFINED
#define NP_HOUSEHOLDER_BLOCK_NB_DEFINED
            static constexpr size_t HOUSEHOLDER_BLOCK_NB = 32;
#endif

            // ---------------------------------------------------------------
            //  Helper: reduce a __m256d to scalar (extract + add)
            // ---------------------------------------------------------------
            AVX2_TARGET_ATTR
            static inline double reduce_add_pd(__m256d v) {
                return _mm256_extractf128_pd(v, 0)[0] + _mm256_extractf128_pd(v, 0)[1] + _mm256_extractf128_pd(v, 1)[0] + _mm256_extractf128_pd(v, 1)[1];
            }

            // ---------------------------------------------------------------
            //  Helper: reduce a __m256 to scalar
            // ---------------------------------------------------------------
            AVX2_TARGET_ATTR
            static inline float reduce_add_ps(__m256 v) {
                // Correct reduction of 8 floats to scalar:
                // v = [a,b,c,d, e,f,g,h]
                // hi = [a+e, b+f, c+g, d+h]
                // shuf = [c+g, d+h, a+e, b+f]
                // sum128 = [a+e+c+g, b+f+d+h, ...]
                // shuf2 = [b+f+d+h, a+e+c+g, ...]
                // sum128 = [a+b+c+d+e+f+g+h, ...]
                __m128 hi = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
                __m128 shuf = _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 sum128 = _mm_add_ps(hi, shuf);
                shuf = _mm_shuffle_ps(sum128, sum128, _MM_SHUFFLE(1, 0, 3, 2));
                sum128 = _mm_add_ps(sum128, shuf);
                return _mm_cvtss_f32(sum128);
            }

            /// C[m][n] += A[m][k] * B[k][n] (row-major C, column-major A, row-major B)
            /// A is column-major: A[row + p * lda]
            /// B is row-major: B[p * ldb + col]
            AVX2_TARGET_ATTR
            static inline void gemm_nn_add_d_avx2(double *C, size_t ldc,
                                                  const double *A, size_t lda,
                                                  const double *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    // 4-way unroll to hide latency
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < kr; p += 16) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 0) * ldb + (nc + j)])));
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 4) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 4) * ldb + (nc + j)])));
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 8) * ldb + (nc + j)])));
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 12) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 12) * ldb + (nc + j)])));
                                    }
                                    for (; p + 3 < kr; p += 4) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p) * ldb + (nc + j)])));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][n] -= A[m][k] * B[k][n]
            AVX2_TARGET_ATTR
            static inline void gemm_nn_sub_d_avx2(double *C, size_t ldc,
                                                  const double *A, size_t lda,
                                                  const double *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < kr; p += 16) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 0) * ldb + (nc + j)])));
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 4) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 4) * ldb + (nc + j)])));
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 8) * ldb + (nc + j)])));
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 12) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 12) * ldb + (nc + j)])));
                                    }
                                    for (; p + 3 < kr; p += 4) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_pd(&B[(kc + p) * ldb + (nc + j)])));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }


            /// C[m][n] += A[m][k] * B[n][k]^T  (B is column-major)
            /// i.e. C += A * B^T where B is column-major: B[col + row * ldb]
            AVX2_TARGET_ATTR
            static inline void gemm_nt_add_d_avx2(double *C, size_t ldc,
                                                  const double *A, size_t lda,
                                                  const double *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < kr; p += 16) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 0) * ldb])));
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 4) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 4) * ldb])));
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 8) * ldb])));
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 12) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 12) * ldb])));
                                    }
                                    for (; p + 3 < kr; p += 4) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p) * ldb])));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(nc + j) + (kc + p) * ldb];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][n] -= A[m][k] * B[n][k]^T  (B is column-major)
            AVX2_TARGET_ATTR
            static inline void gemm_nt_sub_d_avx2(double *C, size_t ldc,
                                                  const double *A, size_t lda,
                                                  const double *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < kr; p += 16) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 0) * ldb])));
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 4) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 4) * ldb])));
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 8) * ldb])));
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p + 12) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p + 12) * ldb])));
                                    }
                                    for (; p + 3 < kr; p += 4) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_pd(&B[(nc + j) + (kc + p) * ldb])));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(nc + j) + (kc + p) * ldb];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][k] += A[m][n] * B[n][k]  (row-major C, row-major A, column-major B)
            AVX2_TARGET_ATTR
            static inline void gemm_tn_add_d_avx2(double *C, size_t ldc,
                                                  const double *A, size_t lda,
                                                  const double *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < k; nc += NC) {
                        size_t nr = std::min(NC, k - nc);
                        for (size_t kc = 0; kc < n; kc += KC) {
                            size_t kr = std::min(KC, n - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < kr; p += 16) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row * lda + (kc + p + 0)]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 0) + (nc + j) * ldb])));
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row * lda + (kc + p + 4)]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 4) + (nc + j) * ldb])));
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row * lda + (kc + p + 8)]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 8) + (nc + j) * ldb])));
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row * lda + (kc + p + 12)]),
                                                                           _mm256_loadu_pd(&B[(kc + p + 12) + (nc + j) * ldb])));
                                    }
                                    for (; p + 3 < kr; p += 4) {
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(
                                                                           _mm256_loadu_pd(&A[row * lda + (kc + p)]),
                                                                           _mm256_loadu_pd(&B[(kc + p) + (nc + j) * ldb])));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row * lda + (kc + p)] * B[(kc + p) + (nc + j) * ldb];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }


            // ---------------------------------------------------------------
            //  Float GEMM helpers (AVX2, 32-element unroll)
            // ---------------------------------------------------------------

            /// C[m][n] += A[m][k] * B[k][n] (row-major C, column-major A, row-major B)
            AVX2_TARGET_ATTR
            static inline void gemm_nn_add_f_avx2(float *C, size_t ldc,
                                                  const float *A, size_t lda,
                                                  const float *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < kr; p += 32) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 0) * ldb + (nc + j)])));
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 8) * ldb + (nc + j)])));
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 16) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 16) * ldb + (nc + j)])));
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 24) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 24) * ldb + (nc + j)])));
                                    }
                                    for (; p + 7 < kr; p += 8) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p) * ldb + (nc + j)])));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][n] -= A[m][k] * B[k][n]
            AVX2_TARGET_ATTR
            static inline void gemm_nn_sub_f_avx2(float *C, size_t ldc,
                                                  const float *A, size_t lda,
                                                  const float *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < kr; p += 32) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 0) * ldb + (nc + j)])));
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 8) * ldb + (nc + j)])));
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 16) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 16) * ldb + (nc + j)])));
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 24) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 24) * ldb + (nc + j)])));
                                    }
                                    for (; p + 7 < kr; p += 8) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_ps(&B[(kc + p) * ldb + (nc + j)])));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][n] += A[m][k] * B[n][k]^T  (B is column-major)
            AVX2_TARGET_ATTR
            static inline void gemm_nt_add_f_avx2(float *C, size_t ldc,
                                                  const float *A, size_t lda,
                                                  const float *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < kr; p += 32) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 0) * ldb])));
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 8) * ldb])));
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 16) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 16) * ldb])));
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 24) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 24) * ldb])));
                                    }
                                    for (; p + 7 < kr; p += 8) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p) * ldb])));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(nc + j) + (kc + p) * ldb];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][n] -= A[m][k] * B[n][k]^T  (B is column-major)
            AVX2_TARGET_ATTR
            static inline void gemm_nt_sub_f_avx2(float *C, size_t ldc,
                                                  const float *A, size_t lda,
                                                  const float *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < n; nc += NC) {
                        size_t nr = std::min(NC, n - nc);
                        for (size_t kc = 0; kc < k; kc += KC) {
                            size_t kr = std::min(KC, k - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < kr; p += 32) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 0) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 0) * ldb])));
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 8) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 8) * ldb])));
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 16) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 16) * ldb])));
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p + 24) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p + 24) * ldb])));
                                    }
                                    for (; p + 7 < kr; p += 8) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row + (kc + p) * lda]),
                                                                           _mm256_loadu_ps(&B[(nc + j) + (kc + p) * ldb])));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row + (kc + p) * lda] * B[(nc + j) + (kc + p) * ldb];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

            /// C[m][k] += A[m][n] * B[n][k]  (row-major C, row-major A, column-major B)
            AVX2_TARGET_ATTR
            static inline void gemm_tn_add_f_avx2(float *C, size_t ldc,
                                                  const float *A, size_t lda,
                                                  const float *B, size_t ldb,
                                                  size_t m, size_t n, size_t k) {
                constexpr size_t MC = 64;
                constexpr size_t NC = 256;
                constexpr size_t KC = 256;

                for (size_t mc = 0; mc < m; mc += MC) {
                    size_t mr = std::min(MC, m - mc);
                    for (size_t nc = 0; nc < k; nc += NC) {
                        size_t nr = std::min(NC, k - nc);
                        for (size_t kc = 0; kc < n; kc += KC) {
                            size_t kr = std::min(KC, n - kc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < kr; p += 32) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row * lda + (kc + p + 0)]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 0) + (nc + j) * ldb])));
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row * lda + (kc + p + 8)]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 8) + (nc + j) * ldb])));
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row * lda + (kc + p + 16)]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 16) + (nc + j) * ldb])));
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row * lda + (kc + p + 24)]),
                                                                           _mm256_loadu_ps(&B[(kc + p + 24) + (nc + j) * ldb])));
                                    }
                                    for (; p + 7 < kr; p += 8) {
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(
                                                                           _mm256_loadu_ps(&A[row * lda + (kc + p)]),
                                                                           _mm256_loadu_ps(&B[(kc + p) + (nc + j) * ldb])));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < kr; ++p)
                                        s_val += A[row * lda + (kc + p)] * B[(kc + p) + (nc + j) * ldb];
                                    C[row * ldc + (nc + j)] += s_val;
                                }
                            }
                        }
                    }
                }
            }


            // ---------------------------------------------------------------
            //  Form the T matrix (double, AVX2)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors.
            /// Y is m x NB stored column-major.
            /// tau[0..NB-1] are the scalar factors.
            /// T is NB x NB stored column-major.
            AVX2_TARGET_ATTR
            /// Form the T matrix for a block of NB Householder reflectors.
            /// Y is m x NB stored column-major.
            /// tau[0..NB-1] are the scalar factors.
            /// T is NB x NB stored column-major.
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            ///
            /// The triangular solve (T^{-1} * w) is the correct formula for the
            /// compact WY representation and works for ANY Y matrix structure.
            /// The alternative formula (T * w) is only valid when Y has the
            /// specific unit lower trapezoidal structure Y[i,i] = 1.
            AVX2_TARGET_ATTR
            static inline void larft_d_avx2(const double *Y, size_t m, size_t NB,
                                            const double *tau, size_t ldy,
                                            double *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    // Step 1: w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
                    // NOTE: We use a scalar dot product here instead of dot_d_avx2
                    // to ensure bit-identical results with the scalar larft function.
                    // The larft algorithm is numerically sensitive: small differences
                    // in dot product results (due to different summation order in AVX2)
                    // propagate through the triangular solve (Step 2) and amplify
                    // across iterations, causing O(1e-4) errors in the T matrix for
                    // blocks with large m (e.g., m=95). Since NB <= 32 and larft is
                    // O(NB^2 * m), the scalar dot product has negligible performance
                    // impact compared to the O(NB * m * n) larfb operations.
                    for (size_t i = 0; i < j; ++i) {
                        double w_i = 0.0;
                        for (size_t k = 0; k < m; ++k)
                            w_i += Y[k + i * ldy] * Y[k + j * ldy];
                        T_[i + j * ldT] = w_i;
                    }

                    // Step 2: w = T[0:j,0:j] * w  (matrix-vector multiply, DTRMV equivalent)
                    // T is upper-triangular column-major: T[i,k] = T_[i + k*ldT]
                    // T[i,k] = 0 for k < i (upper triangular).
                    // w_new[i] = sum_{k=i..j-1} T[i,k] * w[k]
                    for (size_t i = 0; i < j; ++i) {
                        double sum = 0.0;
                        for (size_t k = i; k < j; ++k)
                            sum += T_[i + k * ldT] * T_[k + j * ldT];
                        T_[i + j * ldT] = sum;
                    }

                    // Step 3: T[0:j, j] = -tau[j] * x
                    double tau_j = tau[j];
                    for (size_t i = 0; i < j; ++i)
                        T_[i + j * ldT] *= -tau_j;

                    // Step 4: T[j, j] = tau[j]
                    T_[j + j * ldT] = tau_j;
                }
            }

            // ---------------------------------------------------------------
            //  LAPACK-style blocked application (double, AVX2)
            // ---------------------------------------------------------------

            /// Apply a block of NB Householder reflectors from the LEFT:
            ///   C = (I - Y * T * Y^T) * C
            ///
            /// Y is m x NB stored column-major with leading dimension ldy (>= m).
            /// T is NB x NB stored column-major with leading dimension ldT (>= NB).
            /// C is m x n stored row-major with leading dimension ldc.
            AVX2_TARGET_ATTR
            static inline void larfb_left_d_avx2(const double *Y, size_t m, size_t NB,
                                                 const double *T_, size_t ldT,
                                                 double *C, size_t n, size_t ldc,
                                                 size_t ldy) {
                if (ldy == 0) ldy = m;
                if (NB == 0 || n == 0) return;

                std::vector<double> W(NB * n, 0.0);

                // Step 1: W = Y^T * C (NB x n)
                // W[i, j] = sum_{k=0..m-1} Y[k, i] * C[k, j]
                // Y is column-major (contiguous), C is row-major (strided by ldc).
                // Must gather C elements manually.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < m; kc += KC) {
                        size_t kr = std::min(KC, m - kc);
                        for (size_t i = 0; i < NB; ++i) {
                            const double *Y_col = Y + kc + i * ldy;
                            for (size_t j = 0; j < n; ++j) {
                                __m256d sum0 = _mm256_setzero_pd();
                                __m256d sum1 = _mm256_setzero_pd();
                                __m256d sum2 = _mm256_setzero_pd();
                                __m256d sum3 = _mm256_setzero_pd();
                                size_t k = 0;
                                for (; k + 15 < kr; k += 16) {
                                    __m256d yv0 = _mm256_loadu_pd(Y_col + k + 0);
                                    __m256d cv0 = _mm256_set_pd(C[(kc + k + 3) * ldc + j],
                                                                C[(kc + k + 2) * ldc + j],
                                                                C[(kc + k + 1) * ldc + j],
                                                                C[(kc + k + 0) * ldc + j]);
                                    sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(yv0, cv0));
                                    __m256d yv1 = _mm256_loadu_pd(Y_col + k + 4);
                                    __m256d cv1 = _mm256_set_pd(C[(kc + k + 7) * ldc + j],
                                                                C[(kc + k + 6) * ldc + j],
                                                                C[(kc + k + 5) * ldc + j],
                                                                C[(kc + k + 4) * ldc + j]);
                                    sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(yv1, cv1));
                                    __m256d yv2 = _mm256_loadu_pd(Y_col + k + 8);
                                    __m256d cv2 = _mm256_set_pd(C[(kc + k + 11) * ldc + j],
                                                                C[(kc + k + 10) * ldc + j],
                                                                C[(kc + k + 9) * ldc + j],
                                                                C[(kc + k + 8) * ldc + j]);
                                    sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(yv2, cv2));
                                    __m256d yv3 = _mm256_loadu_pd(Y_col + k + 12);
                                    __m256d cv3 = _mm256_set_pd(C[(kc + k + 15) * ldc + j],
                                                                C[(kc + k + 14) * ldc + j],
                                                                C[(kc + k + 13) * ldc + j],
                                                                C[(kc + k + 12) * ldc + j]);
                                    sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(yv3, cv3));
                                }
                                for (; k + 3 < kr; k += 4) {
                                    __m256d yv = _mm256_loadu_pd(Y_col + k);
                                    __m256d cv = _mm256_set_pd(C[(kc + k + 3) * ldc + j],
                                                               C[(kc + k + 2) * ldc + j],
                                                               C[(kc + k + 1) * ldc + j],
                                                               C[(kc + k + 0) * ldc + j]);
                                    sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(yv, cv));
                                }
                                __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                double s_val = reduce_add_pd(sum);
                                for (; k < kr; ++k)
                                    s_val += Y_col[k] * C[(kc + k) * ldc + j];
                                W[i * n + j] += s_val;
                            }
                        }
                    }
                }

                // Step 2: W = T * W (NB x n)
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < NB; ++i) {
                        double sum = 0.0;
                        for (size_t k = i; k < NB; ++k)
                            sum += T_[i + k * ldT] * W[k * n + j];
                        W[i * n + j] = sum;
                    }
                }

                // Step 3: C = C - Y * W (m x n)
                // Y is column-major: Y[row, p] = Y[row + p * ldy], strided by ldy.
                // W is NB x n row-major: W[p, j] = W[p * n + j], strided by n.
                // Both must be gathered manually.
                {
                    constexpr size_t MC = 64;
                    constexpr size_t NC = 256;
                    for (size_t mc = 0; mc < m; mc += MC) {
                        size_t mr = std::min(MC, m - mc);
                        for (size_t nc = 0; nc < n; nc += NC) {
                            size_t nr = std::min(NC, n - nc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < NB; p += 16) {
                                        __m256d yv0 = _mm256_set_pd(Y[row + (p + 3) * ldy],
                                                                    Y[row + (p + 2) * ldy],
                                                                    Y[row + (p + 1) * ldy],
                                                                    Y[row + p * ldy]);
                                        __m256d wv0 = _mm256_set_pd(W[(p + 3) * n + (nc + j)],
                                                                    W[(p + 2) * n + (nc + j)],
                                                                    W[(p + 1) * n + (nc + j)],
                                                                    W[p * n + (nc + j)]);
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(yv0, wv0));
                                        __m256d yv1 = _mm256_set_pd(Y[row + (p + 7) * ldy],
                                                                    Y[row + (p + 6) * ldy],
                                                                    Y[row + (p + 5) * ldy],
                                                                    Y[row + (p + 4) * ldy]);
                                        __m256d wv1 = _mm256_set_pd(W[(p + 7) * n + (nc + j)],
                                                                    W[(p + 6) * n + (nc + j)],
                                                                    W[(p + 5) * n + (nc + j)],
                                                                    W[(p + 4) * n + (nc + j)]);
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(yv1, wv1));
                                        __m256d yv2 = _mm256_set_pd(Y[row + (p + 11) * ldy],
                                                                    Y[row + (p + 10) * ldy],
                                                                    Y[row + (p + 9) * ldy],
                                                                    Y[row + (p + 8) * ldy]);
                                        __m256d wv2 = _mm256_set_pd(W[(p + 11) * n + (nc + j)],
                                                                    W[(p + 10) * n + (nc + j)],
                                                                    W[(p + 9) * n + (nc + j)],
                                                                    W[(p + 8) * n + (nc + j)]);
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(yv2, wv2));
                                        __m256d yv3 = _mm256_set_pd(Y[row + (p + 15) * ldy],
                                                                    Y[row + (p + 14) * ldy],
                                                                    Y[row + (p + 13) * ldy],
                                                                    Y[row + (p + 12) * ldy]);
                                        __m256d wv3 = _mm256_set_pd(W[(p + 15) * n + (nc + j)],
                                                                    W[(p + 14) * n + (nc + j)],
                                                                    W[(p + 13) * n + (nc + j)],
                                                                    W[(p + 12) * n + (nc + j)]);
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(yv3, wv3));
                                    }
                                    for (; p + 3 < NB; p += 4) {
                                        __m256d yv = _mm256_set_pd(Y[row + (p + 3) * ldy],
                                                                   Y[row + (p + 2) * ldy],
                                                                   Y[row + (p + 1) * ldy],
                                                                   Y[row + p * ldy]);
                                        __m256d wv = _mm256_set_pd(W[(p + 3) * n + (nc + j)],
                                                                   W[(p + 2) * n + (nc + j)],
                                                                   W[(p + 1) * n + (nc + j)],
                                                                   W[p * n + (nc + j)]);
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(yv, wv));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < NB; ++p)
                                        s_val += Y[row + p * ldy] * W[p * n + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }


            /// Apply a block of NB Householder reflectors from the RIGHT:
            ///   C = C * (I - Y * T * Y^T)
            ///
            /// Y is n x NB stored column-major with leading dimension ldy (>= n).
            /// T is NB x NB stored column-major with leading dimension ldT (>= NB).
            /// C is m x n stored row-major with leading dimension ldc.
            AVX2_TARGET_ATTR
            static inline void larfb_right_d_avx2(const double *Y, size_t n, size_t NB,
                                                  const double *T_, size_t ldT,
                                                  double *C, size_t m, size_t ldc,
                                                  size_t ldy) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                std::vector<double> W(m * NB, 0.0);

                // Step 1: W = C * Y (m x NB)
                // C is row-major (contiguous within a row), Y is column-major (contiguous down columns).
                // Both are contiguous, so dot_d_avx2 is correct here.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                double sum = dot_d_avx2(C + p * ldc + kc, Y + kc + i * ldy, kr);
                                W[p * NB + i] += sum;
                            }
                        }
                    }
                }

                // Step 2: W = W * T (m x NB)
                for (size_t p = 0; p < m; ++p) {
                    for (size_t i = NB; i > 0;) {
                        --i;
                        double sum = 0.0;
                        for (size_t k = 0; k <= i; ++k)
                            sum += W[p * NB + k] * T_[k + i * ldT];
                        W[p * NB + i] = sum;
                    }
                }

                // Step 3: C = C - W * Y^T (m x n)
                // Y is column-major: Y[j, p] = Y[j + p * ldy], strided by ldy.
                // Must gather Y elements manually.
                {
                    constexpr size_t MC = 64;
                    constexpr size_t NC = 256;
                    for (size_t mc = 0; mc < m; mc += MC) {
                        size_t mr = std::min(MC, m - mc);
                        for (size_t nc = 0; nc < n; nc += NC) {
                            size_t nr = std::min(NC, n - nc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256d sum0 = _mm256_setzero_pd();
                                    __m256d sum1 = _mm256_setzero_pd();
                                    __m256d sum2 = _mm256_setzero_pd();
                                    __m256d sum3 = _mm256_setzero_pd();
                                    size_t p = 0;
                                    for (; p + 15 < NB; p += 16) {
                                        __m256d wv0 = _mm256_loadu_pd(&W[row * NB + p + 0]);
                                        __m256d yv0 = _mm256_set_pd(Y[(nc + j) + (p + 3) * ldy],
                                                                    Y[(nc + j) + (p + 2) * ldy],
                                                                    Y[(nc + j) + (p + 1) * ldy],
                                                                    Y[(nc + j) + p * ldy]);
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(wv0, yv0));
                                        __m256d wv1 = _mm256_loadu_pd(&W[row * NB + p + 4]);
                                        __m256d yv1 = _mm256_set_pd(Y[(nc + j) + (p + 7) * ldy],
                                                                    Y[(nc + j) + (p + 6) * ldy],
                                                                    Y[(nc + j) + (p + 5) * ldy],
                                                                    Y[(nc + j) + (p + 4) * ldy]);
                                        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(wv1, yv1));
                                        __m256d wv2 = _mm256_loadu_pd(&W[row * NB + p + 8]);
                                        __m256d yv2 = _mm256_set_pd(Y[(nc + j) + (p + 11) * ldy],
                                                                    Y[(nc + j) + (p + 10) * ldy],
                                                                    Y[(nc + j) + (p + 9) * ldy],
                                                                    Y[(nc + j) + (p + 8) * ldy]);
                                        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(wv2, yv2));
                                        __m256d wv3 = _mm256_loadu_pd(&W[row * NB + p + 12]);
                                        __m256d yv3 = _mm256_set_pd(Y[(nc + j) + (p + 15) * ldy],
                                                                    Y[(nc + j) + (p + 14) * ldy],
                                                                    Y[(nc + j) + (p + 13) * ldy],
                                                                    Y[(nc + j) + (p + 12) * ldy]);
                                        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(wv3, yv3));
                                    }
                                    for (; p + 3 < NB; p += 4) {
                                        __m256d wv = _mm256_loadu_pd(&W[row * NB + p]);
                                        __m256d yv = _mm256_set_pd(Y[(nc + j) + (p + 3) * ldy],
                                                                   Y[(nc + j) + (p + 2) * ldy],
                                                                   Y[(nc + j) + (p + 1) * ldy],
                                                                   Y[(nc + j) + p * ldy]);
                                        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(wv, yv));
                                    }
                                    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));
                                    double s_val = reduce_add_pd(sum);
                                    for (; p < NB; ++p)
                                        s_val += W[row * NB + p] * Y[(nc + j) + p * ldy];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }


            // ---------------------------------------------------------------
            //  Form the T matrix (float, AVX2)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors (float).
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            AVX2_TARGET_ATTR
            static inline void larft_f_avx2(const float *Y, size_t m, size_t NB,
                                            const float *tau, size_t ldy,
                                            float *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    for (size_t i = 0; i < j; ++i) {
                        float w_i = dot_f_avx2(Y + i * ldy, Y + j * ldy, m);
                        T_[i + j * ldT] = w_i;
                    }

                    // Step 2: w = T[0:j,0:j] * w  (matrix-vector multiply, DTRMV equivalent)
                    // T is upper-triangular column-major: T[i,k] = T_[i + k*ldT]
                    // T[i,k] = 0 for k < i (upper triangular).
                    // w_new[i] = sum_{k=i..j-1} T[i,k] * w[k]
                    for (size_t i = 0; i < j; ++i) {
                        float sum = 0.0f;
                        for (size_t k = i; k < j; ++k)
                            sum += T_[i + k * ldT] * T_[k + j * ldT];
                        T_[i + j * ldT] = sum;
                    }

                    float tau_j = tau[j];
                    for (size_t i = 0; i < j; ++i)
                        T_[i + j * ldT] *= -tau_j;

                    T_[j + j * ldT] = tau_j;
                }
            }

            // ---------------------------------------------------------------
            //  LAPACK-style blocked application (float, AVX2)
            // ---------------------------------------------------------------

            /// Apply a block of NB Householder reflectors from the LEFT (float).
            AVX2_TARGET_ATTR
            static inline void larfb_left_f_avx2(const float *Y, size_t m, size_t NB,
                                                 const float *T_, size_t ldT,
                                                 float *C, size_t n, size_t ldc,
                                                 size_t ldy) {
                if (ldy == 0) ldy = m;
                if (NB == 0 || n == 0) return;

                std::vector<float> W(NB * n, 0.0f);

                // Step 1: W = Y^T * C (NB x n)
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < m; kc += KC) {
                        size_t kr = std::min(KC, m - kc);
                        for (size_t i = 0; i < NB; ++i) {
                            const float *Y_col = Y + kc + i * ldy;
                            for (size_t j = 0; j < n; ++j) {
                                __m256 sum0 = _mm256_setzero_ps();
                                __m256 sum1 = _mm256_setzero_ps();
                                __m256 sum2 = _mm256_setzero_ps();
                                __m256 sum3 = _mm256_setzero_ps();
                                size_t k = 0;
                                for (; k + 31 < kr; k += 32) {
                                    __m256 yv0 = _mm256_loadu_ps(Y_col + k + 0);
                                    __m256 cv0 = _mm256_set_ps(C[(kc + k + 7) * ldc + j],
                                                               C[(kc + k + 6) * ldc + j],
                                                               C[(kc + k + 5) * ldc + j],
                                                               C[(kc + k + 4) * ldc + j],
                                                               C[(kc + k + 3) * ldc + j],
                                                               C[(kc + k + 2) * ldc + j],
                                                               C[(kc + k + 1) * ldc + j],
                                                               C[(kc + k + 0) * ldc + j]);
                                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(yv0, cv0));
                                    __m256 yv1 = _mm256_loadu_ps(Y_col + k + 8);
                                    __m256 cv1 = _mm256_set_ps(C[(kc + k + 15) * ldc + j],
                                                               C[(kc + k + 14) * ldc + j],
                                                               C[(kc + k + 13) * ldc + j],
                                                               C[(kc + k + 12) * ldc + j],
                                                               C[(kc + k + 11) * ldc + j],
                                                               C[(kc + k + 10) * ldc + j],
                                                               C[(kc + k + 9) * ldc + j],
                                                               C[(kc + k + 8) * ldc + j]);
                                    sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(yv1, cv1));
                                    __m256 yv2 = _mm256_loadu_ps(Y_col + k + 16);
                                    __m256 cv2 = _mm256_set_ps(C[(kc + k + 23) * ldc + j],
                                                               C[(kc + k + 22) * ldc + j],
                                                               C[(kc + k + 21) * ldc + j],
                                                               C[(kc + k + 20) * ldc + j],
                                                               C[(kc + k + 19) * ldc + j],
                                                               C[(kc + k + 18) * ldc + j],
                                                               C[(kc + k + 17) * ldc + j],
                                                               C[(kc + k + 16) * ldc + j]);
                                    sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(yv2, cv2));
                                    __m256 yv3 = _mm256_loadu_ps(Y_col + k + 24);
                                    __m256 cv3 = _mm256_set_ps(C[(kc + k + 31) * ldc + j],
                                                               C[(kc + k + 30) * ldc + j],
                                                               C[(kc + k + 29) * ldc + j],
                                                               C[(kc + k + 28) * ldc + j],
                                                               C[(kc + k + 27) * ldc + j],
                                                               C[(kc + k + 26) * ldc + j],
                                                               C[(kc + k + 25) * ldc + j],
                                                               C[(kc + k + 24) * ldc + j]);
                                    sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(yv3, cv3));
                                }
                                for (; k + 7 < kr; k += 8) {
                                    __m256 yv = _mm256_loadu_ps(Y_col + k);
                                    __m256 cv = _mm256_set_ps(C[(kc + k + 7) * ldc + j],
                                                              C[(kc + k + 6) * ldc + j],
                                                              C[(kc + k + 5) * ldc + j],
                                                              C[(kc + k + 4) * ldc + j],
                                                              C[(kc + k + 3) * ldc + j],
                                                              C[(kc + k + 2) * ldc + j],
                                                              C[(kc + k + 1) * ldc + j],
                                                              C[(kc + k + 0) * ldc + j]);
                                    sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(yv, cv));
                                }
                                __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                float s_val = reduce_add_ps(sum);
                                for (; k < kr; ++k)
                                    s_val += Y_col[k] * C[(kc + k) * ldc + j];
                                W[i * n + j] += s_val;
                            }
                        }
                    }
                }

                // Step 2: W = T * W
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < NB; ++i) {
                        float sum = 0.0f;
                        for (size_t k = i; k < NB; ++k)
                            sum += T_[i + k * ldT] * W[k * n + j];
                        W[i * n + j] = sum;
                    }
                }

                // Step 3: C = C - Y * W (m x n)
                {
                    constexpr size_t MC = 64;
                    constexpr size_t NC = 256;
                    for (size_t mc = 0; mc < m; mc += MC) {
                        size_t mr = std::min(MC, m - mc);
                        for (size_t nc = 0; nc < n; nc += NC) {
                            size_t nr = std::min(NC, n - nc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < NB; p += 32) {
                                        __m256 yv0 = _mm256_set_ps(Y[row + (p + 7) * ldy],
                                                                   Y[row + (p + 6) * ldy],
                                                                   Y[row + (p + 5) * ldy],
                                                                   Y[row + (p + 4) * ldy],
                                                                   Y[row + (p + 3) * ldy],
                                                                   Y[row + (p + 2) * ldy],
                                                                   Y[row + (p + 1) * ldy],
                                                                   Y[row + p * ldy]);
                                        __m256 wv0 = _mm256_set_ps(W[(p + 7) * n + (nc + j)],
                                                                   W[(p + 6) * n + (nc + j)],
                                                                   W[(p + 5) * n + (nc + j)],
                                                                   W[(p + 4) * n + (nc + j)],
                                                                   W[(p + 3) * n + (nc + j)],
                                                                   W[(p + 2) * n + (nc + j)],
                                                                   W[(p + 1) * n + (nc + j)],
                                                                   W[p * n + (nc + j)]);
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(yv0, wv0));
                                        __m256 yv1 = _mm256_set_ps(Y[row + (p + 15) * ldy],
                                                                   Y[row + (p + 14) * ldy],
                                                                   Y[row + (p + 13) * ldy],
                                                                   Y[row + (p + 12) * ldy],
                                                                   Y[row + (p + 11) * ldy],
                                                                   Y[row + (p + 10) * ldy],
                                                                   Y[row + (p + 9) * ldy],
                                                                   Y[row + (p + 8) * ldy]);
                                        __m256 wv1 = _mm256_set_ps(W[(p + 15) * n + (nc + j)],
                                                                   W[(p + 14) * n + (nc + j)],
                                                                   W[(p + 13) * n + (nc + j)],
                                                                   W[(p + 12) * n + (nc + j)],
                                                                   W[(p + 11) * n + (nc + j)],
                                                                   W[(p + 10) * n + (nc + j)],
                                                                   W[(p + 9) * n + (nc + j)],
                                                                   W[(p + 8) * n + (nc + j)]);
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(yv1, wv1));
                                        __m256 yv2 = _mm256_set_ps(Y[row + (p + 23) * ldy],
                                                                   Y[row + (p + 22) * ldy],
                                                                   Y[row + (p + 21) * ldy],
                                                                   Y[row + (p + 20) * ldy],
                                                                   Y[row + (p + 19) * ldy],
                                                                   Y[row + (p + 18) * ldy],
                                                                   Y[row + (p + 17) * ldy],
                                                                   Y[row + (p + 16) * ldy]);
                                        __m256 wv2 = _mm256_set_ps(W[(p + 23) * n + (nc + j)],
                                                                   W[(p + 22) * n + (nc + j)],
                                                                   W[(p + 21) * n + (nc + j)],
                                                                   W[(p + 20) * n + (nc + j)],
                                                                   W[(p + 19) * n + (nc + j)],
                                                                   W[(p + 18) * n + (nc + j)],
                                                                   W[(p + 17) * n + (nc + j)],
                                                                   W[(p + 16) * n + (nc + j)]);
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(yv2, wv2));
                                        __m256 yv3 = _mm256_set_ps(Y[row + (p + 31) * ldy],
                                                                   Y[row + (p + 30) * ldy],
                                                                   Y[row + (p + 29) * ldy],
                                                                   Y[row + (p + 28) * ldy],
                                                                   Y[row + (p + 27) * ldy],
                                                                   Y[row + (p + 26) * ldy],
                                                                   Y[row + (p + 25) * ldy],
                                                                   Y[row + (p + 24) * ldy]);
                                        __m256 wv3 = _mm256_set_ps(W[(p + 31) * n + (nc + j)],
                                                                   W[(p + 30) * n + (nc + j)],
                                                                   W[(p + 29) * n + (nc + j)],
                                                                   W[(p + 28) * n + (nc + j)],
                                                                   W[(p + 27) * n + (nc + j)],
                                                                   W[(p + 26) * n + (nc + j)],
                                                                   W[(p + 25) * n + (nc + j)],
                                                                   W[(p + 24) * n + (nc + j)]);
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(yv3, wv3));
                                    }
                                    for (; p + 7 < NB; p += 8) {
                                        __m256 yv = _mm256_set_ps(Y[row + (p + 7) * ldy],
                                                                  Y[row + (p + 6) * ldy],
                                                                  Y[row + (p + 5) * ldy],
                                                                  Y[row + (p + 4) * ldy],
                                                                  Y[row + (p + 3) * ldy],
                                                                  Y[row + (p + 2) * ldy],
                                                                  Y[row + (p + 1) * ldy],
                                                                  Y[row + p * ldy]);
                                        __m256 wv = _mm256_set_ps(W[(p + 7) * n + (nc + j)],
                                                                  W[(p + 6) * n + (nc + j)],
                                                                  W[(p + 5) * n + (nc + j)],
                                                                  W[(p + 4) * n + (nc + j)],
                                                                  W[(p + 3) * n + (nc + j)],
                                                                  W[(p + 2) * n + (nc + j)],
                                                                  W[(p + 1) * n + (nc + j)],
                                                                  W[p * n + (nc + j)]);
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(yv, wv));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < NB; ++p)
                                        s_val += Y[row + p * ldy] * W[p * n + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }


            /// Apply a block of NB Householder reflectors from the RIGHT (float).
            AVX2_TARGET_ATTR
            static inline void larfb_right_f_avx2(const float *Y, size_t n, size_t NB,
                                                  const float *T_, size_t ldT,
                                                  float *C, size_t m, size_t ldc,
                                                  size_t ldy) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                std::vector<float> W(m * NB, 0.0f);

                // Step 1: W = C * Y (m x NB)
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                float sum = dot_f_avx2(C + p * ldc + kc, Y + kc + i * ldy, kr);
                                W[p * NB + i] += sum;
                            }
                        }
                    }
                }

                // Step 2: W = W * T (m x NB)
                for (size_t p = 0; p < m; ++p) {
                    for (size_t i = NB; i > 0;) {
                        --i;
                        float sum = 0.0f;
                        for (size_t k = 0; k <= i; ++k)
                            sum += W[p * NB + k] * T_[k + i * ldT];
                        W[p * NB + i] = sum;
                    }
                }

                // Step 3: C = C - W * Y^T (m x n)
                {
                    constexpr size_t MC = 64;
                    constexpr size_t NC = 256;
                    for (size_t mc = 0; mc < m; mc += MC) {
                        size_t mr = std::min(MC, m - mc);
                        for (size_t nc = 0; nc < n; nc += NC) {
                            size_t nr = std::min(NC, n - nc);
                            for (size_t i = 0; i < mr; ++i) {
                                size_t row = mc + i;
                                for (size_t j = 0; j < nr; ++j) {
                                    __m256 sum0 = _mm256_setzero_ps();
                                    __m256 sum1 = _mm256_setzero_ps();
                                    __m256 sum2 = _mm256_setzero_ps();
                                    __m256 sum3 = _mm256_setzero_ps();
                                    size_t p = 0;
                                    for (; p + 31 < NB; p += 32) {
                                        __m256 wv0 = _mm256_loadu_ps(&W[row * NB + p + 0]);
                                        __m256 yv0 = _mm256_set_ps(Y[(nc + j) + (p + 7) * ldy],
                                                                   Y[(nc + j) + (p + 6) * ldy],
                                                                   Y[(nc + j) + (p + 5) * ldy],
                                                                   Y[(nc + j) + (p + 4) * ldy],
                                                                   Y[(nc + j) + (p + 3) * ldy],
                                                                   Y[(nc + j) + (p + 2) * ldy],
                                                                   Y[(nc + j) + (p + 1) * ldy],
                                                                   Y[(nc + j) + p * ldy]);
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(wv0, yv0));
                                        __m256 wv1 = _mm256_loadu_ps(&W[row * NB + p + 8]);
                                        __m256 yv1 = _mm256_set_ps(Y[(nc + j) + (p + 15) * ldy],
                                                                   Y[(nc + j) + (p + 14) * ldy],
                                                                   Y[(nc + j) + (p + 13) * ldy],
                                                                   Y[(nc + j) + (p + 12) * ldy],
                                                                   Y[(nc + j) + (p + 11) * ldy],
                                                                   Y[(nc + j) + (p + 10) * ldy],
                                                                   Y[(nc + j) + (p + 9) * ldy],
                                                                   Y[(nc + j) + (p + 8) * ldy]);
                                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(wv1, yv1));
                                        __m256 wv2 = _mm256_loadu_ps(&W[row * NB + p + 16]);
                                        __m256 yv2 = _mm256_set_ps(Y[(nc + j) + (p + 23) * ldy],
                                                                   Y[(nc + j) + (p + 22) * ldy],
                                                                   Y[(nc + j) + (p + 21) * ldy],
                                                                   Y[(nc + j) + (p + 20) * ldy],
                                                                   Y[(nc + j) + (p + 19) * ldy],
                                                                   Y[(nc + j) + (p + 18) * ldy],
                                                                   Y[(nc + j) + (p + 17) * ldy],
                                                                   Y[(nc + j) + (p + 16) * ldy]);
                                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(wv2, yv2));
                                        __m256 wv3 = _mm256_loadu_ps(&W[row * NB + p + 24]);
                                        __m256 yv3 = _mm256_set_ps(Y[(nc + j) + (p + 31) * ldy],
                                                                   Y[(nc + j) + (p + 30) * ldy],
                                                                   Y[(nc + j) + (p + 29) * ldy],
                                                                   Y[(nc + j) + (p + 28) * ldy],
                                                                   Y[(nc + j) + (p + 27) * ldy],
                                                                   Y[(nc + j) + (p + 26) * ldy],
                                                                   Y[(nc + j) + (p + 25) * ldy],
                                                                   Y[(nc + j) + (p + 24) * ldy]);
                                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(wv3, yv3));
                                    }
                                    for (; p + 7 < NB; p += 8) {
                                        __m256 wv = _mm256_loadu_ps(&W[row * NB + p]);
                                        __m256 yv = _mm256_set_ps(Y[(nc + j) + (p + 7) * ldy],
                                                                  Y[(nc + j) + (p + 6) * ldy],
                                                                  Y[(nc + j) + (p + 5) * ldy],
                                                                  Y[(nc + j) + (p + 4) * ldy],
                                                                  Y[(nc + j) + (p + 3) * ldy],
                                                                  Y[(nc + j) + (p + 2) * ldy],
                                                                  Y[(nc + j) + (p + 1) * ldy],
                                                                  Y[(nc + j) + p * ldy]);
                                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(wv, yv));
                                    }
                                    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
                                    float s_val = reduce_add_ps(sum);
                                    for (; p < NB; ++p)
                                        s_val += W[row * NB + p] * Y[(nc + j) + p * ldy];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

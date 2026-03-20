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

/// AVX512-optimized Householder reflection operations for the GELSD solver.
/// Matches the scalar LstSqGelsdHouseholder.hpp exactly, with AVX512 SIMD acceleration.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <immintrin.h>

#include "LstSqGelsdBlas_avx512.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AVX512-optimized Householder generate (double)
            //  Matches scalar householder_generate exactly, using AVX512 dot
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
            AVX512_TARGET_ATTR
            static inline double householder_generate_d_avx512(double *x, size_t n, double *alpha_out) {
                if (n == 0) return 0.0;
                if (n == 1) {
                    double alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0;
                    return 0.0;
                }
                double x0 = x[0];
                double sigma = dot_d_avx512(x + 1, x + 1, n - 1);

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
            //  AVX512-optimized Householder apply left (double, row-major)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void householder_apply_left_d_avx512(double tau, const double *v,
                                                               double *A, size_t m, size_t n,
                                                               size_t lda) {
                if (tau == 0.0) return;
                for (size_t j = 0; j < n; ++j) {
                    double s_val = 0.0;
                    size_t i = 0;
                    for (; i + 7 < m; i += 8) {
                        __m512d vi = _mm512_loadu_pd(v + i);
                        __m512d ai = _mm512_set_pd(A[(i + 7) * lda + j],
                                                   A[(i + 6) * lda + j],
                                                   A[(i + 5) * lda + j],
                                                   A[(i + 4) * lda + j],
                                                   A[(i + 3) * lda + j],
                                                   A[(i + 2) * lda + j],
                                                   A[(i + 1) * lda + j],
                                                   A[i * lda + j]);
                        __m512d prod = _mm512_mul_pd(vi, ai);
                        s_val += _mm512_reduce_add_pd(prod);
                    }
                    for (; i < m; ++i) s_val += v[i] * A[i * lda + j];
                    s_val *= tau;
                    i = 0;
                    __m512d sv = _mm512_set1_pd(s_val);
                    for (; i + 7 < m; i += 8) {
                        __m512d vi = _mm512_loadu_pd(v + i);
                        __m512d ai = _mm512_set_pd(A[(i + 7) * lda + j],
                                                   A[(i + 6) * lda + j],
                                                   A[(i + 5) * lda + j],
                                                   A[(i + 4) * lda + j],
                                                   A[(i + 3) * lda + j],
                                                   A[(i + 2) * lda + j],
                                                   A[(i + 1) * lda + j],
                                                   A[i * lda + j]);
                        __m512d result = _mm512_sub_pd(ai, _mm512_mul_pd(sv, vi));
                        A[i * lda + j] = result[0];
                        A[(i + 1) * lda + j] = result[1];
                        A[(i + 2) * lda + j] = result[2];
                        A[(i + 3) * lda + j] = result[3];
                        A[(i + 4) * lda + j] = result[4];
                        A[(i + 5) * lda + j] = result[5];
                        A[(i + 6) * lda + j] = result[6];
                        A[(i + 7) * lda + j] = result[7];
                    }
                    for (; i < m; ++i) A[i * lda + j] -= s_val * v[i];
                }
            }

            // ============================================================
            //  AVX512-optimized Householder apply right (double, row-major)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void householder_apply_right_d_avx512(double tau, const double *v,
                                                                double *A, size_t m, size_t n,
                                                                size_t lda) {
                if (tau == 0.0) return;
                for (size_t i = 0; i < m; ++i) {
                    __m512d s = _mm512_setzero_pd();
                    size_t j = 0;
                    for (; j + 7 < n; j += 8) {
                        __m512d aj = _mm512_loadu_pd(&A[i * lda + j]);
                        __m512d vj = _mm512_loadu_pd(v + j);
                        s = _mm512_add_pd(s, _mm512_mul_pd(aj, vj));
                    }
                    double s_val = _mm512_reduce_add_pd(s);
                    for (; j < n; ++j) s_val += A[i * lda + j] * v[j];
                    s_val *= tau;
                    j = 0;
                    __m512d sv = _mm512_set1_pd(s_val);
                    for (; j + 7 < n; j += 8) {
                        __m512d aj = _mm512_loadu_pd(&A[i * lda + j]);
                        __m512d vj = _mm512_loadu_pd(v + j);
                        _mm512_storeu_pd(&A[i * lda + j], _mm512_sub_pd(aj, _mm512_mul_pd(sv, vj)));
                    }
                    for (; j < n; ++j) A[i * lda + j] -= s_val * v[j];
                }
            }

            // ============================================================
            //  AVX512-optimized Householder generate (float)
            //  Matches scalar householder_generate exactly, using AVX512 dot
            // ============================================================

            AVX512_TARGET_ATTR
            static inline float householder_generate_f_avx512(float *x, size_t n, float *alpha_out) {
                if (n == 0) return 0.0f;
                if (n == 1) {
                    float alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0f;
                    return 0.0f;
                }
                float x0 = x[0];
                float sigma = dot_f_avx512(x + 1, x + 1, n - 1);

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
            //  AVX512-optimized Householder apply left (float, row-major)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void householder_apply_left_f_avx512(float tau, const float *v,
                                                               float *A, size_t m, size_t n,
                                                               size_t lda) {
                if (tau == 0.0f) return;
                for (size_t j = 0; j < n; ++j) {
                    float s_val = 0.0f;
                    size_t i = 0;
                    for (; i + 15 < m; i += 16) {
                        __m512 vi = _mm512_loadu_ps(v + i);
                        __m512 ai = _mm512_set_ps(A[(i + 15) * lda + j],
                                                  A[(i + 14) * lda + j],
                                                  A[(i + 13) * lda + j],
                                                  A[(i + 12) * lda + j],
                                                  A[(i + 11) * lda + j],
                                                  A[(i + 10) * lda + j],
                                                  A[(i + 9) * lda + j],
                                                  A[(i + 8) * lda + j],
                                                  A[(i + 7) * lda + j],
                                                  A[(i + 6) * lda + j],
                                                  A[(i + 5) * lda + j],
                                                  A[(i + 4) * lda + j],
                                                  A[(i + 3) * lda + j],
                                                  A[(i + 2) * lda + j],
                                                  A[(i + 1) * lda + j],
                                                  A[i * lda + j]);
                        __m512 prod = _mm512_mul_ps(vi, ai);
                        s_val += _mm512_reduce_add_ps(prod);
                    }
                    for (; i < m; ++i) s_val += v[i] * A[i * lda + j];
                    s_val *= tau;
                    i = 0;
                    __m512 sv = _mm512_set1_ps(s_val);
                    for (; i + 15 < m; i += 16) {
                        __m512 vi = _mm512_loadu_ps(v + i);
                        __m512 ai = _mm512_set_ps(A[(i + 15) * lda + j],
                                                  A[(i + 14) * lda + j],
                                                  A[(i + 13) * lda + j],
                                                  A[(i + 12) * lda + j],
                                                  A[(i + 11) * lda + j],
                                                  A[(i + 10) * lda + j],
                                                  A[(i + 9) * lda + j],
                                                  A[(i + 8) * lda + j],
                                                  A[(i + 7) * lda + j],
                                                  A[(i + 6) * lda + j],
                                                  A[(i + 5) * lda + j],
                                                  A[(i + 4) * lda + j],
                                                  A[(i + 3) * lda + j],
                                                  A[(i + 2) * lda + j],
                                                  A[(i + 1) * lda + j],
                                                  A[i * lda + j]);
                        __m512 result = _mm512_sub_ps(ai, _mm512_mul_ps(sv, vi));
                        for (size_t k = 0; k < 16; ++k)
                            A[(i + k) * lda + j] = result[k];
                    }
                    for (; i < m; ++i) A[i * lda + j] -= s_val * v[i];
                }
            }

            // ============================================================
            //  AVX512-optimized Householder apply right (float, row-major)
            // ============================================================

            AVX512_TARGET_ATTR
            static inline void householder_apply_right_f_avx512(float tau, const float *v,
                                                                float *A, size_t m, size_t n,
                                                                size_t lda) {
                if (tau == 0.0f) return;
                for (size_t i = 0; i < m; ++i) {
                    __m512 s = _mm512_setzero_ps();
                    size_t j = 0;
                    for (; j + 15 < n; j += 16) {
                        __m512 aj = _mm512_loadu_ps(&A[i * lda + j]);
                        __m512 vj = _mm512_loadu_ps(v + j);
                        s = _mm512_add_ps(s, _mm512_mul_ps(aj, vj));
                    }
                    float s_val = _mm512_reduce_add_ps(s);
                    for (; j < n; ++j) s_val += A[i * lda + j] * v[j];
                    s_val *= tau;
                    j = 0;
                    __m512 sv = _mm512_set1_ps(s_val);
                    for (; j + 15 < n; j += 16) {
                        __m512 aj = _mm512_loadu_ps(&A[i * lda + j]);
                        __m512 vj = _mm512_loadu_ps(v + j);
                        _mm512_storeu_ps(&A[i * lda + j], _mm512_sub_ps(aj, _mm512_mul_ps(sv, vj)));
                    }
                    for (; j < n; ++j) A[i * lda + j] -= s_val * v[j];
                }
            }

            // ============================================================
            //  Block size for compact WY transforms (matches scalar)
            // ============================================================

            // Use preprocessor macro to avoid redefinition when both scalar and AVX512 headers are included
#ifndef NP_HOUSEHOLDER_BLOCK_NB_DEFINED
#define NP_HOUSEHOLDER_BLOCK_NB_DEFINED
            static constexpr size_t HOUSEHOLDER_BLOCK_NB = 32;
#endif

            // ---------------------------------------------------------------
            //  Form the T matrix (DLARFT equivalent, double, AVX512)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors (double, AVX512).
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            AVX512_TARGET_ATTR
            static inline void larft_d_avx512(const double *Y, size_t m, size_t NB,
                                              const double *tau, size_t ldy,
                                              double *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    // Step 1: w[i] = Y[:,i]^T * Y[:,j] for i = 0..j-1
                    for (size_t i = 0; i < j; ++i) {
                        double w_i = dot_d_avx512(Y + i * ldy, Y + j * ldy, m);
                        T_[i + j * ldT] = w_i;
                    }

                    // Step 2: w = T[0:j,0:j] * w (matrix-vector multiply, DTRMV equivalent)
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
            //  LAPACK-style blocked application (DLARFB equivalent, double, AVX512)
            // ---------------------------------------------------------------

            AVX512_TARGET_ATTR
            static inline void larfb_left_d_avx512(const double *Y, size_t m, size_t NB,
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
                                __m512d sum = _mm512_setzero_pd();
                                size_t k = 0;
                                for (; k + 7 < kr; k += 8) {
                                    __m512d yv = _mm512_loadu_pd(Y_col + k);
                                    __m512d cv = _mm512_set_pd(C[(kc + k + 7) * ldc + j],
                                                               C[(kc + k + 6) * ldc + j],
                                                               C[(kc + k + 5) * ldc + j],
                                                               C[(kc + k + 4) * ldc + j],
                                                               C[(kc + k + 3) * ldc + j],
                                                               C[(kc + k + 2) * ldc + j],
                                                               C[(kc + k + 1) * ldc + j],
                                                               C[(kc + k + 0) * ldc + j]);
                                    sum = _mm512_add_pd(sum, _mm512_mul_pd(yv, cv));
                                }
                                double s_val = _mm512_reduce_add_pd(sum);
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
                                    double s_val = 0.0;
                                    size_t p = 0;
                                    for (; p + 7 < NB; p += 8) {
                                        __m512d yv = _mm512_set_pd(Y[row + (p + 7) * ldy],
                                                                   Y[row + (p + 6) * ldy],
                                                                   Y[row + (p + 5) * ldy],
                                                                   Y[row + (p + 4) * ldy],
                                                                   Y[row + (p + 3) * ldy],
                                                                   Y[row + (p + 2) * ldy],
                                                                   Y[row + (p + 1) * ldy],
                                                                   Y[row + p * ldy]);
                                        __m512d wv = _mm512_set_pd(W[(p + 7) * n + (nc + j)],
                                                                   W[(p + 6) * n + (nc + j)],
                                                                   W[(p + 5) * n + (nc + j)],
                                                                   W[(p + 4) * n + (nc + j)],
                                                                   W[(p + 3) * n + (nc + j)],
                                                                   W[(p + 2) * n + (nc + j)],
                                                                   W[(p + 1) * n + (nc + j)],
                                                                   W[p * n + (nc + j)]);
                                        __m512d prod = _mm512_mul_pd(yv, wv);
                                        s_val += _mm512_reduce_add_pd(prod);
                                    }
                                    for (; p < NB; ++p)
                                        s_val += Y[row + p * ldy] * W[p * n + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

            AVX512_TARGET_ATTR
            static inline void larfb_right_d_avx512(const double *Y, size_t n, size_t NB,
                                                    const double *T_, size_t ldT,
                                                    double *C, size_t m, size_t ldc,
                                                    size_t ldy) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                std::vector<double> W(m * NB, 0.0);

                // Step 1: W = C * Y (m x NB)
                // C is row-major (contiguous within a row), Y is column-major (contiguous down columns).
                // Both are contiguous, so dot_d_avx512 is correct here.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                double sum = dot_d_avx512(C + p * ldc + kc, Y + kc + i * ldy, kr);
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
                                    double s_val = 0.0;
                                    size_t p = 0;
                                    for (; p + 7 < NB; p += 8) {
                                        __m512d wv = _mm512_loadu_pd(&W[row * NB + p]);
                                        __m512d yv = _mm512_set_pd(Y[(nc + j) + (p + 7) * ldy],
                                                                   Y[(nc + j) + (p + 6) * ldy],
                                                                   Y[(nc + j) + (p + 5) * ldy],
                                                                   Y[(nc + j) + (p + 4) * ldy],
                                                                   Y[(nc + j) + (p + 3) * ldy],
                                                                   Y[(nc + j) + (p + 2) * ldy],
                                                                   Y[(nc + j) + (p + 1) * ldy],
                                                                   Y[(nc + j) + p * ldy]);
                                        __m512d prod = _mm512_mul_pd(wv, yv);
                                        s_val += _mm512_reduce_add_pd(prod);
                                    }
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
            //  Form the T matrix (float, AVX512)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors (float, AVX512).
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            AVX512_TARGET_ATTR
            static inline void larft_f_avx512(const float *Y, size_t m, size_t NB,
                                              const float *tau, size_t ldy,
                                              float *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    for (size_t i = 0; i < j; ++i) {
                        float w_i = dot_f_avx512(Y + i * ldy, Y + j * ldy, m);
                        T_[i + j * ldT] = w_i;
                    }

                    // Step 2: w = T[0:j,0:j] * w (matrix-vector multiply, DTRMV equivalent)
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
            //  LAPACK-style blocked application (float, AVX512)
            // ---------------------------------------------------------------

            AVX512_TARGET_ATTR
            static inline void larfb_left_f_avx512(const float *Y, size_t m, size_t NB,
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
                                __m512 sum = _mm512_setzero_ps();
                                size_t k = 0;
                                for (; k + 15 < kr; k += 16) {
                                    __m512 yv = _mm512_loadu_ps(Y_col + k);
                                    __m512 cv = _mm512_set_ps(C[(kc + k + 15) * ldc + j],
                                                              C[(kc + k + 14) * ldc + j],
                                                              C[(kc + k + 13) * ldc + j],
                                                              C[(kc + k + 12) * ldc + j],
                                                              C[(kc + k + 11) * ldc + j],
                                                              C[(kc + k + 10) * ldc + j],
                                                              C[(kc + k + 9) * ldc + j],
                                                              C[(kc + k + 8) * ldc + j],
                                                              C[(kc + k + 7) * ldc + j],
                                                              C[(kc + k + 6) * ldc + j],
                                                              C[(kc + k + 5) * ldc + j],
                                                              C[(kc + k + 4) * ldc + j],
                                                              C[(kc + k + 3) * ldc + j],
                                                              C[(kc + k + 2) * ldc + j],
                                                              C[(kc + k + 1) * ldc + j],
                                                              C[(kc + k + 0) * ldc + j]);
                                    sum = _mm512_add_ps(sum, _mm512_mul_ps(yv, cv));
                                }
                                float s_val = _mm512_reduce_add_ps(sum);
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
                                    float s_val = 0.0f;
                                    size_t p = 0;
                                    for (; p + 15 < NB; p += 16) {
                                        __m512 yv = _mm512_set_ps(Y[row + (p + 15) * ldy],
                                                                  Y[row + (p + 14) * ldy],
                                                                  Y[row + (p + 13) * ldy],
                                                                  Y[row + (p + 12) * ldy],
                                                                  Y[row + (p + 11) * ldy],
                                                                  Y[row + (p + 10) * ldy],
                                                                  Y[row + (p + 9) * ldy],
                                                                  Y[row + (p + 8) * ldy],
                                                                  Y[row + (p + 7) * ldy],
                                                                  Y[row + (p + 6) * ldy],
                                                                  Y[row + (p + 5) * ldy],
                                                                  Y[row + (p + 4) * ldy],
                                                                  Y[row + (p + 3) * ldy],
                                                                  Y[row + (p + 2) * ldy],
                                                                  Y[row + (p + 1) * ldy],
                                                                  Y[row + p * ldy]);
                                        __m512 wv = _mm512_set_ps(W[(p + 15) * n + (nc + j)],
                                                                  W[(p + 14) * n + (nc + j)],
                                                                  W[(p + 13) * n + (nc + j)],
                                                                  W[(p + 12) * n + (nc + j)],
                                                                  W[(p + 11) * n + (nc + j)],
                                                                  W[(p + 10) * n + (nc + j)],
                                                                  W[(p + 9) * n + (nc + j)],
                                                                  W[(p + 8) * n + (nc + j)],
                                                                  W[(p + 7) * n + (nc + j)],
                                                                  W[(p + 6) * n + (nc + j)],
                                                                  W[(p + 5) * n + (nc + j)],
                                                                  W[(p + 4) * n + (nc + j)],
                                                                  W[(p + 3) * n + (nc + j)],
                                                                  W[(p + 2) * n + (nc + j)],
                                                                  W[(p + 1) * n + (nc + j)],
                                                                  W[p * n + (nc + j)]);
                                        __m512 prod = _mm512_mul_ps(yv, wv);
                                        s_val += _mm512_reduce_add_ps(prod);
                                    }
                                    for (; p < NB; ++p)
                                        s_val += Y[row + p * ldy] * W[p * n + (nc + j)];
                                    C[row * ldc + (nc + j)] -= s_val;
                                }
                            }
                        }
                    }
                }
            }

            AVX512_TARGET_ATTR
            static inline void larfb_right_f_avx512(const float *Y, size_t n, size_t NB,
                                                    const float *T_, size_t ldT,
                                                    float *C, size_t m, size_t ldc,
                                                    size_t ldy) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                std::vector<float> W(m * NB, 0.0f);

                // Step 1: W = C * Y (m x NB)
                // C is row-major (contiguous within a row), Y is column-major (contiguous down columns).
                // Both are contiguous, so dot_f_avx512 is correct here.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                float sum = dot_f_avx512(C + p * ldc + kc, Y + kc + i * ldy, kr);
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
                                    float s_val = 0.0f;
                                    size_t p = 0;
                                    for (; p + 15 < NB; p += 16) {
                                        __m512 wv = _mm512_loadu_ps(&W[row * NB + p]);
                                        __m512 yv = _mm512_set_ps(Y[(nc + j) + (p + 15) * ldy],
                                                                  Y[(nc + j) + (p + 14) * ldy],
                                                                  Y[(nc + j) + (p + 13) * ldy],
                                                                  Y[(nc + j) + (p + 12) * ldy],
                                                                  Y[(nc + j) + (p + 11) * ldy],
                                                                  Y[(nc + j) + (p + 10) * ldy],
                                                                  Y[(nc + j) + (p + 9) * ldy],
                                                                  Y[(nc + j) + (p + 8) * ldy],
                                                                  Y[(nc + j) + (p + 7) * ldy],
                                                                  Y[(nc + j) + (p + 6) * ldy],
                                                                  Y[(nc + j) + (p + 5) * ldy],
                                                                  Y[(nc + j) + (p + 4) * ldy],
                                                                  Y[(nc + j) + (p + 3) * ldy],
                                                                  Y[(nc + j) + (p + 2) * ldy],
                                                                  Y[(nc + j) + (p + 1) * ldy],
                                                                  Y[(nc + j) + p * ldy]);
                                        __m512 prod = _mm512_mul_ps(wv, yv);
                                        s_val += _mm512_reduce_add_ps(prod);
                                    }
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

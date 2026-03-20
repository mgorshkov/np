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

/// AMX + AVX512 Householder operations for the GELSD least-squares solver.
/// Matches the scalar LstSqGelsdHouseholder.hpp exactly, with AMX tile loads
/// + AVX512 SIMD acceleration.
/// All matrices are stored in row-major layout.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <immintrin.h>

#include <np/internal/cpu/LstSqGelsdBlas_amx.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AMX + AVX512: Householder generate (double)
            //  Matches scalar householder_generate exactly, with alpha_out
            // ============================================================

            AMX_TARGET_ATTR
            static inline double householder_generate_d_amx(double *x, size_t n, double *alpha_out) {
                if (n == 0) return 0.0;
                if (n == 1) {
                    double alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0;
                    return 0.0;
                }
                double x0 = x[0];
                double sigma = dot_d_amx(x + 1, x + 1, n - 1);

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
            //  AMX + AVX512: Householder apply left (double, row-major)
            //  Matches scalar householder_apply_left exactly
            // ============================================================

            /// Apply Householder reflector from the left: A = (I - tau*v*v^T) * A
            /// A is m x n stored row-major. v has length m.
            /// For each column j: s = sum_i v[i] * A[i*n + j], then A[i*n + j] -= tau * s * v[i]
            ///
            /// Optimized: two-phase approach eliminates double row traversal per column group.
            /// Phase 1: compute s[j..j+7] = tau * sum_i v[i] * A[i*n + j] for ALL columns at once.
            /// Phase 2: apply A[i*n + j] -= s_work[j] * v[i] in a single row traversal.
            /// This halves memory traffic compared to the naive per-column-group double traversal.
            AMX_TARGET_ATTR
            static inline void householder_apply_left_d_amx(double tau, const double *v,
                                                            double *A, size_t m, size_t n,
                                                            size_t lda) {
                if (tau == 0.0) return;
                __m512d vtau = _mm512_set1_pd(tau);

                // Phase 1: compute s[j..j+7] = tau * sum_i v[i] * A[i*n + j] for ALL columns.
                // Store s values so Phase 2 can apply in a single row traversal.
                std::vector<double> s_work(n, 0.0);
                size_t j = 0;
                for (; j + 7 < n; j += 8) {
                    __m512d s = _mm512_setzero_pd();

                    // Process rows using AMX tiles for wide loads of v and A
                    size_t i = 0;
                    for (; i + kAmxDoublesPerTile - 1 < m; i += kAmxDoublesPerTile) {
                        // Load v tile
                        _tile_loadd(0, const_cast<double *>(v + i), kAmxTileColBytes);

                        for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                            std::size_t offset = row * (kAmxTileColBytes / 8);
                            __m512d vi = _mm512_loadu_pd(v + i + offset);

                            // Load 8 columns of A at this row (row-major: A[(i+offset)*lda + (j+0..7)])
                            __m512d a0 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 0)]);
                            __m512d a1 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 1)]);
                            __m512d a2 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 2)]);
                            __m512d a3 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 3)]);
                            __m512d a4 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 4)]);
                            __m512d a5 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 5)]);
                            __m512d a6 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 6)]);
                            __m512d a7 = _mm512_loadu_pd(&A[(i + offset) * lda + (j + 7)]);

                            // s += v[i] * A[i][j] for each column j using FMA
                            s = _mm512_fmadd_pd(vi, a0, s);
                            s = _mm512_fmadd_pd(vi, a1, s);
                            s = _mm512_fmadd_pd(vi, a2, s);
                            s = _mm512_fmadd_pd(vi, a3, s);
                            s = _mm512_fmadd_pd(vi, a4, s);
                            s = _mm512_fmadd_pd(vi, a5, s);
                            s = _mm512_fmadd_pd(vi, a6, s);
                            s = _mm512_fmadd_pd(vi, a7, s);
                        }
                    }

                    // Process remaining rows (non-tiled)
                    for (; i < m; ++i) {
                        __m512d vi = _mm512_set1_pd(v[i]);
                        __m512d a_cols = _mm512_loadu_pd(&A[i * lda + j]);
                        s = _mm512_fmadd_pd(vi, a_cols, s);
                    }

                    s = _mm512_mul_pd(s, vtau);
                    _mm512_storeu_pd(&s_work[j], s);
                }

                // Process remaining columns (non-tiled)
                for (; j < n; ++j) {
                    double s = 0.0;
                    size_t i = 0;
                    for (; i + kAmxDoublesPerTile - 1 < m; i += kAmxDoublesPerTile) {
                        _tile_loadd(0, const_cast<double *>(v + i), kAmxTileColBytes);
                        for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                            std::size_t offset = row * (kAmxTileColBytes / 8);
                            __m512d vi = _mm512_loadu_pd(v + i + offset);
                            __m512d ai = _mm512_loadu_pd(&A[(i + offset) * lda + j]);
                            s += _mm512_reduce_add_pd(_mm512_mul_pd(vi, ai));
                        }
                    }
                    for (; i < m; ++i) s += v[i] * A[i * lda + j];
                    s_work[j] = s * tau;
                }

                // Phase 2: apply A[i*lda + j] -= s_work[j] * v[i] in a single row traversal.
                for (size_t i = 0; i < m; ++i) {
                    __m512d vi = _mm512_set1_pd(v[i]);
                    j = 0;
                    for (; j + 7 < n; j += 8) {
                        __m512d a0 = _mm512_loadu_pd(&A[i * lda + (j + 0)]);
                        __m512d a1 = _mm512_loadu_pd(&A[i * lda + (j + 1)]);
                        __m512d a2 = _mm512_loadu_pd(&A[i * lda + (j + 2)]);
                        __m512d a3 = _mm512_loadu_pd(&A[i * lda + (j + 3)]);
                        __m512d a4 = _mm512_loadu_pd(&A[i * lda + (j + 4)]);
                        __m512d a5 = _mm512_loadu_pd(&A[i * lda + (j + 5)]);
                        __m512d a6 = _mm512_loadu_pd(&A[i * lda + (j + 6)]);
                        __m512d a7 = _mm512_loadu_pd(&A[i * lda + (j + 7)]);

                        __m512d sj = _mm512_loadu_pd(&s_work[j]);

                        _mm512_storeu_pd(&A[i * lda + (j + 0)], _mm512_fnmadd_pd(sj, vi, a0));
                        _mm512_storeu_pd(&A[i * lda + (j + 1)], _mm512_fnmadd_pd(sj, vi, a1));
                        _mm512_storeu_pd(&A[i * lda + (j + 2)], _mm512_fnmadd_pd(sj, vi, a2));
                        _mm512_storeu_pd(&A[i * lda + (j + 3)], _mm512_fnmadd_pd(sj, vi, a3));
                        _mm512_storeu_pd(&A[i * lda + (j + 4)], _mm512_fnmadd_pd(sj, vi, a4));
                        _mm512_storeu_pd(&A[i * lda + (j + 5)], _mm512_fnmadd_pd(sj, vi, a5));
                        _mm512_storeu_pd(&A[i * lda + (j + 6)], _mm512_fnmadd_pd(sj, vi, a6));
                        _mm512_storeu_pd(&A[i * lda + (j + 7)], _mm512_fnmadd_pd(sj, vi, a7));
                    }
                    for (; j < n; ++j) {
                        A[i * lda + j] -= s_work[j] * v[i];
                    }
                }
            }

            // ============================================================
            //  AMX + AVX512: Householder apply right (double, row-major)
            //  Matches scalar householder_apply_right exactly
            // ============================================================

            /// Apply Householder reflector from the right: A = A * (I - tau*v*v^T)
            /// A is m x n stored row-major. v has length n.
            /// For each row i: s = sum_j A[i*n + j] * v[j], then A[i*n + j] -= tau * s * v[j]
            AMX_TARGET_ATTR
            static inline void householder_apply_right_d_amx(double tau, const double *v,
                                                             double *A, size_t m, size_t n,
                                                             size_t lda) {
                if (tau == 0.0) return;
                __m512d vtau = _mm512_set1_pd(tau);

                // Process rows in groups of 8 (AVX512 width)
                size_t i = 0;
                for (; i + 7 < m; i += 8) {
                    __m512d s = _mm512_setzero_pd();

                    // Compute s = sum over columns of A[i][col] * v[col] for each of 8 rows
                    // Use FMA: s += aj * vj
                    for (size_t j = 0; j < n; ++j) {
                        __m512d vj = _mm512_set1_pd(v[j]);
                        __m512d aj = _mm512_loadu_pd(&A[i * lda + j]);
                        s = _mm512_fmadd_pd(aj, vj, s);
                    }
                    s = _mm512_mul_pd(s, vtau);

                    // Apply: A[i..i+7][j] -= s[i..i+7] * v[j]  =>  aj - s*vj
                    // Use FNMADD: aj = aj - s * vj  =>  fnmadd(s, vj, aj)
                    for (size_t j = 0; j < n; ++j) {
                        __m512d vj = _mm512_set1_pd(v[j]);
                        __m512d aj = _mm512_loadu_pd(&A[i * lda + j]);
                        _mm512_storeu_pd(&A[i * lda + j], _mm512_fnmadd_pd(s, vj, aj));
                    }
                }

                // Process remaining rows
                for (; i < m; ++i) {
                    double s = 0.0;
                    for (size_t j = 0; j < n; ++j) s += A[i * lda + j] * v[j];
                    s *= tau;
                    for (size_t j = 0; j < n; ++j) A[i * lda + j] -= s * v[j];
                }
            }

            // ============================================================
            //  Block size for compact WY transforms (matches scalar)
            // ============================================================

#ifndef NP_HOUSEHOLDER_BLOCK_NB_DEFINED
#define NP_HOUSEHOLDER_BLOCK_NB_DEFINED
            static constexpr size_t HOUSEHOLDER_BLOCK_NB = 32;
#endif

            // ---------------------------------------------------------------
            //  Form the T matrix (DLARFT equivalent, double, AMX)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors (double, AMX).
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            AMX_TARGET_ATTR
            static inline void larft_d_amx(const double *Y, size_t m, size_t NB,
                                           const double *tau, size_t ldy,
                                           double *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    // Step 1: w[i] = Y[:,i]^T * Y[:,j] for i = 0..j-1
                    for (size_t i = 0; i < j; ++i) {
                        double w_i = dot_d_amx(Y + i * ldy, Y + j * ldy, m);
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
            //  LAPACK-style blocked application (DLARFB equivalent, double, AMX)
            // ---------------------------------------------------------------

            AMX_TARGET_ATTR
            static inline void larfb_left_d_amx(const double *Y, size_t m, size_t NB,
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

            AMX_TARGET_ATTR
            static inline void larfb_right_d_amx(const double *Y, size_t n, size_t NB,
                                                 const double *T_, size_t ldT,
                                                 double *C, size_t m, size_t ldc,
                                                 size_t ldy) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                std::vector<double> W(m * NB, 0.0);

                // Step 1: W = C * Y (m x NB)
                // C is row-major (contiguous within a row), Y is column-major (contiguous down columns).
                // Both are contiguous, so dot_d_amx is correct here.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                double sum = dot_d_amx(C + p * ldc + kc, Y + kc + i * ldy, kr);
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

            // ============================================================
            //  AMX + AVX512: Householder generate (float)
            //  Matches scalar householder_generate exactly, with alpha_out
            // ============================================================

            AMX_TARGET_ATTR
            static inline float householder_generate_f_amx(float *x, size_t n, float *alpha_out) {
                if (n == 0) return 0.0f;
                if (n == 1) {
                    float alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = 1.0f;
                    return 0.0f;
                }
                float x0 = x[0];
                float sigma = dot_f_amx(x + 1, x + 1, n - 1);

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
            //  AMX + AVX512: Householder apply left (float, row-major)
            //  Matches scalar householder_apply_left exactly
            // ============================================================

            /// Apply Householder reflector from the left: A = (I - tau*v*v^T) * A
            /// A is m x n stored row-major. v has length m.
            /// Two-phase approach: Phase 1 computes s for ALL columns, Phase 2 applies.
            AMX_TARGET_ATTR
            static inline void householder_apply_left_f_amx(float tau, const float *v,
                                                            float *A, size_t m, size_t n,
                                                            size_t lda) {
                if (tau == 0.0f) return;
                __m512 vtau = _mm512_set1_ps(tau);

                // Phase 1: compute s[j..j+15] = tau * sum_i v[i] * A[i*n + j] for ALL columns.
                std::vector<float> s_work(n, 0.0f);
                size_t j = 0;
                for (; j + 15 < n; j += 16) {
                    __m512 s = _mm512_setzero_ps();

                    // Process rows using AMX tiles for wide loads of v and A
                    size_t i = 0;
                    for (; i + kAmxFloatsPerTile - 1 < m; i += kAmxFloatsPerTile) {
                        _tile_loadd(0, const_cast<float *>(v + i), kAmxTileColBytes);

                        for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                            std::size_t offset = row * (kAmxTileColBytes / 4);
                            __m512 vi = _mm512_loadu_ps(v + i + offset);

                            // Load 16 columns of A at this row
                            __m512 a0 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 0)]);
                            __m512 a1 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 1)]);
                            __m512 a2 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 2)]);
                            __m512 a3 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 3)]);
                            __m512 a4 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 4)]);
                            __m512 a5 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 5)]);
                            __m512 a6 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 6)]);
                            __m512 a7 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 7)]);
                            __m512 a8 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 8)]);
                            __m512 a9 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 9)]);
                            __m512 a10 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 10)]);
                            __m512 a11 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 11)]);
                            __m512 a12 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 12)]);
                            __m512 a13 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 13)]);
                            __m512 a14 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 14)]);
                            __m512 a15 = _mm512_loadu_ps(&A[(i + offset) * lda + (j + 15)]);

                            s = _mm512_fmadd_ps(vi, a0, s);
                            s = _mm512_fmadd_ps(vi, a1, s);
                            s = _mm512_fmadd_ps(vi, a2, s);
                            s = _mm512_fmadd_ps(vi, a3, s);
                            s = _mm512_fmadd_ps(vi, a4, s);
                            s = _mm512_fmadd_ps(vi, a5, s);
                            s = _mm512_fmadd_ps(vi, a6, s);
                            s = _mm512_fmadd_ps(vi, a7, s);
                            s = _mm512_fmadd_ps(vi, a8, s);
                            s = _mm512_fmadd_ps(vi, a9, s);
                            s = _mm512_fmadd_ps(vi, a10, s);
                            s = _mm512_fmadd_ps(vi, a11, s);
                            s = _mm512_fmadd_ps(vi, a12, s);
                            s = _mm512_fmadd_ps(vi, a13, s);
                            s = _mm512_fmadd_ps(vi, a14, s);
                            s = _mm512_fmadd_ps(vi, a15, s);
                        }
                    }

                    // Process remaining rows (non-tiled)
                    for (; i < m; ++i) {
                        __m512 vi = _mm512_set1_ps(v[i]);
                        __m512 a_cols = _mm512_loadu_ps(&A[i * lda + j]);
                        s = _mm512_fmadd_ps(vi, a_cols, s);
                    }

                    s = _mm512_mul_ps(s, vtau);
                    _mm512_storeu_ps(&s_work[j], s);
                }

                // Process remaining columns (non-tiled)
                for (; j < n; ++j) {
                    float s = 0.0f;
                    size_t i = 0;
                    for (; i + kAmxFloatsPerTile - 1 < m; i += kAmxFloatsPerTile) {
                        _tile_loadd(0, const_cast<float *>(v + i), kAmxTileColBytes);
                        for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                            std::size_t offset = row * (kAmxTileColBytes / 4);
                            __m512 vi = _mm512_loadu_ps(v + i + offset);
                            __m512 ai = _mm512_loadu_ps(&A[(i + offset) * lda + j]);
                            s += _mm512_reduce_add_ps(_mm512_mul_ps(vi, ai));
                        }
                    }
                    for (; i < m; ++i) s += v[i] * A[i * lda + j];
                    s_work[j] = s * tau;
                }

                // Phase 2: apply A[i*lda + j] -= s_work[j] * v[i] in a single row traversal.
                for (size_t i = 0; i < m; ++i) {
                    __m512 vi = _mm512_set1_ps(v[i]);
                    j = 0;
                    for (; j + 15 < n; j += 16) {
                        __m512 a0 = _mm512_loadu_ps(&A[i * lda + (j + 0)]);
                        __m512 a1 = _mm512_loadu_ps(&A[i * lda + (j + 1)]);
                        __m512 a2 = _mm512_loadu_ps(&A[i * lda + (j + 2)]);
                        __m512 a3 = _mm512_loadu_ps(&A[i * lda + (j + 3)]);
                        __m512 a4 = _mm512_loadu_ps(&A[i * lda + (j + 4)]);
                        __m512 a5 = _mm512_loadu_ps(&A[i * lda + (j + 5)]);
                        __m512 a6 = _mm512_loadu_ps(&A[i * lda + (j + 6)]);
                        __m512 a7 = _mm512_loadu_ps(&A[i * lda + (j + 7)]);
                        __m512 a8 = _mm512_loadu_ps(&A[i * lda + (j + 8)]);
                        __m512 a9 = _mm512_loadu_ps(&A[i * lda + (j + 9)]);
                        __m512 a10 = _mm512_loadu_ps(&A[i * lda + (j + 10)]);
                        __m512 a11 = _mm512_loadu_ps(&A[i * lda + (j + 11)]);
                        __m512 a12 = _mm512_loadu_ps(&A[i * lda + (j + 12)]);
                        __m512 a13 = _mm512_loadu_ps(&A[i * lda + (j + 13)]);
                        __m512 a14 = _mm512_loadu_ps(&A[i * lda + (j + 14)]);
                        __m512 a15 = _mm512_loadu_ps(&A[i * lda + (j + 15)]);

                        __m512 sj = _mm512_loadu_ps(&s_work[j]);

                        _mm512_storeu_ps(&A[i * lda + (j + 0)], _mm512_fnmadd_ps(sj, vi, a0));
                        _mm512_storeu_ps(&A[i * lda + (j + 1)], _mm512_fnmadd_ps(sj, vi, a1));
                        _mm512_storeu_ps(&A[i * lda + (j + 2)], _mm512_fnmadd_ps(sj, vi, a2));
                        _mm512_storeu_ps(&A[i * lda + (j + 3)], _mm512_fnmadd_ps(sj, vi, a3));
                        _mm512_storeu_ps(&A[i * lda + (j + 4)], _mm512_fnmadd_ps(sj, vi, a4));
                        _mm512_storeu_ps(&A[i * lda + (j + 5)], _mm512_fnmadd_ps(sj, vi, a5));
                        _mm512_storeu_ps(&A[i * lda + (j + 6)], _mm512_fnmadd_ps(sj, vi, a6));
                        _mm512_storeu_ps(&A[i * lda + (j + 7)], _mm512_fnmadd_ps(sj, vi, a7));
                        _mm512_storeu_ps(&A[i * lda + (j + 8)], _mm512_fnmadd_ps(sj, vi, a8));
                        _mm512_storeu_ps(&A[i * lda + (j + 9)], _mm512_fnmadd_ps(sj, vi, a9));
                        _mm512_storeu_ps(&A[i * lda + (j + 10)], _mm512_fnmadd_ps(sj, vi, a10));
                        _mm512_storeu_ps(&A[i * lda + (j + 11)], _mm512_fnmadd_ps(sj, vi, a11));
                        _mm512_storeu_ps(&A[i * lda + (j + 12)], _mm512_fnmadd_ps(sj, vi, a12));
                        _mm512_storeu_ps(&A[i * lda + (j + 13)], _mm512_fnmadd_ps(sj, vi, a13));
                        _mm512_storeu_ps(&A[i * lda + (j + 14)], _mm512_fnmadd_ps(sj, vi, a14));
                        _mm512_storeu_ps(&A[i * lda + (j + 15)], _mm512_fnmadd_ps(sj, vi, a15));
                    }
                    for (; j < n; ++j) {
                        A[i * lda + j] -= s_work[j] * v[i];
                    }
                }
            }

            // ============================================================
            //  AMX + AVX512: Householder apply right (float, row-major)
            //  Matches scalar householder_apply_right exactly
            // ============================================================

            /// Apply Householder reflector from the right: A = A * (I - tau*v*v^T)
            /// A is m x n stored row-major. v has length n.
            AMX_TARGET_ATTR
            static inline void householder_apply_right_f_amx(float tau, const float *v,
                                                             float *A, size_t m, size_t n,
                                                             size_t lda) {
                if (tau == 0.0f) return;
                __m512 vtau = _mm512_set1_ps(tau);

                // Process rows in groups of 16 (AVX512 width for float)
                size_t i = 0;
                for (; i + 15 < m; i += 16) {
                    __m512 s = _mm512_setzero_ps();

                    // Compute s = sum over columns of A[i][col] * v[col] for each of 16 rows
                    for (size_t j = 0; j < n; ++j) {
                        __m512 vj = _mm512_set1_ps(v[j]);
                        __m512 aj = _mm512_loadu_ps(&A[i * lda + j]);
                        s = _mm512_fmadd_ps(aj, vj, s);
                    }
                    s = _mm512_mul_ps(s, vtau);

                    // Apply: A[i..i+15][j] -= s[i..i+15] * v[j]
                    for (size_t j = 0; j < n; ++j) {
                        __m512 vj = _mm512_set1_ps(v[j]);
                        __m512 aj = _mm512_loadu_ps(&A[i * lda + j]);
                        _mm512_storeu_ps(&A[i * lda + j], _mm512_fnmadd_ps(s, vj, aj));
                    }
                }

                // Process remaining rows
                for (; i < m; ++i) {
                    float s = 0.0f;
                    for (size_t j = 0; j < n; ++j) s += A[i * lda + j] * v[j];
                    s *= tau;
                    for (size_t j = 0; j < n; ++j) A[i * lda + j] -= s * v[j];
                }
            }

            // ---------------------------------------------------------------
            //  Form the T matrix (DLARFT equivalent, float, AMX)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors (float, AMX).
            ///
            /// Uses the standard LAPACK DLARFT formula with triangular solve:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            AMX_TARGET_ATTR
            static inline void larft_f_amx(const float *Y, size_t m, size_t NB,
                                           const float *tau, size_t ldy,
                                           float *T_, size_t ldT) {
                if (NB == 0) return;

                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    // Step 1: w[i] = Y[:,i]^T * Y[:,j] for i = 0..j-1
                    for (size_t i = 0; i < j; ++i) {
                        float w_i = dot_f_amx(Y + i * ldy, Y + j * ldy, m);
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

                    // Step 3: T[0:j, j] = -tau[j] * x
                    float tau_j = tau[j];
                    for (size_t i = 0; i < j; ++i)
                        T_[i + j * ldT] *= -tau_j;

                    // Step 4: T[j, j] = tau[j]
                    T_[j + j * ldT] = tau_j;
                }
            }

            // ---------------------------------------------------------------
            //  LAPACK-style blocked application (DLARFB equivalent, float, AMX)
            // ---------------------------------------------------------------

            AMX_TARGET_ATTR
            static inline void larfb_left_f_amx(const float *Y, size_t m, size_t NB,
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

                // Step 2: W = T * W (NB x n)
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

            AMX_TARGET_ATTR
            static inline void larfb_right_f_amx(const float *Y, size_t n, size_t NB,
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
                                float sum = dot_f_amx(C + p * ldc + kc, Y + kc + i * ldy, kr);
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

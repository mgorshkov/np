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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace np {
    namespace internal {
        namespace cpu {

            // ---------------------------------------------------------------
            //  Householder reflector generation (DLARFG equivalent)
            // ---------------------------------------------------------------

            /// Generate a Householder reflector H = I - tau * v * v^T
            /// such that H * x = [alpha, 0, ..., 0]^T.
            ///
            /// On input, x[0..n-1] is the vector to be reflected.
            /// On output, x[0] = 1 (implicit), x[1..n-1] contains the reflector.
            /// alpha_out receives the diagonal element (the norm with sign).
            /// Returns tau = (alpha - x[0]) / alpha.
            template<typename T>
            /// Generate a Householder reflector H = I - tau * v * v^T
            /// such that H * x = [alpha, 0, ..., 0]^T.
            ///
            /// This is the LAPACK DLARFG equivalent.
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
            static inline T householder_generate(T *x, size_t n, T *alpha_out = nullptr) {
                if (n == 0) return T(0);
                if (n == 1) {
                    T alpha = x[0];
                    if (alpha_out) *alpha_out = alpha;
                    x[0] = T(1);
                    return T(0);
                }

                T x0 = x[0];
                T sigma = T(0);
                for (size_t i = 1; i < n; ++i)
                    sigma += x[i] * x[i];

                // alpha = -sign(x0) * sqrt(x0^2 + sigma)  (LAPACK's beta)
                // LAPACK DLARFG: beta = -sign(x0_in) * sqrt(x0_in^2 + sigma)
                // If x0 >= 0: alpha = -sqrt(...) (negative)
                // If x0 < 0:  alpha = +sqrt(...) (positive)
                T alpha = std::sqrt(x0 * x0 + sigma);
                if (x0 >= T(0)) alpha = -alpha;
                if (alpha_out) *alpha_out = alpha;

                // beta = alpha - x0  (LAPACK's beta - x0_in)
                T beta = alpha - x0;

                // Check for zero norm (x is already on the axis)
                if (beta == T(0) && sigma == T(0)) {
                    x[0] = T(1);
                    return T(0);
                }

                // Store reflector: v[0] = 1 (implicit)
                // v[1..n-1] = x[1..n-1] / (x0 - alpha) = -x[1..n-1] / beta
                // This matches LAPACK DLARFG: x[i] = x[i] / (x0_in - beta_lapack)
                T inv_scale = T(1) / (x0 - alpha);// = -1/beta
                x[0] = T(1);
                for (size_t i = 1; i < n; ++i)
                    x[i] *= inv_scale;

                return beta / alpha;
            }

            // ---------------------------------------------------------------
            //  Apply a single Householder reflector (DLARF equivalent)
            // ---------------------------------------------------------------

            /// Apply a Householder reflector from the LEFT:
            ///   C = (I - tau * v * v^T) * C
            ///
            /// v[0..m-1] is the reflector (v[0] = 1 implicit).
            /// C is m x n stored row-major with leading dimension ldc.
            template<typename T>
            static inline void householder_apply_left(T tau, const T *v,
                                                      T *C, size_t m, size_t n, size_t ldc) {
                if (tau == T(0) || m == 0 || n == 0) return;

                // w[j] = sum_{i=0..m-1} v[i] * C[i, j]
                for (size_t j = 0; j < n; ++j) {
                    T w = T(0);
                    for (size_t i = 0; i < m; ++i)
                        w += v[i] * C[i * ldc + j];
                    w *= tau;
                    for (size_t i = 0; i < m; ++i)
                        C[i * ldc + j] -= w * v[i];
                }
            }

            /// Apply a Householder reflector from the RIGHT:
            ///   C = C * (I - tau * v * v^T)
            ///
            /// v[0..n-1] is the reflector (v[0] = 1 implicit).
            /// C is m x n stored row-major with leading dimension ldc.
            template<typename T>
            static inline void householder_apply_right(T tau, const T *v,
                                                       T *C, size_t m, size_t n, size_t ldc) {
                if (tau == T(0) || m == 0 || n == 0) return;

                // w[i] = sum_{j=0..n-1} C[i, j] * v[j]
                for (size_t i = 0; i < m; ++i) {
                    T w = T(0);
                    for (size_t j = 0; j < n; ++j)
                        w += C[i * ldc + j] * v[j];
                    w *= tau;
                    for (size_t j = 0; j < n; ++j)
                        C[i * ldc + j] -= w * v[j];
                }
            }

            // ---------------------------------------------------------------
            //  Tiled GEMM helpers for compact WY
            // ---------------------------------------------------------------

            /// C += A * B   (all row-major, A is m x k, B is k x n, C is m x n)
            template<typename T>
            static inline void gemm_nn_add(T *C, size_t ldc,
                                           const T *A, size_t lda,
                                           const T *B, size_t ldb,
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < kr; ++p)
                                        sum += A[row * lda + (kc + p)] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] += sum;
                                }
                            }
                        }
                    }
                }
            }

            /// C -= A * B   (all row-major, A is m x k, B is k x n, C is m x n)
            template<typename T>
            static inline void gemm_nn_sub(T *C, size_t ldc,
                                           const T *A, size_t lda,
                                           const T *B, size_t ldb,
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < kr; ++p)
                                        sum += A[row * lda + (kc + p)] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] -= sum;
                                }
                            }
                        }
                    }
                }
            }

            /// C += A * B^T   (C and A row-major, B^T is B transposed)
            /// A is m x k, B is n x k (so B^T is k x n), C is m x n
            template<typename T>
            static inline void gemm_nt_add(T *C, size_t ldc,
                                           const T *A, size_t lda,
                                           const T *B, size_t ldb,
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < kr; ++p)
                                        sum += A[row * lda + (kc + p)] * B[(nc + j) * ldb + (kc + p)];
                                    C[row * ldc + (nc + j)] += sum;
                                }
                            }
                        }
                    }
                }
            }

            /// C -= A * B^T   (C and A row-major, B^T is B transposed)
            template<typename T>
            static inline void gemm_nt_sub(T *C, size_t ldc,
                                           const T *A, size_t lda,
                                           const T *B, size_t ldb,
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < kr; ++p)
                                        sum += A[row * lda + (kc + p)] * B[(nc + j) * ldb + (kc + p)];
                                    C[row * ldc + (nc + j)] -= sum;
                                }
                            }
                        }
                    }
                }
            }

            /// C += A^T * B   (C and B row-major, A^T is A transposed)
            /// A is k x m (so A^T is m x k), B is k x n, C is m x n
            template<typename T>
            static inline void gemm_tn_add(T *C, size_t ldc,
                                           const T *A, size_t lda,
                                           const T *B, size_t ldb,
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < kr; ++p)
                                        sum += A[(kc + p) * lda + row] * B[(kc + p) * ldb + (nc + j)];
                                    C[row * ldc + (nc + j)] += sum;
                                }
                            }
                        }
                    }
                }
            }

            /// Block size for compact WY transforms
            // Use preprocessor macro to avoid redefinition when both scalar and SIMD headers are included
#ifndef NP_HOUSEHOLDER_BLOCK_NB_DEFINED
#define NP_HOUSEHOLDER_BLOCK_NB_DEFINED
            static constexpr size_t HOUSEHOLDER_BLOCK_NB = 32;
#endif

            // ---------------------------------------------------------------
            //  Form the T matrix (DLARFT equivalent)
            // ---------------------------------------------------------------

            /// Form the T matrix for a block of NB Householder reflectors.
            ///
            /// Y is m x NB stored column-major in an NB*ldy buffer.
            ///   - Y[0..m-1, j] is the j-th reflector (v[0]=1 implicit).
            ///   - ldy >= m is the leading dimension of Y (in elements).
            /// tau[0..NB-1] are the scalar factors.
            /// T is NB x NB stored column-major with ldT >= NB.
            ///   On output, T is upper-triangular (strict lower part not referenced).
            ///
            /// This is the DLARFT equivalent.
            ///
            /// Uses the standard LAPACK DLARFT formula:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     Solve T[0:j,0:j] * x = w  (triangular solve, forward substitution)
            ///     T[0:j,j] = -tau[j] * x
            ///     T[j,j] = tau[j]
            ///
            /// This formula works for ANY set of reflector vectors, regardless
            /// of where the implicit 1s are positioned. The triangular solve
            /// (T^{-1} * w) is the correct formula for the compact WY representation.
            /// The alternative formula (T * w) is only valid when Y has the
            /// specific unit lower trapezoidal structure Y[i,i] = 1.
            /// Form the T matrix for a block of NB Householder reflectors.
            ///
            /// Y is m x NB stored column-major in an NB*ldy buffer.
            ///   - Y[0..m-1, j] is the j-th reflector (v[0]=1 implicit).
            ///   - ldy >= m is the leading dimension of Y (in elements).
            /// tau[0..NB-1] are the scalar factors.
            /// T is NB x NB stored column-major with ldT >= NB.
            ///   On output, T is upper-triangular (strict lower part not referenced).
            ///
            /// This is the DLARFT equivalent.
            ///
            /// Uses the standard LAPACK DLARFT formula:
            ///   T[0,0] = tau[0]
            ///   For j = 1..NB-1:
            ///     w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
            ///     T[0:j,j] = -tau[j] * T[0:j,0:j] * w  (DTRMV with 'No transpose')
            ///     T[j,j] = tau[j]
            ///
            /// This is the standard LAPACK DLARFT formula using T * w (DTRMV).
            /// The alternative formula using T^{-1} * w (triangular solve) is
            /// only needed when Y has a non-standard structure (e.g., anti-diagonal
            /// implicit 1s in multiply_right_pt).
            template<typename T>
            static inline void larft(const T *Y, size_t m, size_t NB,
                                     const T *tau, size_t ldy,
                                     T *T_, size_t ldT) {
                if (NB == 0) return;

                // T[0,0] = tau[0]
                T_[0] = tau[0];

                for (size_t j = 1; j < NB; ++j) {
                    // Step 1: w[i] = Y[:,i]^T * Y[:,j]  for i = 0..j-1
                    // Y is column-major: Y[k + i*ldy] is element (k,i)
                    // Store w_i temporarily in T[j, i] (we'll overwrite later)
                    for (size_t i = 0; i < j; ++i) {
                        T w_i = T(0);
                        for (size_t k = 0; k < m; ++k)
                            w_i += Y[k + i * ldy] * Y[k + j * ldy];
                        T_[i + j * ldT] = w_i;
                    }

                    // Step 2: w = T[0:j,0:j] * w  (matrix-vector multiply, DTRMV equivalent)
                    // T is upper-triangular column-major: T[i,k] = T_[i + k*ldT]
                    // T[i,k] = 0 for k < i (upper triangular).
                    // w_new[i] = sum_{k=i..j-1} T[i,k] * w[k]
                    for (size_t i = 0; i < j; ++i) {
                        T sum = T(0);
                        for (size_t k = i; k < j; ++k)
                            sum += T_[i + k * ldT] * T_[k + j * ldT];
                        T_[i + j * ldT] = sum;
                    }

                    // Step 3: T[0:j, j] = -tau[j] * x
                    T tau_j = tau[j];
                    for (size_t i = 0; i < j; ++i)
                        T_[i + j * ldT] *= -tau_j;

                    // Step 4: T[j, j] = tau[j]
                    T_[j + j * ldT] = tau_j;
                }
            }

            // ---------------------------------------------------------------
            //  LAPACK-style blocked application (DLARFB equivalent)
            // ---------------------------------------------------------------
            //
            // LAPACK's DORMBR/DORMQR uses DGEMM for the Y^T*C and Y*W products,
            // and DTRMM for the T*W product. We implement the same approach
            // with explicit cache tiling in the GEMM operations.
            //
            // The key difference from the naive triple-loop implementation:
            //   - Cache-friendly tiling in the m and n dimensions
            //   - Better memory access patterns (contiguous inner loops)
            //   - Reduced TLB misses for large matrices

            /// Apply a block of NB Householder reflectors from the LEFT:
            ///   C = (I - Y * T * Y^T) * C
            ///
            /// Y is m x NB stored column-major with leading dimension ldy (>= m).
            /// T is NB x NB stored column-major with leading dimension ldT (>= NB).
            /// C is m x n stored row-major with leading dimension ldc (original columns).
            ///
            /// This is the DLARFB equivalent for left application.
            /// If W_buf is non-null, it is used as the NB x n work buffer (must be
            /// pre-allocated with at least NB * n elements). Otherwise, a local
            /// std::vector is allocated.
            template<typename T>
            static inline void larfb_left(const T *Y, size_t m, size_t NB,
                                          const T *T_, size_t ldT,
                                          T *C, size_t n, size_t ldc,
                                          size_t ldy = 0,
                                          T *W_buf = nullptr) {
                if (ldy == 0) ldy = m;
                if (NB == 0 || n == 0) return;

                // LAPACK approach:
                //   W = Y^T * C   (NB x n)  — GEMM (Y^T is NB x m, C is m x n)
                //   W = T * W     (NB x n)  — TRMM (T is NB x NB upper-tri)
                //   C = C - Y * W (m x n)   — GEMM (Y is m x NB, W is NB x n)

                // Use pre-allocated buffer if provided, otherwise allocate locally
                std::vector<T> W_local;
                T *W = W_buf;
                if (W == nullptr) {
                    W_local.resize(NB * n, T(0));
                    W = W_local.data();
                } else {
                    // Zero the work buffer
                    std::memset(W, 0, NB * n * sizeof(T));
                }

                // Step 1: W = Y^T * C   (NB x n)
                // Tile over m (the reduction dimension) for cache efficiency.
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < m; kc += KC) {
                        size_t kr = std::min(KC, m - kc);
                        for (size_t i = 0; i < NB; ++i) {
                            for (size_t j = 0; j < n; ++j) {
                                T sum = T(0);
                                for (size_t k = 0; k < kr; ++k)
                                    sum += Y[(kc + k) + i * ldy] * C[(kc + k) * ldc + j];
                                W[i * n + j] += sum;
                            }
                        }
                    }
                }

                // Step 2: W = T * W   (NB x n), T is NB x NB upper-triangular column-major
                for (size_t j = 0; j < n; ++j) {
                    for (size_t i = 0; i < NB; ++i) {
                        T sum = T(0);
                        for (size_t k = i; k < NB; ++k)
                            sum += T_[i + k * ldT] * W[k * n + j];
                        W[i * n + j] = sum;
                    }
                }

                // Step 3: C = C - Y * W   (m x n)
                // Tile over m and n.
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < NB; ++p)
                                        sum += Y[row + p * ldy] * W[p * n + (nc + j)];
                                    C[row * ldc + (nc + j)] -= sum;
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
            /// C is m x n stored row-major with leading dimension ldc (original columns).
            ///
            /// This is the DLARFB equivalent for right application.
            /// If W_buf is non-null, it is used as the m x NB work buffer (must be
            /// pre-allocated with at least m * NB elements). Otherwise, a local
            /// std::vector is allocated.
            template<typename T>
            static inline void larfb_right(const T *Y, size_t n, size_t NB,
                                           const T *T_, size_t ldT,
                                           T *C, size_t m, size_t ldc,
                                           size_t ldy = 0,
                                           T *W_buf = nullptr) {
                if (ldy == 0) ldy = n;
                if (NB == 0 || m == 0) return;

                // LAPACK approach:
                //   W = C * Y     (m x NB)  — GEMM (C is m x n, Y is n x NB)
                //   W = W * T     (m x NB)  — TRMM (T is NB x NB upper-tri)
                //   C = C - W * Y^T (m x n) — GEMM (W is m x NB, Y^T is NB x n)

                // Use pre-allocated buffer if provided, otherwise allocate locally
                std::vector<T> W_local;
                T *W = W_buf;
                if (W == nullptr) {
                    W_local.resize(m * NB, T(0));
                    W = W_local.data();
                } else {
                    std::memset(W, 0, m * NB * sizeof(T));
                }

                // Step 1: W = C * Y   (m x NB)
                // Tile over n (the reduction dimension).
                {
                    constexpr size_t KC = 256;
                    for (size_t kc = 0; kc < n; kc += KC) {
                        size_t kr = std::min(KC, n - kc);
                        for (size_t p = 0; p < m; ++p) {
                            for (size_t i = 0; i < NB; ++i) {
                                T sum = T(0);
                                for (size_t j = 0; j < kr; ++j)
                                    sum += C[p * ldc + (kc + j)] * Y[(kc + j) + i * ldy];
                                W[p * NB + i] += sum;
                            }
                        }
                    }
                }

                // Step 2: W = W * T   (m x NB), T is NB x NB upper-triangular column-major
                for (size_t p = 0; p < m; ++p) {
                    for (size_t i = NB; i > 0;) {
                        --i;
                        T sum = T(0);
                        for (size_t k = 0; k <= i; ++k)
                            sum += W[p * NB + k] * T_[k + i * ldT];
                        W[p * NB + i] = sum;
                    }
                }

                // Step 3: C = C - W * Y^T   (m x n)
                // Tile over m and n.
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
                                    T sum = T(0);
                                    for (size_t p = 0; p < NB; ++p)
                                        sum += W[row * NB + p] * Y[(nc + j) + p * ldy];
                                    C[row * ldc + (nc + j)] -= sum;
                                }
                            }
                        }
                    }
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

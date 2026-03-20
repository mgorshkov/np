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
#include <cstddef>
#include <cstring>
#include <vector>

#include "LstSqGelsdHouseholder.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Back-transform Householder reflectors (row-major)
            //  with compact WY blocking
            // ============================================================
            //
            // The bidiagonal reduction gives A = Q * B * P^T where
            //   Q = H_0 * H_1 * ... * H_{k-1}  (forward order)
            //   P = G_0 * G_1 * ... * G_{k-1}  (forward order)
            //
            // To compute U = Q * U_bidiag, we apply H_{k-1} first, then H_{k-2}, etc.
            // So the iteration must be REVERSE (i = k-1 to 0).
            //
            // Instead of applying each reflector individually (O(k*m*nru)),
            // we accumulate NB reflectors into a compact WY block and apply
            // via GEMM (O(m*nru*NB) per block, with k/NB blocks).
            //
            // The compact WY representation: Q_block = I - Y * T * Y^T
            // where Y is m x NB with unit lower trapezoidal structure:
            //   Y[orig_idx + j*m] = 1  (implicit 1 at the reflector's diagonal)
            //   Y[r + j*m] = reflector element for r > orig_idx
            //   Y[r + j*m] = 0 for r < orig_idx
            //
            // ============================================================

            /// Apply the left reflectors stored in A (from gebrd) to U.
            /// A is m x n row-major. Reflectors are stored in columns of A.
            /// U is m x nru stored row-major.
            template<typename T>
            static void multiply_left_q(const T *A, size_t m, size_t n,
                                        const T *tauq, size_t k,
                                        T *U, size_t nru) {
                if (k == 0 || nru == 0) return;
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Pre-allocate persistent buffers (reused across blocks)
                std::vector<T> Y(m * NB, T(0));
                std::vector<T> T_buf(NB * NB, T(0));
                std::vector<T> tau_block(NB);
                // Pre-allocate W buffer for larfb_left (NB x nru)
                std::vector<T> W_buf(NB * nru, T(0));

                // Process reflectors from bottom to top in blocks of NB
                size_t i = k;
                while (i > 0) {
                    size_t block_end = i;
                    size_t block_start = (i > NB) ? (i - NB) : 0;
                    size_t nb = block_end - block_start;

                    // Extract the nb reflectors into Y in FORWARD order.
                    // Left reflectors are stored in COLUMNS of A (strided by n).
                    // We copy element-by-element since they are not contiguous.
                    size_t nb_active = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        size_t orig = block_start + j;
                        T tau = tauq[orig];
                        if (tau == T(0)) {
                            continue;
                        }
                        tau_block[nb_active] = tau;
                        size_t v_len = m - orig;
                        // Implicit 1 at position orig
                        Y[orig + nb_active * m] = T(1);
                        // Elements below the implicit 1 — strided by n in A
                        // Use pointer arithmetic for slightly better codegen
                        const T *A_col = A + orig * n + orig;
                        T *Y_col = Y.data() + (orig + 1) + nb_active * m;
                        for (size_t r = 1; r < v_len; ++r)
                            Y_col[r - 1] = A_col[r * n];
                        ++nb_active;
                    }
                    nb = nb_active;

                    if (nb > 0) {
                        size_t y_len = m - block_start;

                        larft(Y.data() + block_start, y_len, nb,
                              tau_block.data(), m,
                              T_buf.data(), NB);

                        // Apply Q_block = I - Y*T*Y^T to U[block_start..m-1, 0..nru-1]
                        // Use pre-allocated W buffer to avoid re-allocation
                        larfb_left(Y.data() + block_start, y_len, nb,
                                   T_buf.data(), NB,
                                   &U[block_start * nru], nru, nru, m,
                                   W_buf.data());
                    }

                    i = block_start;
                }
            }

            /// Apply the right reflectors stored in A (from gebrd) to VT.
            /// A is m x n row-major. Reflectors are stored in rows of A.
            /// VT is ncv x n stored row-major (each row is a right singular vector).
            ///
            /// P^T = G_{k-1} * ... * G_0 (reverse order).
            /// To compute VT = VT_bidiag * P^T, we apply G_{k-1} first, then G_{k-2}, etc.
            /// So the iteration must be REVERSE (i = k-1 to 0).
            ///
            /// Uses unblocked application: each reflector is applied individually.
            /// This avoids the complexity of compact WY blocking with reverse-ordered
            /// reflectors, which would require a non-standard Y structure for larft.
            /// The unblocked approach is acceptable because VT is typically small
            /// (ncv <= n, and often ncv == n).
            template<typename T>
            static void multiply_right_pt(const T *A, size_t m, size_t n,
                                          const T *taup, size_t k,
                                          T *VT, size_t ncv) {
                if (k == 0 || ncv == 0) return;
                (void) m;

                // Apply reflectors from bottom to top: G_{k-1}, G_{k-2}, ..., G_0
                // Each right reflector v is stored in row orig of A:
                //   v[0] = 1 (implicit) at position (orig+1)
                //   v[1..v_len-1] = A[orig * n + (orig+2) .. orig * n + (n-1)]
                // The reflector vector v has length v_len = n - orig - 1.
                //
                // VT = VT * (I - tau * v * v^T)
                for (size_t i = k; i > 0;) {
                    --i;
                    T tau = taup[i];
                    if (tau == T(0)) continue;
                    size_t v_len = n - i - 1;
                    if (v_len == 0) continue;

                    // Build the reflector vector v with implicit 1 at position 0
                    // v[0] = 1 (implicit)
                    // v[1..v_len-1] = A[i * n + (i+2) .. i * n + (n-1)]
                    // We store v in a temporary buffer and pass it to householder_apply_right.
                    // The implicit 1 is at position 0 of v, and the matrix C starts at
                    // column (i+1) of VT.
                    //
                    // householder_apply_right(tau, v, C, m, n, ldc) computes:
                    //   C = C * (I - tau * v * v^T)
                    // where C is m x n, v is length n.
                    //
                    // We apply to VT[:, i+1:] (ncv rows, v_len columns).
                    // The reflector v has length v_len, with v[0] = 1 (implicit).

                    // Build v in a local buffer
                    T v_buf[256];// Stack buffer for small sizes
                    std::vector<T> v_heap;
                    T *v = v_buf;
                    if (v_len > 256) {
                        v_heap.resize(v_len);
                        v = v_heap.data();
                    }
                    v[0] = T(1);// Implicit 1
                    if (v_len > 1) {
                        std::memcpy(v + 1, A + i * n + (i + 2),
                                    (v_len - 1) * sizeof(T));
                    }

                    // Apply to VT[:, i+1:]
                    householder_apply_right(tau, v,
                                            &VT[0 * n + (i + 1)],
                                            ncv, v_len, n);
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

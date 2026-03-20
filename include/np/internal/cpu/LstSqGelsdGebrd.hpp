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
#include <vector>

#include "LstSqGelsdHouseholder.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Bidiagonal reduction (GEBRD) - row-major
            //
            //  This implementation uses a HYBRID approach:
            //
            //  Phase 1 (inner loop, j = 0..nb-1):
            //    Generate left reflector v_col from column col
            //    Apply v_col to ALL columns col+1..n-1 (not just block columns)
            //    Store v_col in Y_left for back-transform use
            //    Generate right reflector u_col from row col (now fully updated)
            //    Apply u_col to ALL rows below col+1..m-1
            //
            //  Phase 2 is SKIPPED because the left reflectors have already been
            //  applied to all columns in Phase 1.
            //
            //  IMPORTANT: Unlike LAPACK's blocked DGEBRD which uses DLABRD
            //  (panel reduction) + compact WY (BLAS-3 trailing update), this
            //  implementation applies left reflectors individually to ALL columns.
            //  This is necessary because in row-major layout, the right reflectors
            //  must be generated from fully-updated rows. LAPACK's DLABRD achieves
            //  this with incremental DGEMV updates within the panel reduction.
            //
            //  The performance impact of applying left reflectors to all columns
            //  (instead of just block columns) is acceptable because:
            //  - NB = 32 is small
            //  - The right reflector application is already O(NB * m * n)
            //  - The total complexity is O(NB * m * n) for both left and right
            // ============================================================

            /// Reduce A (m x n, row-major) to bidiagonal form.
            ///
            /// Hybrid blocked/unblocked algorithm:
            /// - Left reflectors are applied individually to ALL columns
            ///   (unblocked, to ensure right reflectors are generated correctly)
            /// - Right reflectors are applied individually to ALL rows below
            ///   (unblocked, cache-friendly in row-major)
            ///
            /// NB = HOUSEHOLDER_BLOCK_NB (32) is used only for the Y_left buffer
            /// size (needed by back-transform routines), not for blocking.
            ///
            /// On exit:
            ///   d[0..k-1]     = diagonal elements of the bidiagonal matrix
            ///   e[1..k-1]     = superdiagonal elements (e[0] is unused)
            ///   tauq[0..k-1]  = scalar factors for the left reflectors (Q)
            ///   taup[0..k-1]  = scalar factors for the right reflectors (P^T)
            ///   A             = stores the Householder reflectors in factored form
            template<typename T>
            void gebrd(T *A, size_t m, size_t n,
                       T *d, T *e, T *tauq, T *taup) {
                size_t k = std::min(m, n);
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Work buffer for reflector generation (max dimension)
                std::vector<T> work(std::max(m, n));

                // Y_left buffer for back-transform routines (stores left reflectors)
                // Y_left: m x NB column-major (ldy = m)
                std::vector<T> Y_left(m * NB, T(0));

                for (size_t i = 0; i < k; i += NB) {
                    size_t nb = std::min(NB, k - i);

                    // ====================================================
                    // Phase 1: Generate nb left and right reflectors.
                    //
                    // Left reflectors are applied to ALL columns to the right
                    // (col+1..n-1), ensuring the right reflector is generated
                    // from the fully updated row.
                    //
                    // Right reflectors are applied to ALL rows below
                    // (col+1..m-1), cache-friendly in row-major.
                    // ====================================================
                    for (size_t j = 0; j < nb; ++j) {
                        size_t col = i + j;
                        size_t m_i = m - col;
                        size_t n_i = n - col;

                        // --- Left reflector (column col) ---
                        // Extract column col below diagonal (strided by n in row-major)
                        for (size_t r = 0; r < m_i; ++r)
                            work[r] = A[(col + r) * n + col];

                        T tauq_i = householder_generate(work.data(), m_i, &d[col]);
                        tauq[col] = tauq_i;

                        // Store the reflector into Y_left (column-major)
                        // for use by back-transform routines
                        for (size_t r = 0; r < m_i; ++r)
                            Y_left[(col + r) + j * m] = work[r];

                        // Apply left reflector to ALL columns to the right: col+1 .. n-1
                        // This ensures the right reflector is generated from the
                        // fully updated row.
                        size_t n_right = n - col - 1;
                        if (n_right > 0) {
                            householder_apply_left(tauq_i, work.data(),
                                                   &A[col * n + (col + 1)], m_i, n_right, n);
                        }

                        // Store reflector back into column col of A
                        for (size_t r = 0; r < m_i; ++r)
                            A[(col + r) * n + col] = work[r];
                        A[col * n + col] = d[col];

                        if (col + 1 >= k) {
                            taup[col] = T(0);
                            for (size_t jj = j + 1; jj < nb; ++jj)
                                if (i + jj < k)
                                    taup[i + jj] = T(0);
                            nb = j + 1;
                            break;
                        }

                        // --- Right reflector (row col) ---
                        // Extract row col right of superdiagonal (contiguous in row-major)
                        size_t n_i2 = n_i - 1;
                        for (size_t c = 0; c < n_i2; ++c)
                            work[c] = A[col * n + (col + 1 + c)];

                        T taup_i = householder_generate(work.data(), n_i2, &e[col + 1]);
                        taup[col] = taup_i;

                        // Apply right reflector to ALL rows below: col+1 .. m-1
                        // This is cache-friendly in row-major (contiguous row access).
                        if (m_i > 1) {
                            householder_apply_right(taup_i, work.data(),
                                                    &A[(col + 1) * n + (col + 1)], m_i - 1, n_i2, n);
                        }

                        // Store reflector back into row col of A
                        for (size_t c = 0; c < n_i2; ++c)
                            A[col * n + (col + 1 + c)] = work[c];
                        A[col * n + (col + 1)] = e[col + 1];
                    }

                    if (nb == 0) break;

                    // ====================================================
                    // Phase 2 is SKIPPED.
                    //
                    // The left reflectors have already been applied to ALL
                    // columns in Phase 1, so there is no need for a compact
                    // WY trailing update.
                    //
                    // The Y_left buffer is not used by the solver (reflectors
                    // are read from A), so we don't need to clear it.
                    // ====================================================
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

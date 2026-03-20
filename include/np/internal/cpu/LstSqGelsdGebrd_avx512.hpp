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

/// AVX512-optimized GEBRD (bidiagonal reduction) for the GELSD solver.
/// Matches the blocked scalar LstSqGelsdGebrd.hpp exactly, with AVX512 SIMD
/// acceleration in the Phase 1 inner loop.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "LstSqGelsdHouseholder_avx512.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AVX512-optimized GEBRD (bidiagonal reduction, double)
            //
            //  Hybrid blocked/unblocked algorithm matching the scalar version:
            //
            //  LEFT reflector application (column access, strided by n in
            //  row-major) uses compact WY blocking (BLAS-3) via larft/larfb_left.
            //
            //  RIGHT reflector application (row access, contiguous in row-major)
            //  is unblocked (cache-friendly).
            //
            //  Phase 1 uses AVX512-accelerated Householder functions.
            //  Phase 2 is SKIPPED — left reflectors are applied to ALL columns
            //  in Phase 1 to ensure right reflectors are generated correctly.
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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
            AVX512_TARGET_ATTR
            static void gebrd_d_avx512(double *A, size_t m, size_t n,
                                       double *d, double *e, double *tauq, double *taup) {
                size_t k = std::min(m, n);
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Work buffer for reflector generation (max dimension)
                std::vector<double> work(std::max(m, n));

                // Y_left buffer for back-transform routines (stores left reflectors)
                // Y_left: m x NB column-major (ldy = m)
                std::vector<double> Y_left(m * NB, 0.0);

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
                    //
                    // Uses AVX512-accelerated Householder functions.
                    // ====================================================
                    for (size_t j = 0; j < nb; ++j) {
                        size_t col = i + j;
                        size_t m_i = m - col;
                        size_t n_i = n - col;

                        // --- Left reflector (column col) ---
                        // Extract column col below diagonal (strided by n in row-major)
                        for (size_t r = 0; r < m_i; ++r)
                            work[r] = A[(col + r) * n + col];

                        double tauq_i = householder_generate_d_avx512(work.data(), m_i, &d[col]);
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
                            householder_apply_left_d_avx512(tauq_i, work.data(),
                                                            &A[col * n + (col + 1)], m_i, n_right, n);
                        }

                        // Store reflector back into column col of A
                        for (size_t r = 0; r < m_i; ++r)
                            A[(col + r) * n + col] = work[r];
                        A[col * n + col] = d[col];

                        if (col + 1 >= k) {
                            taup[col] = 0.0;
                            // Shrink the block — remaining reflectors don't exist
                            for (size_t jj = j + 1; jj < nb; ++jj)
                                taup[i + jj] = 0.0;
                            nb = j + 1;
                            break;
                        }

                        // --- Right reflector (row col) ---
                        // Extract row col right of superdiagonal (contiguous in row-major)
                        size_t n_i2 = n_i - 1;
                        for (size_t c = 0; c < n_i2; ++c)
                            work[c] = A[col * n + (col + 1 + c)];

                        double taup_i = householder_generate_d_avx512(work.data(), n_i2, &e[col + 1]);
                        taup[col] = taup_i;

                        // Apply right reflector to ALL rows below: col+1 .. m-1
                        // This is cache-friendly in row-major (contiguous row access).
                        if (m_i > 1) {
                            householder_apply_right_d_avx512(taup_i, work.data(),
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
#pragma GCC diagnostic pop

            // ============================================================
            //  AVX512-optimized GEBRD (bidiagonal reduction, float)
            //  Matches the scalar version with AVX512 SIMD acceleration
            // ============================================================

            /// Reduce A (m x n, row-major) to bidiagonal form (float).
            ///
            /// Hybrid blocked/unblocked algorithm:
            /// - Left reflectors are applied individually to ALL columns
            ///   (unblocked, to ensure right reflectors are generated correctly)
            /// - Right reflectors are applied individually to ALL rows below
            ///   (unblocked, cache-friendly in row-major)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
            AVX512_TARGET_ATTR
            static void gebrd_f_avx512(float *A, size_t m, size_t n,
                                       float *d, float *e, float *tauq, float *taup) {
                size_t k = std::min(m, n);
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Work buffer for reflector generation (max dimension)
                std::vector<float> work(std::max(m, n));

                // Y_left buffer for back-transform routines (stores left reflectors)
                // Y_left: m x NB column-major (ldy = m)
                std::vector<float> Y_left(m * NB, 0.0f);

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
                    //
                    // Uses AVX512-accelerated Householder functions.
                    // ====================================================
                    for (size_t j = 0; j < nb; ++j) {
                        size_t col = i + j;
                        size_t m_i = m - col;
                        size_t n_i = n - col;

                        // --- Left reflector (column col) ---
                        // Extract column col below diagonal (strided by n in row-major)
                        for (size_t r = 0; r < m_i; ++r)
                            work[r] = A[(col + r) * n + col];

                        float tauq_i = householder_generate_f_avx512(work.data(), m_i, &d[col]);
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
                            householder_apply_left_f_avx512(tauq_i, work.data(),
                                                            &A[col * n + (col + 1)], m_i, n_right, n);
                        }

                        // Store reflector back into column col of A
                        for (size_t r = 0; r < m_i; ++r)
                            A[(col + r) * n + col] = work[r];
                        A[col * n + col] = d[col];

                        if (col + 1 >= k) {
                            taup[col] = 0.0f;
                            // Shrink the block — remaining reflectors don't exist
                            for (size_t jj = j + 1; jj < nb; ++jj)
                                taup[i + jj] = 0.0f;
                            nb = j + 1;
                            break;
                        }

                        // --- Right reflector (row col) ---
                        // Extract row col right of superdiagonal (contiguous in row-major)
                        size_t n_i2 = n_i - 1;
                        for (size_t c = 0; c < n_i2; ++c)
                            work[c] = A[col * n + (col + 1 + c)];

                        float taup_i = householder_generate_f_avx512(work.data(), n_i2, &e[col + 1]);
                        taup[col] = taup_i;

                        // Apply right reflector to ALL rows below: col+1 .. m-1
                        // This is cache-friendly in row-major (contiguous row access).
                        if (m_i > 1) {
                            householder_apply_right_f_avx512(taup_i, work.data(),
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
#pragma GCC diagnostic pop
        }// namespace cpu
    }// namespace internal
}// namespace np

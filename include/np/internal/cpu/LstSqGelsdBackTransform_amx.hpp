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

/// AMX + AVX512 back-transform of Householder reflectors for the GELSD solver.
/// Matches the scalar LstSqGelsdBackTransform.hpp exactly, with AMX tile loads
/// + AVX512 SIMD acceleration.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include <np/internal/cpu/LstSqGelsdHouseholder_amx.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Back-transform left reflectors (double, AMX)
            //  Matches scalar multiply_left_q exactly, using AMX WY helpers
            // ============================================================

            /// Apply the left reflectors stored in A (from gebrd) to U.
            /// A is m x n row-major. Reflectors are stored in columns of A.
            /// U is m x nru stored row-major.
            AMX_TARGET_ATTR
            static void multiply_left_q_d_amx(const double *A, size_t m, size_t n,
                                              const double *tauq, size_t k,
                                              double *U, size_t nru) {
                if (k == 0 || nru == 0) return;
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Pre-allocate persistent buffers (reused across blocks)
                std::vector<double> Y(m * NB, 0.0);
                std::vector<double> T_buf(NB * NB, 0.0);
                std::vector<double> tau_block(NB);

                // Process reflectors from bottom to top in blocks of NB
                size_t i = k;
                while (i > 0) {
                    size_t block_end = i;
                    size_t block_start = (i > NB) ? (i - NB) : 0;
                    size_t nb = block_end - block_start;

                    // Extract the nb reflectors into Y in FORWARD order.
                    // Left reflectors are stored in COLUMNS of A (strided by n).
                    size_t nb_active = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        size_t orig = block_start + j;
                        double tau = tauq[orig];
                        if (tau == 0.0) {
                            continue;
                        }
                        tau_block[nb_active] = tau;
                        size_t v_len = m - orig;
                        // Implicit 1 at position orig
                        Y[orig + nb_active * m] = 1.0;
                        // Elements below the implicit 1 — strided by n in A
                        for (size_t r = 1; r < v_len; ++r)
                            Y[(orig + r) + nb_active * m] = A[(orig + r) * n + orig];
                        ++nb_active;
                    }
                    nb = nb_active;

                    if (nb > 0) {
                        size_t y_len = m - block_start;

                        larft_d_amx(Y.data() + block_start, y_len, nb,
                                    tau_block.data(), m,
                                    T_buf.data(), NB);

                        // Apply Q_block = I - Y*T*Y^T to U[block_start..m-1, 0..nru-1]
                        larfb_left_d_amx(Y.data() + block_start, y_len, nb,
                                         T_buf.data(), NB,
                                         &U[block_start * nru], nru, nru, m);
                    }

                    i = block_start;
                }
            }

            // ============================================================
            //  Back-transform right reflectors (double, AMX)
            //  Matches scalar multiply_right_pt exactly, using AMX WY helpers
            // ============================================================

            /// Apply the right reflectors stored in A (from gebrd) to VT.
            /// A is m x n row-major. Reflectors are stored in rows of A.
            /// VT is ncv x n stored row-major (each row is a right singular vector).
            AMX_TARGET_ATTR
            static void multiply_right_pt_d_amx(const double *A, size_t m, size_t n,
                                                const double *taup, size_t k,
                                                double *VT, size_t ncv) {
                if (k == 0 || ncv == 0) return;
                (void) m;
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                // Pre-allocate persistent buffers
                std::vector<double> Y(n * NB, 0.0);
                std::vector<double> T_buf(NB * NB, 0.0);
                std::vector<double> tau_block(NB);

                // Process reflectors from bottom to top in blocks of NB
                size_t i = k;
                while (i > 0) {
                    size_t block_end = i;
                    size_t block_start = (i > NB) ? (i - NB) : 0;
                    size_t nb = block_end - block_start;

                    // Extract the nb reflectors into Y in REVERSE order.
                    // Right reflectors are stored in ROWS of A (contiguous).
                    size_t nb_active = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        // Reverse order: j=0 gets the LAST reflector in the block
                        size_t orig = block_start + (nb - 1 - j);
                        double tau = taup[orig];
                        if (tau == 0.0) continue;
                        size_t v_len = n - orig - 1;
                        if (v_len == 0) continue;
                        tau_block[nb_active] = tau;
                        // Implicit 1 at position (orig+1)
                        Y[(orig + 1) + nb_active * n] = 1.0;
                        // Elements after the implicit 1 — contiguous in A row
                        for (size_t c = 1; c < v_len; ++c)
                            Y[(orig + 1 + c) + nb_active * n] = A[orig * n + (orig + 1 + c)];
                        ++nb_active;
                    }
                    nb = nb_active;

                    if (nb > 0) {
                        size_t y_offset = block_start + 1;
                        size_t y_len = n - y_offset;

                        larft_d_amx(Y.data() + y_offset, y_len, nb,
                                    tau_block.data(), n,
                                    T_buf.data(), NB);

                        // Apply to VT (ncv x n row-major).
                        // VT = VT * (I - Y*T*Y^T)
                        larfb_right_d_amx(Y.data() + y_offset, y_len, nb,
                                          T_buf.data(), NB,
                                          &VT[0 * n + y_offset],
                                          ncv, n, n);
                    }

                    i = block_start;
                }
            }

            // ============================================================
            //  Back-transform left reflectors (float, AMX)
            //  Matches scalar multiply_left_q exactly, using AMX WY helpers
            // ============================================================

            AMX_TARGET_ATTR
            static void multiply_left_q_f_amx(const float *A, size_t m, size_t n,
                                              const float *tauq, size_t k,
                                              float *U, size_t nru) {
                if (k == 0 || nru == 0) return;
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                std::vector<float> Y(m * NB, 0.0f);
                std::vector<float> T_buf(NB * NB, 0.0f);
                std::vector<float> tau_block(NB);

                size_t i = k;
                while (i > 0) {
                    size_t block_end = i;
                    size_t block_start = (i > NB) ? (i - NB) : 0;
                    size_t nb = block_end - block_start;

                    size_t nb_active = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        size_t orig = block_start + j;
                        float tau = tauq[orig];
                        if (tau == 0.0f) {
                            continue;
                        }
                        tau_block[nb_active] = tau;
                        size_t v_len = m - orig;
                        Y[orig + nb_active * m] = 1.0f;
                        for (size_t r = 1; r < v_len; ++r)
                            Y[(orig + r) + nb_active * m] = A[(orig + r) * n + orig];
                        ++nb_active;
                    }
                    nb = nb_active;

                    if (nb > 0) {
                        size_t y_len = m - block_start;

                        larft_f_amx(Y.data() + block_start, y_len, nb,
                                    tau_block.data(), m,
                                    T_buf.data(), NB);

                        larfb_left_f_amx(Y.data() + block_start, y_len, nb,
                                         T_buf.data(), NB,
                                         &U[block_start * nru], nru, nru, m);
                    }

                    i = block_start;
                }
            }

            // ============================================================
            //  Back-transform right reflectors (float, AMX)
            //  Matches scalar multiply_right_pt exactly, using AMX WY helpers
            // ============================================================

            AMX_TARGET_ATTR
            static void multiply_right_pt_f_amx(const float *A, size_t m, size_t n,
                                                const float *taup, size_t k,
                                                float *VT, size_t ncv) {
                if (k == 0 || ncv == 0) return;
                (void) m;
                constexpr size_t NB = HOUSEHOLDER_BLOCK_NB;

                std::vector<float> Y(n * NB, 0.0f);
                std::vector<float> T_buf(NB * NB, 0.0f);
                std::vector<float> tau_block(NB);

                size_t i = k;
                while (i > 0) {
                    size_t block_end = i;
                    size_t block_start = (i > NB) ? (i - NB) : 0;
                    size_t nb = block_end - block_start;

                    size_t nb_active = 0;
                    for (size_t j = 0; j < nb; ++j) {
                        size_t orig = block_start + (nb - 1 - j);
                        float tau = taup[orig];
                        if (tau == 0.0f) continue;
                        size_t v_len = n - orig - 1;
                        if (v_len == 0) continue;
                        tau_block[nb_active] = tau;
                        Y[(orig + 1) + nb_active * n] = 1.0f;
                        for (size_t c = 1; c < v_len; ++c)
                            Y[(orig + 1 + c) + nb_active * n] = A[orig * n + (orig + 1 + c)];
                        ++nb_active;
                    }
                    nb = nb_active;

                    if (nb > 0) {
                        size_t y_offset = block_start + 1;
                        size_t y_len = n - y_offset;

                        larft_f_amx(Y.data() + y_offset, y_len, nb,
                                    tau_block.data(), n,
                                    T_buf.data(), NB);

                        larfb_right_f_amx(Y.data() + y_offset, y_len, nb,
                                          T_buf.data(), NB,
                                          &VT[0 * n + y_offset],
                                          ncv, n, n);
                    }

                    i = block_start;
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

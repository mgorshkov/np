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

/// AMX + AVX512 GEBRD (bidiagonal reduction) for the GELSD solver.
/// Matches the scalar LstSqGelsdGebrd.hpp exactly, with AMX tile loads
/// + AVX512 SIMD acceleration.

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <np/internal/cpu/LstSqGelsdHouseholder_amx.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  AMX + AVX512: GEBRD (bidiagonal reduction, double)
            //  Matches scalar gebrd exactly, using AMX for dot/Householder
            // ============================================================

            AMX_TARGET_ATTR
            static void gebrd_d_amx(double *A, size_t m, size_t n,
                                    double *d, double *e, double *tauq, double *taup) {
                size_t k = std::min(m, n);
                std::vector<double> work(std::max(m, n));

                for (size_t i = 0; i < k; ++i) {
                    size_t m_i = m - i;
                    size_t n_i = n - i;

                    // Extract column i below diagonal (strided by n in row-major)
                    for (size_t r = 0; r < m_i; ++r)
                        work[r] = A[(i + r) * n + i];

                    double tauq_i = householder_generate_d_amx(work.data(), m_i, &d[i]);
                    tauq[i] = tauq_i;
                    work[0] = 1.0;

                    // Apply left reflector to submatrix rows i..m-1, cols i+1..n-1
                    if (n_i > 1) {
                        householder_apply_left_d_amx(tauq_i, work.data(),
                                                     &A[i * n + (i + 1)], m_i, n_i - 1, n);
                    }

                    // Store reflector back into column i
                    for (size_t r = 0; r < m_i; ++r)
                        A[(i + r) * n + i] = work[r];
                    A[i * n + i] = d[i];

                    if (i + 1 >= k) {
                        taup[i] = 0.0;
                        break;
                    }

                    size_t n_i2 = n_i - 1;
                    // Extract row i right of superdiagonal (contiguous in row-major)
                    for (size_t c = 0; c < n_i2; ++c)
                        work[c] = A[i * n + (i + 1 + c)];

                    double taup_i = householder_generate_d_amx(work.data(), n_i2, &e[i + 1]);
                    taup[i] = taup_i;

                    // Apply right reflector to submatrix rows i+1..m-1, cols i+1..n-1
                    if (m_i > 1) {
                        householder_apply_right_d_amx(taup_i, work.data(),
                                                      &A[(i + 1) * n + (i + 1)], m_i - 1, n_i2, n);
                    }

                    // Store reflector back into row i
                    for (size_t c = 0; c < n_i2; ++c)
                        A[i * n + (i + 1 + c)] = work[c];
                    A[i * n + (i + 1)] = e[i + 1];
                }
            }

            // ============================================================
            //  AMX + AVX512: GEBRD (bidiagonal reduction, float)
            //  Matches scalar gebrd exactly, using AMX for dot/Householder
            // ============================================================

            AMX_TARGET_ATTR
            static void gebrd_f_amx(float *A, size_t m, size_t n,
                                    float *d, float *e, float *tauq, float *taup) {
                size_t k = std::min(m, n);
                std::vector<float> work(std::max(m, n));

                for (size_t i = 0; i < k; ++i) {
                    size_t m_i = m - i;
                    size_t n_i = n - i;

                    // Extract column i below diagonal (strided by n in row-major)
                    for (size_t r = 0; r < m_i; ++r)
                        work[r] = A[(i + r) * n + i];

                    float tauq_i = householder_generate_f_amx(work.data(), m_i, &d[i]);
                    tauq[i] = tauq_i;
                    work[0] = 1.0f;

                    // Apply left reflector to submatrix rows i..m-1, cols i+1..n-1
                    if (n_i > 1) {
                        householder_apply_left_f_amx(tauq_i, work.data(),
                                                     &A[i * n + (i + 1)], m_i, n_i - 1, n);
                    }

                    // Store reflector back into column i
                    for (size_t r = 0; r < m_i; ++r)
                        A[(i + r) * n + i] = work[r];
                    A[i * n + i] = d[i];

                    if (i + 1 >= k) {
                        taup[i] = 0.0f;
                        break;
                    }

                    size_t n_i2 = n_i - 1;
                    // Extract row i right of superdiagonal (contiguous in row-major)
                    for (size_t c = 0; c < n_i2; ++c)
                        work[c] = A[i * n + (i + 1 + c)];

                    float taup_i = householder_generate_f_amx(work.data(), n_i2, &e[i + 1]);
                    taup[i] = taup_i;

                    // Apply right reflector to submatrix rows i+1..m-1, cols i+1..n-1
                    if (m_i > 1) {
                        householder_apply_right_f_amx(taup_i, work.data(),
                                                      &A[(i + 1) * n + (i + 1)], m_i - 1, n_i2, n);
                    }

                    // Store reflector back into row i
                    for (size_t c = 0; c < n_i2; ++c)
                        A[i * n + (i + 1 + c)] = work[c];
                    A[i * n + (i + 1)] = e[i + 1];
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

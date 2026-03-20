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

/// AVX2-optimized main GELSD solver.
/// Matches the scalar LstSqGelsdSolver.hpp exactly, with AVX2 SIMD acceleration.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "LstSqGelsdBackTransform_avx2.hpp"
#include "LstSqGelsdGebrd_avx2.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Main GELSD solver (double, AVX2)
            //  Matches scalar lstsq_gelsd_scalar exactly, with AVX2 helpers
            // ============================================================

            AVX2_TARGET_ATTR
            static int lstsq_gelsd_double_avx2_impl(const double *A, const double *b, double *x,
                                                    size_t m, size_t n, double rcond) {
                if (m == 0 || n == 0) return 0;

                size_t k = std::min(m, n);

                // Copy A since gebrd modifies it in-place
                std::vector<double> A_work(A, A + m * n);
                std::vector<double> b_work(b, b + m);

                std::vector<double> d(k), e(k, 0.0);
                std::vector<double> tauq(k), taup(k);
                gebrd_d_avx2(A_work.data(), m, n, d.data(), e.data(),
                             tauq.data(), taup.data());

                std::vector<double> s(k);
                std::vector<double> U_bidiag(k * k);
                std::vector<double> VT_bidiag(k * k);
                bdsvd_dc_d(d.data(), e.data(), k, s.data(),
                           U_bidiag.data(), VT_bidiag.data());

                // U_full is m x k row-major
                std::vector<double> U_full(m * k, 0.0);
                // VT_full is k x n row-major
                std::vector<double> VT_full(k * n, 0.0);

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        U_full[i * k + j] = U_bidiag[i * k + j];

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        VT_full[i * n + j] = VT_bidiag[i * k + j];

                multiply_left_q_d_avx2(A_work.data(), m, n, tauq.data(), k,
                                       U_full.data(), k);
                multiply_right_pt_d_avx2(A_work.data(), m, n, taup.data(), k,
                                         VT_full.data(), k);

                double smax = (k > 0) ? s[0] : 0.0;
                double rcond_abs = (rcond < 0.0) ? (std::numeric_limits<double>::epsilon() * smax) : (rcond * smax);
                int rank = 0;
                for (size_t i = 0; i < k; ++i)
                    if (s[i] > rcond_abs) ++rank;
                int r = rank;

                // c = U^T * b
                std::vector<double> c(k, 0.0);
                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < m; ++j)
                        c[i] += U_full[j * k + i] * b_work[j];

                for (size_t i = 0; i < k; ++i)
                    c[i] = ((int) i < r) ? (c[i] / s[i]) : 0.0;

                // x = V * c
                for (size_t i = 0; i < n; ++i) {
                    x[i] = 0.0;
                    for (size_t j = 0; j < k; ++j)
                        x[i] += VT_full[j * n + i] * c[j];
                }

                return r;
            }

            // ============================================================
            //  Main GELSD solver (float, AVX2)
            //  Matches scalar lstsq_gelsd_scalar exactly, with AVX2 helpers
            // ============================================================

            AVX2_TARGET_ATTR
            static int lstsq_gelsd_float_avx2_impl(const float *A, const float *b, float *x,
                                                   size_t m, size_t n, float rcond) {
                if (m == 0 || n == 0) return 0;

                size_t k = std::min(m, n);

                std::vector<float> A_work(A, A + m * n);
                std::vector<float> b_work(b, b + m);

                std::vector<float> d(k), e(k, 0.0f);
                std::vector<float> tauq(k), taup(k);
                gebrd_f_avx2(A_work.data(), m, n, d.data(), e.data(),
                             tauq.data(), taup.data());

                std::vector<float> s(k);
                std::vector<float> U_bidiag(k * k);
                std::vector<float> VT_bidiag(k * k);
                bdsvd_dc_f(d.data(), e.data(), k, s.data(),
                           U_bidiag.data(), VT_bidiag.data());

                std::vector<float> U_full(m * k, 0.0f);
                std::vector<float> VT_full(k * n, 0.0f);

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        U_full[i * k + j] = U_bidiag[i * k + j];

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        VT_full[i * n + j] = VT_bidiag[i * k + j];

                multiply_left_q_f_avx2(A_work.data(), m, n, tauq.data(), k,
                                       U_full.data(), k);
                multiply_right_pt_f_avx2(A_work.data(), m, n, taup.data(), k,
                                         VT_full.data(), k);

                float smax = (k > 0) ? s[0] : 0.0f;
                float rcond_abs = (rcond < 0.0f) ? (std::numeric_limits<float>::epsilon() * smax) : (rcond * smax);
                int rank = 0;
                for (size_t i = 0; i < k; ++i)
                    if (s[i] > rcond_abs) ++rank;
                int r = rank;

                std::vector<float> c(k, 0.0f);
                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < m; ++j)
                        c[i] += U_full[j * k + i] * b_work[j];

                for (size_t i = 0; i < k; ++i)
                    c[i] = ((int) i < r) ? (c[i] / s[i]) : 0.0f;

                for (size_t i = 0; i < n; ++i) {
                    x[i] = 0.0f;
                    for (size_t j = 0; j < k; ++j)
                        x[i] += VT_full[j * n + i] * c[j];
                }

                return r;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

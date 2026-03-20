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
#include <limits>
#include <vector>

#include "LstSqGelsdBackTransform.hpp"
#include "LstSqGelsdBdsvdQr.hpp"
#include "LstSqGelsdDc.hpp"
#include "LstSqGelsdGebrd.hpp"
#include "LstSqGelsdTraits.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Main GELSD solver (row-major)
            // ============================================================

            template<typename T>
            int lstsq_gelsd_scalar(const T *A, const T *b, T *x,
                                   size_t m, size_t n, T rcond) {
                if (m == 0 || n == 0) return 0;

                size_t k = std::min(m, n);

                // Copy A since gebrd modifies it in-place
                std::vector<T> A_work(A, A + m * n);
                std::vector<T> b_work(b, b + m);

                std::vector<T> d(k), e(k, T(0));
                std::vector<T> tauq(k), taup(k);
                gebrd(A_work.data(), m, n, d.data(), e.data(),
                      tauq.data(), taup.data());

                std::vector<T> s(k);
                std::vector<T> U_bidiag(k * k);
                std::vector<T> VT_bidiag(k * k);
                bdsvd_dc(d.data(), e.data(), k, s.data(),
                         U_bidiag.data(), VT_bidiag.data());

                // U_full is m x k row-major
                std::vector<T> U_full(m * k, T(0));
                // VT_full is k x n row-major
                std::vector<T> VT_full(k * n, T(0));

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        U_full[i * k + j] = U_bidiag[i * k + j];

                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < k; ++j)
                        VT_full[i * n + j] = VT_bidiag[i * k + j];

                multiply_left_q(A_work.data(), m, n, tauq.data(), k,
                                U_full.data(), k);
                multiply_right_pt(A_work.data(), m, n, taup.data(), k,
                                  VT_full.data(), k);

                T smax = (k > 0) ? s[0] : T(0);
                T rcond_abs = (rcond < T(0)) ? (std::numeric_limits<T>::epsilon() * smax) : (rcond * smax);
                int rank = 0;
                for (size_t i = 0; i < k; ++i)
                    if (s[i] > rcond_abs) ++rank;
                int r = rank;

                // c = U^T * b
                std::vector<T> c(k, T(0));
                for (size_t i = 0; i < k; ++i)
                    for (size_t j = 0; j < m; ++j)
                        c[i] += U_full[j * k + i] * b_work[j];

                for (size_t i = 0; i < k; ++i)
                    c[i] = ((int) i < r) ? (c[i] / s[i]) : T(0);

                // x = V * c
                for (size_t i = 0; i < n; ++i) {
                    x[i] = T(0);
                    for (size_t j = 0; j < k; ++j)
                        x[i] += VT_full[j * n + i] * c[j];
                }

                return r;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

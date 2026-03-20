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

#include <cmath>
#include <cstddef>
#include <cstring>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  BLAS Level 1 helpers
            // ============================================================

            template<typename T>
            static inline T dot(const T *x, const T *y, size_t n) {
                T s = 0;
                for (size_t i = 0; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            template<typename T>
            static inline T nrm2(const T *x, size_t n) {
                T scale = 0, ssq = 1;
                for (size_t i = 0; i < n; ++i) {
                    T xi = x[i];
                    if (xi != T(0)) {
                        T absxi = std::abs(xi);
                        if (scale < absxi) {
                            ssq = T(1) + ssq * (scale / absxi) * (scale / absxi);
                            scale = absxi;
                        } else {
                            ssq += (absxi / scale) * (absxi / scale);
                        }
                    }
                }
                return scale * std::sqrt(ssq);
            }

            template<typename T>
            static inline void copy(const T *x, T *y, size_t n) {
                std::memcpy(y, x, n * sizeof(T));
            }

            template<typename T>
            static inline void scal(T a, T *x, size_t n) {
                for (size_t i = 0; i < n; ++i) x[i] *= a;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

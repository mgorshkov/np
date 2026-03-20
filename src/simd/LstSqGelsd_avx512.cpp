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

/// AVX512-optimized implementation of the GELSD least-squares solver.
/// Matches the scalar algorithm (LstSqGelsdScalar.hpp) exactly,
/// but with AVX512 SIMD acceleration for dot products, scaling,
/// and Householder operations.
/// All matrices are stored in row-major layout.
///
/// This file provides only the non-static wrapper functions called
/// from the dispatch mechanism. All implementation is in the headers.

#include <np/internal/cpu/LstSqGelsdSimdAttr.hpp>

#include <cstddef>

namespace np {
    namespace internal {
        namespace cpu {
            // Forward declarations from scalar compilation units
            void bdsvd_dc_d(const double *d_in, const double *e_in, size_t n,
                            double *s, double *U, double *VT);
            void bdsvd_dc_f(const float *d_in, const float *e_in, size_t n,
                            float *s, float *U, float *VT);
        } // namespace cpu
    } // namespace internal
} // namespace np

#include <np/internal/cpu/LstSqGelsdSolver_avx512.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Non-static wrapper: double
            // ============================================================

            AVX512_TARGET_ATTR
            int lstsq_gelsd_double_avx512(const double *A, const double *b, double *x,
                                           size_t m, size_t n, double rcond) {
                return lstsq_gelsd_double_avx512_impl(A, b, x, m, n, rcond);
            }

            // ============================================================
            //  Non-static wrapper: float
            // ============================================================

            AVX512_TARGET_ATTR
            int lstsq_gelsd_float_avx512(const float *A, const float *b, float *x,
                                           size_t m, size_t n, float rcond) {
                return lstsq_gelsd_float_avx512_impl(A, b, x, m, n, rcond);
            }

        } // namespace cpu
    } // namespace internal
} // namespace np

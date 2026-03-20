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

/// AVX2-optimized implementation of the GELSD least-squares solver.
/// Matches the scalar algorithm (LstSqGelsdScalar.hpp) exactly,
/// but with AVX2 SIMD acceleration for dot products, scaling,
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

#include <np/internal/cpu/LstSqGelsdSolver_avx2.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Non-static wrapper: double
            // ============================================================

            AVX2_TARGET_ATTR
            int lstsq_gelsd_double_avx2(const double *A, const double *b, double *x,
                                         size_t m, size_t n, double rcond) {
                return lstsq_gelsd_double_avx2_impl(A, b, x, m, n, rcond);
            }

            // ============================================================
            //  Non-static wrapper: float
            // ============================================================

            AVX2_TARGET_ATTR
            int lstsq_gelsd_float_avx2(const float *A, const float *b, float *x,
                                         size_t m, size_t n, float rcond) {
                return lstsq_gelsd_float_avx2_impl(A, b, x, m, n, rcond);
            }

            // ============================================================
            //  Non-static wrappers for AVX2 functions (debug/testing)
            // ============================================================

            AVX2_TARGET_ATTR
            void gebrd_d_avx2_wrapper(double *A, size_t m, size_t n,
                                       double *d, double *e,
                                       double *tauq, double *taup) {
                gebrd_d_avx2(A, m, n, d, e, tauq, taup);
            }

            AVX2_TARGET_ATTR
            void multiply_left_q_d_avx2_wrapper(const double *A, size_t m, size_t n,
                                                  const double *tauq, size_t k,
                                                  double *U, size_t nru) {
                multiply_left_q_d_avx2(A, m, n, tauq, k, U, nru);
            }

            AVX2_TARGET_ATTR
            void multiply_right_pt_d_avx2_wrapper(const double *A, size_t m, size_t n,
                                                     const double *taup, size_t k,
                                                     double *VT, size_t ncv) {
                multiply_right_pt_d_avx2(A, m, n, taup, k, VT, ncv);
            }

            // ============================================================
            //  Wrappers for larft/larfb (debug/testing)
            // ============================================================

            AVX2_TARGET_ATTR
            void larft_d_avx2_wrapper(const double *Y, size_t m, size_t NB,
                                       const double *tau, size_t ldy,
                                       double *T_, size_t ldT) {
                larft_d_avx2(Y, m, NB, tau, ldy, T_, ldT);
            }

            AVX2_TARGET_ATTR
            void larfb_right_d_avx2_wrapper(const double *Y, size_t n, size_t NB,
                                              const double *T_, size_t ldT,
                                              double *C, size_t m, size_t ldc,
                                              size_t ldy) {
                larfb_right_d_avx2(Y, n, NB, T_, ldT, C, m, ldc, ldy);
            }

            AVX2_TARGET_ATTR
            double dot_d_avx2_wrapper(const double *x, const double *y, size_t n) {
                return dot_d_avx2(x, y, n);
            }

        } // namespace cpu
    } // namespace internal
} // namespace np

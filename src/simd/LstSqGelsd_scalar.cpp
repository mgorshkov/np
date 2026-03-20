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

/// Scalar (no-SIMD) implementation of the GELSD least-squares solver (double).
/// Implements the same divide-and-conquer SVD algorithm as LAPACK's DGELSD.
/// All matrices are stored in row-major layout.
///
/// This file instantiates the templated implementation from LstSqGelsdScalar.hpp
/// for the double type.

#include <np/internal/cpu/LstSqGelsdScalar.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // Explicit instantiation of all templated functions for double

            template double dot(const double *, const double *, size_t);
            template double nrm2(const double *, size_t);
            template void copy(const double *, double *, size_t);
            template void scal(double, double *, size_t);
            template double householder_generate(double *, size_t, double *);
            template void householder_apply_left(double, const double *, double *, size_t, size_t, size_t);
            template void householder_apply_right(double, const double *, double *, size_t, size_t, size_t);
            template void gebrd(double *, size_t, size_t, double *, double *, double *, double *);
            template void bdsvd_qr(const double *, const double *, size_t, double *, double *, double *);
            template void bdsvd_dc(const double *, const double *, size_t, double *, double *, double *);
            template void multiply_left_q(const double *, size_t, size_t, const double *, size_t, double *, size_t);
            template void multiply_right_pt(const double *, size_t, size_t, const double *, size_t, double *, size_t);
            template int lstsq_gelsd_scalar(const double *, const double *, double *, size_t, size_t, double);

            // Non-template wrappers for AVX2/AVX512/AMX code that use extern declarations
            void bdsvd_dc_d(const double *d_in, const double *e_in, size_t n,
                            double *s, double *U, double *VT) {
                bdsvd_dc(d_in, e_in, n, s, U, VT);
            }

            // Legacy entry point for backward compatibility
            int lstsq_gelsd_double_scalar(const double *A, const double *b, double *x,
                                          size_t m, size_t n, double rcond) {
                return lstsq_gelsd_scalar(A, b, x, m, n, rcond);
            }

        } // namespace cpu
    } // namespace internal
} // namespace np

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

/// Scalar (no-SIMD) implementation of the GELSD least-squares solver (float).
/// Implements the same divide-and-conquer SVD algorithm as LAPACK's SGELSD.
/// All matrices are stored in row-major layout.
///
/// This file instantiates the templated implementation from LstSqGelsdScalar.hpp
/// for the float type.

#include <np/internal/cpu/LstSqGelsdScalar.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // Explicit instantiation of all templated functions for float

            template float dot(const float *, const float *, size_t);
            template float nrm2(const float *, size_t);
            template void copy(const float *, float *, size_t);
            template void scal(float, float *, size_t);
            template float householder_generate(float *, size_t, float *);
            template void householder_apply_left(float, const float *, float *, size_t, size_t, size_t);
            template void householder_apply_right(float, const float *, float *, size_t, size_t, size_t);
            template void gebrd(float *, size_t, size_t, float *, float *, float *, float *);
            template void bdsvd_qr(const float *, const float *, size_t, float *, float *, float *);
            template void bdsvd_dc(const float *, const float *, size_t, float *, float *, float *);
            template void multiply_left_q(const float *, size_t, size_t, const float *, size_t, float *, size_t);
            template void multiply_right_pt(const float *, size_t, size_t, const float *, size_t, float *, size_t);
            template int lstsq_gelsd_scalar(const float *, const float *, float *, size_t, size_t, float);

            // Non-template wrappers for AVX2/AVX512/AMX code that use extern declarations
            void bdsvd_dc_f(const float *d_in, const float *e_in, size_t n,
                            float *s, float *U, float *VT) {
                bdsvd_dc(d_in, e_in, n, s, U, VT);
            }

            // Legacy entry point for backward compatibility
            int lstsq_gelsd_float_scalar(const float *A, const float *b, float *x,
                                         size_t m, size_t n, float rcond) {
                return lstsq_gelsd_scalar(A, b, x, m, n, rcond);
            }

        } // namespace cpu
    } // namespace internal
} // namespace np

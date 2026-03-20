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

#include <cstddef>
#include <functional>
#include <vector>

#include <np/internal/CpuDispatch.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            /// @defgroup gelsd_lstsq GELSD-based Least Squares (SIMD-optimized)
            ///
            /// Implements the same algorithm as LAPACK's DGELSD / SGELSD:
            /// divide-and-conquer SVD-based least-squares solver.
            ///
            /// The algorithm follows these steps:
            ///   1. Reduce A to bidiagonal form using Householder reflections (GEBRD).
            ///   2. Compute SVD of the bidiagonal matrix using divide-and-conquer (DBDSDC).
            ///   3. Back-transform to get SVD of A.
            ///   4. Solve the least-squares problem min ||b - A*x|| using the SVD.
            ///
            /// This is the same algorithm used by numpy.linalg.lstsq (which calls LAPACK
            /// DGELSD/SGELSD).

            // ---- std::function types for SIMD dispatch ----

            /// Templatized std::function type for the GELSD least-squares solver.
            /// T is the floating-point type (double or float).
            /// Signature: int(const T *A, const T *b, T *x, size_t m, size_t n, T rcond)
            template<typename T>
            using lstsq_gelsd_fn = std::function<int(const T *, const T *, T *, size_t, size_t, T)>;

            // ---- Function pointer declarations (set by init) ----

            extern lstsq_gelsd_fn<double> lstsq_gelsd_double;
            extern lstsq_gelsd_fn<float> lstsq_gelsd_float;

            /// Initialize the GELSD function pointers based on runtime CPU detection.
            void init_lstsq_gelsd_dispatch();

        }// namespace cpu
    }// namespace internal
}// namespace np

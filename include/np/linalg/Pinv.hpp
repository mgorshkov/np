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

#include <np/Array.hpp>
#include <np/Creators.hpp>
#include <np/Exception.hpp>
#include <np/linalg/Inv.hpp>

namespace np {
    namespace linalg {
        /* Moore-Penrose pseudoinverse via Tikhonov regularization.
         *
         * Computes the pseudoinverse of a matrix A using the formula:
         *   pinv(A) ≈ (AᵀA + λI)⁻¹Aᵀ   for m ≥ n (tall matrix)
         *   pinv(A) ≈ Aᵀ(AAᵀ + λI)⁻¹   for m < n (wide matrix)
         *
         * where λ is a small regularization parameter (default 1e-8).
         * This regularization ensures invertibility for singular matrices.
         *
         * array - matrix to invert (2D)
         * lambda - regularization parameter (optional)
         * returns pseudoinverse matrix
         */
        template<typename DType, typename Derived, typename Storage>
        inline auto pinv(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, DType lambda = 1e-8) {
            if (array.empty()) {
                return Array<DType>{};
            }
            if (array.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "pinv expects a 2D array");
            }
            auto shape = array.shape();
            Size m = shape[0];
            Size n = shape[1];

            // Transpose of A
            auto A_T = array.transpose();

            if (m >= n) {
                // Tall or square matrix: use (AᵀA + λI)⁻¹Aᵀ
                auto ATA = A_T.dot(array);// n x n
                auto I = eye<DType>(n);   // identity n x n
                auto regularized = ATA + I * lambda;
                auto inv_reg = inv(regularized);// n x n
                return inv_reg.dot(A_T);        // n x m
            } else {
                // Wide matrix: use Aᵀ(AAᵀ + λI)⁻¹
                auto AAT = array.dot(A_T);// m x m
                auto I = eye<DType>(m);   // identity m x m
                auto regularized = AAT + I * lambda;
                auto inv_reg = inv(regularized);// m x m
                return A_T.dot(inv_reg);        // n x m
            }
        }
    }// namespace linalg
}// namespace np

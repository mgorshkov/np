/*
C++ numpy-like template-based array implementation

Copyright (c) 2023 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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
#include <np/internal/cuda/Tikhonov.hpp>
#include <np/linalg/Inv.hpp>

namespace np {
    namespace linalg {
        // Least squares with Cholesky/Tikhonov Regularized EVD
        template<typename DType, typename Derived, typename Storage>
        inline auto lstsq(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &A,
            const ndarray::internal::NDArrayBase<DType, Derived, Storage> &b, DType lambda = 1e-6) {
            if (A.ndim() != 2) {
                throw std::invalid_argument("Expected 2D array.");
            }
            if (b.ndim() != 1) {
                throw std::invalid_argument("Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                throw std::invalid_argument("Invalid size.");
            }
            if (A.size() < 100) {
                // Use Cholesky for small matrices
                auto column = A.shape()[0];
                auto ones = Array<np::float_>{Shape{column}, 1.0};
                auto a = column_stack(ones, A);
                auto aT = a.transpose();
                auto coeffs = inv(aT.dot(a)).dot(aT).dot(b);
                auto coeff = coeffs["1:"];
                auto intercept = coeffs.get(0);
                return A * coeff + intercept;
            }
            // Use Tikhonov for big matrices
            std::vector<DType> x;
            x.reserve(n);
            internal::cuda::lstsqTikhonov(A.data(), b.data(), x.data(), m, n, lambda);
            return NDArrayDynamic<DType>(x);
        }
    }// namespace linalg

}// namespace np

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

namespace np {
    namespace linalg {
        /* Matrix inversion by partition method: https://paramanands.blogspot.com/2012/08/matrix-inversion-partition-method.html
         *
         * A        [A B
         *   n+1 =   C d]
         *
         * A.shape() = (n, n)
         * B.shape() = (1, n)
         * C.shape() = (n, 1)
         * d.shape() = (1, 1)
         *
         * A^(-1)   [X y
         *   n+1 =   Z t]
         *
         * X.shape() = (n, n)
         * y.shape() = (1, n)
         * Z.shape() = (n, 1)
         * t.shape() = (1, 1)
         *
         * t = (d - CA^(-1)B)^(-1)
         * Y = -A^(-1)Bt
         * Z = -CA^(-1)t
         * X = A^(-1)(I - BZ)
         *
         * array - square matrix to invert
         * returns inverted matrix
         */
        template<typename DType, typename Derived, typename Storage>
        inline auto inv(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
            if (array.empty()) {
                return Array<DType>{};
            }

            auto shape = array.shape();
            Size n = shape[0];
            if (n == 1) {
                auto denominator = array.get(0);
                if (internal::element_equal(denominator, 0.0)) {
                    throw std::runtime_error("Singular matrix");
                }
                return Array<DType>{1.0 / denominator};
            }
            NDArrayDynamic<float_> result{shape};
            if constexpr (Storage::kDepth < ndarray::internal::kMaxArrayDims) {
                std::string a_index = "0:" + std::to_string(n - 1) + ", 0:" + std::to_string(n - 1);
                const auto A = array[a_index];

                std::string b_index = "0:" + std::to_string(n - 1) + "," + std::to_string(n - 1) + ":" + std::to_string(n);
                const auto B = array[b_index];

                std::string c_index = std::to_string(n - 1) + ":" + std::to_string(n) + ", 0:" + std::to_string(n - 1);
                const auto C = array[c_index];

                std::string d_index =
                        std::to_string(n - 1) + ":" + std::to_string(n) + "," + std::to_string(n - 1) + ":" +
                        std::to_string(n);
                const auto d = array[d_index];

                const auto A_minus_1 = inv(A);
                auto denominator = d.get(0) - C.dot(A_minus_1).dot(B).get(0);
                if (internal::element_equal(denominator, 0.0)) {
                    throw std::runtime_error("Singular matrix");
                }
                const auto t = 1.0 / denominator;
                const auto Y = A_minus_1.multiply(-1).dot(B).multiply(t);
                const auto Z = C.multiply(-1).dot(A_minus_1).multiply(t);
                const auto X = A_minus_1.dot(eye(n - 1) - B.dot(Z));

                Size index = 0;
                for (Size i = 0; i < n - 1; ++i) {    // rows
                    for (Size j = 0; j < n - 1; ++j) {// columns
                        result.set(index++, X.get(i * (n - 1) + j));
                    }
                    result.set(index++, Y.get(i));
                }
                for (Size j = 0; j < n - 1; ++j) {
                    result.set(index++, Z.get(j));
                }
                result.set(index, t);
            } else {
                assert(false);
            }
            return result;
        }
    }// namespace linalg

}// namespace np
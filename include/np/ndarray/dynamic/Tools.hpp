/*
C++ numpy-like template-based array implementation

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/ndarray/dynamic/NDArrayDynamicDecl.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {

            template<typename DType, typename Storage>
            inline static DType
            vectorCorr(const NDArrayDynamic<DType, Storage> &v1, const NDArrayDynamic<DType, Storage> &v2) {
                auto sh1 = v1.shape();
                if (sh1.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                auto sh2 = v2.shape();
                if (sh2.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                if (v1.len() != v2.len()) {
                    throw std::runtime_error("Sizes are different");
                }

                DType sum1 = 0;
                DType sum2 = 0;
                DType sum12 = 0;
                DType squareSum1 = 0;
                DType squareSum2 = 0;

                Size n = v1.len();

                for (Size i = 0; i < n; i++) {
                    sum1 += v1.get(i);
                    sum2 += v2.get(i);
                    sum12 += sum12 + v1.get(i) * v2.get(i);

                    squareSum1 += v1.get(i) * v1.get(i);
                    squareSum2 += v2.get(i) * v2.get(i);
                }

                DType corr = static_cast<DType>(n * sum12 - sum1 * sum2) / static_cast<DType>(sqrt((n * squareSum1 - sum1 * sum1) * (n * squareSum2 - sum2 * sum2)));

                return corr;
            }

            template<typename DType, typename Storage>
            inline static DType
            vectorCov(const NDArrayDynamic<DType, Storage> &v1, const NDArrayDynamic<DType, Storage> &v2) {
                auto sh1 = v1.shape();
                if (sh1.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                auto sh2 = v2.shape();
                if (sh2.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                if (v1.len() != v2.len()) {
                    throw std::runtime_error("Sizes are different");
                }

                auto v1_mean = v1.mean();
                auto v2_mean = v2.mean();

                DType sum = 0;

                for (Size i = 0; i < v1.len(); ++i) {
                    sum += ((v1[i] - v1_mean) * (v2[i] - v2_mean));
                }

                return sum / (v1.len() - 1);
            }
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

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

#include <np/ndarray/internal/Math.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>

namespace np {
    namespace ndarray {
        namespace internal {

            template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
            inline static float_ vectorCorr(const NDArrayBase<DType, Derived1, Storage1> &v1, const NDArrayBase<DType, Derived2, Storage2> &v2) {
                auto sh1 = v1.shape();
                if (sh1.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                auto sh2 = v2.shape();
                if (sh2.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                if (v1.len() != v2.len()) {
                    throw std::runtime_error("Sizes are different");
                }

                float_ sum1 = 0;
                float_ sum2 = 0;
                float_ sum12 = 0;
                float_ squareSum1 = 0;
                float_ squareSum2 = 0;

                Size n = v1.len();

                for (Size i = 0; i < n; i++) {
                    sum1 += v1.get(i);
                    sum2 += v2.get(i);
                    sum12 += v1.get(i) * v2.get(i);

                    squareSum1 += v1.get(i) * v1.get(i);
                    squareSum2 += v2.get(i) * v2.get(i);
                }

                float_ minusResult1 = n * squareSum1 - sum1 * sum1;
                float_ minusResult2 = n * squareSum2 - sum2 * sum2;
                float_ mulResult = minusResult1 * minusResult2;
                float_ sqrtResult = std::sqrt(mulResult);
                float_ corr = (n * sum12 - sum1 * sum2) / sqrtResult;

                return corr;
            }

            template<typename Derived1, typename Storage1, typename Derived2, typename Storage2, typename DType>
            inline static float_ vectorCov(const NDArrayBase<DType, Derived1, Storage1> &v1, const NDArrayBase<DType, Derived2, Storage2> &v2) {
                auto sh1 = v1.shape();
                if (sh1.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                auto sh2 = v2.shape();
                if (sh2.size() != 1)
                    throw std::runtime_error("Only 1D arrays supported");

                if (v1.len() != v2.len()) {
                    throw std::runtime_error("Sizes are different");
                }

                float_ sum{};
                for (Size i = 0; i < v1.len(); ++i) {
                    float_ result1;
                    ndarray::internal::subtract<float_>(v1.get(i), v1.mean(), result1);
                    float_ result2;
                    ndarray::internal::subtract<float_>(v2.get(i), v2.mean(), result2);
                    float_ result3;
                    ndarray::internal::multiply<float_>(result1, result2, result3);
                    sum += result3;
                }

                float_ result = sum / (v1.len() - 1);
                return result;
            }

        }// namespace internal
    }    // namespace ndarray
}// namespace np

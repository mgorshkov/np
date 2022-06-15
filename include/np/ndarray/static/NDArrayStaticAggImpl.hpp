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

#include <np/ndarray/static/NDArrayStaticDecl.hpp>
#include <np/ndarray/static/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {

            // Array-wise sum
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::sum() const {
                DType result{};
                for (Size i = 0; i < SizeT; ++i) {
                    const auto &element = at(i);
                    result += element.sum();
                }
                return result;
            }

            // Array-wise minimum value
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::min() const {
                DType result{};
                bool inited{false};
                for (Size i = 0; i < SizeT; ++i) {
                    const auto &element = at(i);
                    DType min_ = element.min();
                    if (!inited || min_ < result) {
                        result = min_;
                        inited = true;
                    }
                }
                return result;
            }

            // Array-wise maximum value
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::max() const {
                DType result{};
                bool inited{false};
                for (Size i = 0; i < SizeT; ++i) {
                    const auto &element = at(i);
                    DType max_ = element.max();
                    if (!inited || max_ > result) {
                        result = max_;
                        inited = true;
                    }
                }
                return result;
            }

            // Cumulative sum of the elements
            template<typename DType, Size SizeT, Size... SizeTs>
            inline auto NDArrayStatic<DType, SizeT, SizeTs...>::cumsum() const {
                NDArrayStatic<DType, (SizeT * ... * SizeTs)> result;
                DType sum = 0;
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    sum += element;
                    result.set(i, sum);
                }
                return result;
            }

            // Mean
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::mean() const {
                DType result{};
                for (Size i = 0; i < SizeT; ++i) {
                    const auto &element = at(i);
                    result += element.mean();
                }
                if constexpr (SizeT == 0) {
                    return 0;
                } else {
                    return result / SizeT;
                }
            }

            // Median
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::median() const {
                auto s = size();
                if (s == 0)
                    return 0;
                auto array = m_ArrayImpl;
                auto middle1 = array.begin();
                std::advance(middle1, s / 2);
                if (s % 2 == 0) {
                    auto middle2 = array.begin();
                    std::advance(middle2, (s - 1) / 2);

                    std::nth_element(array.begin(),
                                     middle1,
                                     array.end());

                    std::nth_element(array.begin(),
                                     middle2,
                                     array.end());

                    // Find the average of values at indices size / 2 and (size - 1) / 2
                    return static_cast<DType>((array.get((s - 1) / 2) + array.get(s / 2)) / 2.0);
                }
                std::nth_element(array.begin(),
                                 middle1,
                                 array.end());
                return static_cast<DType>(array.get(s / 2));
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::cov() const {
                auto sh = shape();
                if (sh.size() == 0)
                    return 0;
                if (sh.size() != 1 && sh.size() != 2)
                    throw std::runtime_error("Only 1D and 2D arrays supported");

                if (sh.size() == 1)
                    return NDArrayDynamic<DType>{1.0};

                Shape resultShape({len(), len()});
                NDArrayDynamic<DType> result{resultShape};
                for (Size i = 0; i < len(); ++i) {
                    Shape subResultShape({1, len()});
                    NDArrayDynamic<DType> subResult{subResultShape};
                    for (Size j = 0; j < len(); ++j) {
                        auto subArray1 = at(i);
                        auto subArray2 = at(j);
                        DType c = vectorCov(subArray1, subArray2);
                        set(subResult, j, c);
                    }
                    set(result, i, subResult);
                }
                return result;
            }

            // Correlation coefficient
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::corrcoef() const {
                auto sh = shape();
                if (sh.size() == 0)
                    return NDArrayDynamic<DType>{};

                if (sh.size() != 1 && sh.size() != 2)
                    throw std::runtime_error("Only 1D and 2D arrays supported");

                if (sh.size() == 1)
                    return NDArrayDynamic<DType>{1};

                Shape resultShape({len(), len()});
                NDArrayDynamic<DType> result{resultShape};
                for (Size i = 0; i < len(); ++i) {
                    Shape subResultShape({1, len()});
                    NDArrayDynamic<DType> subResult{subResultShape};
                    for (Size j = 0; j < len(); ++j) {
                        auto subArray1 = at(i);
                        auto subArray2 = at(j);
                        DType c = vectorCorr(subArray1, subArray2);
                        subResult.set(j, c);
                    }
                    result.set(i, subResult);
                }
                return result;
            }

            // Standard deviation
            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::std_() const {
                NDArrayDynamic<DType> x{shape()};
                DType m = mean();
                for (Size i = 0; i < size(); ++i) {
                    auto a = abs(get(i) - m);
                    x.set(i, a * a);
                }
                return static_cast<DType>(std::sqrt(x.mean()));
            }
        }// namespace array_static
    }// namespace ndarray
}// namespace np

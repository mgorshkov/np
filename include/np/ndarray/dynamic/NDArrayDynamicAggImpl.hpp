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

#include <algorithm>

#include <np/ndarray/dynamic/NDArrayDynamicDecl.hpp>

namespace np::ndarray::array_dynamic {

    // Array-wise sum
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::sum() const {
        DType result{};
        auto size = m_ArrayImpl.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto& element = get(i);
            result += element;
        }
        return result;
    }

    // Array-wise minimum value
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::min() const {
        DType result{};
        bool inited{false};
        auto size = m_ArrayImpl.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto& element = get(i);
            if (!inited || element < result) {
                result = element;
                inited = true;
            }
        }
        return result;
    }

    // Array-wise maximum value
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::max() const {
        DType result{};
        bool inited{false};
        auto size = m_ArrayImpl.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto& element = get(i);
            if (!inited || element > result) {
                result = element;
                inited = true;
            }
        }
        return result;
    }

    // Cumulative sum of the elements
    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::cumsum() const {
        Shape sh = shape();
        sh.flatten();
        NDArrayDynamic<DType, Storage> result{sh};
        DType sum = 0;
        auto size = m_ArrayImpl.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto& element = get(i);
            sum += element;
            result.m_ArrayImpl.set(i, sum);
        }
        return result;
    }

    // Mean
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::mean() const {
        DType result{};    
        auto size = m_ArrayImpl.size();
        for (std::size_t i = 0; i < size; ++i) {
            const auto& element = get(i);
            result += element;
        }
        return size != 0 ? result / size : 0;
    }

    // Median
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::median() const {
        auto size = m_ArrayImpl.size();
        if (size == 0)
            return 0;
        auto v = m_ArrayImpl.m_Impl;
        if (size % 2 == 0) {
            std::nth_element(v.begin(),
                v.begin() + size / 2,
                v.end());

            std::nth_element(v.begin(),
                v.begin() + (size - 1) / 2,
                v.end());
    
            // Find the average of values at indices size / 2 and (size - 1) / 2
            return static_cast<DType>(v[(size - 1) / 2] + v[(size / 2)] / 2.0);
        }
        std::nth_element(v.begin(),
            v.begin() + size / 2,
            v.end());
        return static_cast<DType>(v[size / 2]);
    }

    template<typename DType, typename Storage>
    inline static DType vectorCov(const NDArrayDynamic<DType, Storage>& v1, const NDArrayDynamic<DType, Storage>& v2) {
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

    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::cov() const {
        auto sh = shape();
        if (sh.size() == 0)
            return 0;
        if (sh.size() != 1 && sh.size() != 2)
            throw std::runtime_error("Only 1D and 2D arrays supported");

        if (sh.size() == 1)
            return NDArrayDynamic<DType, Storage>{1.0};

        Shape resultShape({len(), len()});
        NDArrayDynamic<DType, Storage> result{resultShape};
        for (Size i = 0; i < len(); ++i) {
            Shape subResultShape({1, len()});
            NDArrayDynamic<DType, Storage> subResult{subResultShape};
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

    template<typename DType, typename Storage>
    inline static DType vectorCorr(const NDArrayDynamic<DType, Storage>& v1, const NDArrayDynamic<DType, Storage>& v2) {
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

        DType corr = static_cast<DType>(n * sum12 - sum1 * sum2)
                     / static_cast<DType>(sqrt((n * squareSum1 - sum1 * sum1)
                            * (n * squareSum2 - sum2 * sum2)));

        return corr;
    }

    // Correlation coefficient
    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::corrcoef() const {
        auto sh = shape();
        if (sh.size() == 0)
            return NDArrayDynamic<DType, Storage>{};

        if (sh.size() != 1 && sh.size() != 2)
            throw std::runtime_error("Only 1D and 2D arrays supported");

        if (sh.size() == 1)
            return NDArrayDynamic<DType, Storage>{1};

        Shape resultShape({len(), len()});
        NDArrayDynamic<DType, Storage> result{resultShape};
        for (Size i = 0; i < len(); ++i) {
            Shape subResultShape({1, len()});
            NDArrayDynamic<DType, Storage> subResult{subResultShape};
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
    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::std_() const {
        NDArrayDynamic<DType, Storage> x{shape()};
        DType m = mean();
        for (std::size_t i = 0; i < m_ArrayImpl.size(); ++i) {
            auto a = abs(get(i) - m);
            x.set(i, a * a);
        }
        return static_cast<DType>(std::sqrt(x.mean()));
    }
} // np::ndarray::array_dynamic


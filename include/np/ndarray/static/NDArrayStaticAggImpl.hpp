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

namespace np::ndarray::array_static {

    // Array-wise sum
    template<typename DType, Size SizeT, Size... SizeTs>
    inline DType NDArrayStatic<DType, SizeT, SizeTs...>::sum() const {
        DType result{};    
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
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
            const auto& element = at(i);
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
            const auto& element = at(i);
            DType max_ = element.max();
            if (!inited || max_ > result) {
                result = max_;
                inited = true;
            }
        }
        return result;
    }

    // Cumulative sum of the elements
    /*
    template<typename DType, Size SizeT, Size... SizeTs>
    inline auto NDArrayStatic<DType, SizeT, SizeTs...>::cumsum() const {
        NDArrayStatic<DType, (SizeT * ... * SizeTs)> result;
        Size index = 0;
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            //result = result.append(element.cumsum());
        }
        return result;
    }

    // Mean
    template<typename DType, Size SizeT, Size... SizeTs>
    inline DType NDArrayStatic<DType, SizeT, SizeTs...>::mean() const {
        DType result{};
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            result += element.mean();
        }
        if constexpr(SizeT == 0) {
            return 0;
        } else {
            return result / SizeT;
        }
    }

    // Median
    template<typename DType, Size SizeT, Size... SizeTs>
    inline DType NDArrayStatic<DType, SizeT, SizeTs...>::median() const {
        DType result{};
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            result += element.median();
        }
        if constexpr(SizeT == 0) {
            return 0;
        } else {
            return result / SizeT;
        }
    }

    template<typename DType, Size SizeT, Size... SizeTs>
    inline auto NDArrayStatic<DType, SizeT, SizeTs...>::cov() const {
        NDArrayStatic<DType, SizeT, SizeTs...> result{};
        return result;
    }

    // Correlation coefficient
    template<typename DType, Size SizeT, Size... SizeTs>
    inline auto NDArrayStatic<DType, SizeT, SizeTs...>::corrcoef() const {
        NDArrayStatic<DType, SizeT, SizeTs...> result{};
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            result += element.corrcoef();
        }
        if constexpr(SizeT == 0) {
            return 0;
        } else {
            return result / SizeT;
        }
        return result;
    }

    // Standard deviation
    template<typename DType, Size SizeT, Size... SizeTs>
    inline DType NDArrayStatic<DType, SizeT, SizeTs...>::std_() const {
        DType result{};    
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            result += element.std_();
        }
        if constexpr(SizeT == 0) {
            return 0;
        } else {
            return result / SizeT;
        }
        return result;
    }*/
}


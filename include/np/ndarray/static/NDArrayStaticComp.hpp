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
    // Elementwise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<bool_, SizeT, SizeTs...>
    NDArrayStatic<DType, SizeT, SizeTs...>::operator==(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
        NDArrayStatic<bool_, SizeT, SizeTs...> result;
        for (Size i = 0; i < size(); ++i) {
            auto equals = get(i) == array.get(i);
            result.set(i, equals);
        }
        return result;
    }

    // Elementwise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<bool_, SizeT, SizeTs...>
    NDArrayStatic<DType, SizeT, SizeTs...>::operator<(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
        NDArrayStatic<bool_, SizeT, SizeTs...> result;
        for (Size i = 0; i < size(); ++i) {
            auto less = get(i) < array.get(i);
            result.set(i, less);
        }
        return result;
    }

    // Elementwise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<bool_, SizeT, SizeTs...>
    NDArrayStatic<DType, SizeT, SizeTs...>::operator>(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
        NDArrayStatic<bool_, SizeT, SizeTs...> result;
        for (Size i = 0; i < size(); ++i) {
            auto more = get(i) > array.get(i);
            result.set(i, more);
        }
        return result;
    }

    // Array-wise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline bool NDArrayStatic<DType, SizeT, SizeTs...>::array_equal(const DType& element) const {
        if (shape().size() != 1 || shape()[0] != 1)
            return false;
        return element == m_ArrayImpl[0];
    }

    // Array-wise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline bool NDArrayStatic<DType, SizeT, SizeTs...>::array_equal(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
        return m_ArrayImpl == array.m_ArrayImpl;
    }
}

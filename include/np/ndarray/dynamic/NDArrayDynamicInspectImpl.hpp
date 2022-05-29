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

namespace np::ndarray::array_dynamic {

    // Array dimensions
    template<typename DType, typename Storage>
    inline Shape NDArrayDynamic<DType, Storage>::shape() const {
        return m_ArrayImpl.getShape();
    }

    // Array length
    template<typename DType, typename Storage>
    inline Size NDArrayDynamic<DType, Storage>::len() const {
        auto shape = m_ArrayImpl.getShape();
        return shape.empty() ? 0 : shape[0];
    }

    // Number of array dimensions
    template<typename DType, typename Storage>
    Size NDArrayDynamic<DType, Storage>::ndim() const {
        return m_ArrayImpl.getShape().size();
    }

    // Number of array elements
    template<typename DType, typename Storage>
    Size NDArrayDynamic<DType, Storage>::size() const {
        return m_ArrayImpl.size();
    }

    template<typename DType, typename Storage>
    inline constexpr DType NDArrayDynamic<DType, Storage>::dtype() {
        return DType{};
    }

    // Convert an array to a different type
    template<typename DType, typename DTypeNew>
    inline DTypeNew convertValue(const DType& value) {
        return static_cast<DTypeNew>(value);
    }

    template<typename DType, typename Storage>
    template<typename DTypeNew>
    inline NDArrayDynamic<DTypeNew, internal::NDArrayDynamicInternalStorageVector<DTypeNew>> NDArrayDynamic<DType, Storage>::astype() {
        internal::NDArrayDynamicInternal<DTypeNew, internal::NDArrayDynamicInternalStorageVector<DTypeNew>> inter(shape());
        for (std::size_t i = 0; i < m_ArrayImpl.size(); ++i) {
            inter.set(i, convertValue<DType, DTypeNew>(m_ArrayImpl.get(i)));
        }

        NDArrayDynamic<DTypeNew, internal::NDArrayDynamicInternalStorageVector<DTypeNew>> result{inter};

        return result;
    }

}


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
#include <np/ndarray/dynamic/internal/Tools.hpp>

namespace np::ndarray::array_dynamic {
    // Subsetting
    // Select an element at an index
    // a[2]
    template <typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::set(std::size_t i, const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>>& array) {
        m_ArrayImpl[i] = array.m_ArrayImpl;
    }

    template <typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::set(std::size_t i, const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>& array) {
        m_ArrayImpl[i] = array.m_ArrayImpl;
    }

    template <typename DType, typename Storage>
    inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> NDArrayDynamic<DType, Storage>::operator[](std::size_t i) const {
        auto subArray = m_ArrayImpl[i];
        return NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>>{subArray};
    }

    template <typename DType, typename Storage>
    inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> NDArrayDynamic<DType, Storage>::at(std::size_t i) {
        auto subArray = m_ArrayImpl[i];
        return NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>>{subArray};
    }

    template <typename DType, typename Storage>
    inline const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> NDArrayDynamic<DType, Storage>::at(std::size_t i) const {
        const auto subArray = m_ArrayImpl[i];
        return NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>>{subArray};
    }

    // Boolean indexing
    // a[a < 2]                             Select elements from a less than 2
    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::operator[](const std::string &cond) const {
        internal::OperatorWithArg operatorWithArg = internal::getOperatorWithArg(cond);
        auto pred = [&operatorWithArg](DType value) {
            switch (operatorWithArg.first) {
                case internal::Operator::More:
                    return value > operatorWithArg.second;
                case internal::Operator::MoreOrEqual:
                    return value >= operatorWithArg.second;
                case internal::Operator::Equal:
                    return value == operatorWithArg.second;
                case internal::Operator::LessOrEqual:
                    return value <= operatorWithArg.second;
                case internal::Operator::Less:
                    return value < operatorWithArg.second;
                default:
                    std::runtime_error("Invalid operator");
            }
            return false;
        };
        std::vector<DType> result;
        std::copy_if(m_ArrayImpl.m_Impl.begin(),
                     m_ArrayImpl.m_Impl.end(),
                     std::back_inserter(result),
                     pred);

        Shape shape{static_cast<Size>(result.size())};
        return NDArrayDynamic<DType, Storage>{internal::NDArrayDynamicInternal{result, shape}};
    }

    template<typename DType, typename Storage>
    inline DType NDArrayDynamic<DType, Storage>::get(std::size_t i) const {
        return m_ArrayImpl.get(i);
    }

    template<typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::set(std::size_t i, const DType& value) {
        m_ArrayImpl.set(i, value);
    }

} // np::ndarray::array_dynamic


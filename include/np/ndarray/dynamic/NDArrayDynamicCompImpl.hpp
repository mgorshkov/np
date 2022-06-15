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

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // Elementwise comparison
            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator==(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) == array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator==(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) == array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            // Elementwise comparison
            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator<(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) < array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator<(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) < array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            // Elementwise comparison
            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator>(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) > array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
            NDArrayDynamic<DType, Storage>::operator>(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const {
                NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    auto equals = get(i) > array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline bool NDArrayDynamic<DType, Storage>::array_equal(const DType &element) const {
                auto sh = shape();
                return sh.size() == 1 && sh[0] == 1 && m_ArrayImpl.get(0) == element;
            }

            template<typename DType, typename Storage>
            inline bool NDArrayDynamic<DType, Storage>::array_equal(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const {
                return internal::array_equal(m_ArrayImpl, array.m_ArrayImpl);
            }

            template<typename DType, typename Storage>
            inline bool NDArrayDynamic<DType, Storage>::array_equal(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const {
                return internal::array_equal(m_ArrayImpl, array.m_ArrayImpl);
            }

            template<typename DType, typename Storage>
            inline bool NDArrayDynamic<DType, Storage>::array_equal(
                    const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageConstSpan<DType>> &array) const {
                return internal::array_equal(m_ArrayImpl, array.m_ArrayImpl);
            }
        }// namespace array_dynamic
    }// namespace ndarray
}// namespace np
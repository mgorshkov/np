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
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::operator+(const NDArrayDynamic<DType, Storage> &array) const {
                return add(array);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::add(const NDArrayDynamic<DType, Storage> &array) const {
                if (shape() != array.shape())
                    throw std::runtime_error("Shapes are different");
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) + array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::operator-(const NDArrayDynamic<DType, Storage> &array) const {
                return subtract(array);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::subtract(const NDArrayDynamic<DType, Storage> &array) const {
                if (shape() != array.shape())
                    throw std::runtime_error("Shapes are different");
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) - array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::operator*(const NDArrayDynamic<DType, Storage> &array) const {
                return multiply(array);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::multiply(const NDArrayDynamic<DType, Storage> &array) const {
                if (shape() != array.shape())
                    throw std::runtime_error("Shapes are different");
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) * array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::operator/(const NDArrayDynamic<DType, Storage> &array) const {
                return divide(array);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::divide(const NDArrayDynamic<DType, Storage> &array) const {
                if (shape() != array.shape())
                    throw std::runtime_error("Shapes are different");
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) / array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::exp() const {
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, std::exp(get(i)));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::sqrt() const {
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, std::sqrt(get(i)));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::sin() const {
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, std::sin(get(i)));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::cos() const {
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, std::cos(get(i)));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
            NDArrayDynamic<DType, Storage>::log() const {
                NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, std::log(get(i)));
                }
                return result;
            }

            template<typename DType, typename Storage>
            inline DType NDArrayDynamic<DType, Storage>::dot(const NDArrayDynamic<DType, Storage> &array) const {
                if (shape().size() != 1 || array.shape().size() != 1 || shape() != array.shape()) {
                    throw std::runtime_error("Shapes are different or arguments are not 1D arrays");
                }
                DType result{0};
                for (Size i = 0; i < size(); ++i) {
                    result += get(i) * array.get(i);
                }
                return result;
            }
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

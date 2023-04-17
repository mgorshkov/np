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

#include <algorithm>
#include <random>

#include <np/Shape.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // Dynamic array creators
            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic() noexcept
                : NDArrayDynamicBase<DType>{Shape{}} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const Shape &shape, const DType &value)
                : NDArrayDynamicBase<DType>{shape, shape.calcSizeByShape(), value} {
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const CArray1DType<Size1T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const CArray1DType<Size1T> &array,
                                                         bool isColumnVector) noexcept
                : NDArrayDynamicBase<DType>{isColumnVector, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const CArray2DType<Size1T, Size2T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size2T, Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(
                    const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size4T, Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdArray1DType<Size1T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdArray1DType<Size1T> &array,
                                                         bool isColumnVector) noexcept
                : NDArrayDynamicBase<DType>{isColumnVector, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdArray2DType<Size1T, Size2T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size2T, Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(
                    const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : NDArrayDynamicBase<DType>{Shape{Size4T, Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector1DType &vector) noexcept
                : NDArrayDynamicBase<DType>{Shape{static_cast<Size>(vector.size())}, vector} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector1DType &vector, const Shape &shape) noexcept
                : NDArrayDynamicBase<DType>{shape, vector} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector1DType &vector, bool isColumnVector) noexcept
                : NDArrayDynamicBase<DType>{vector, isColumnVector} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(DType *data, Size size) noexcept
                : NDArrayDynamicBase<DType>{data, size} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(DType *data, const Shape &shape) noexcept
                : NDArrayDynamicBase<DType>{data, shape} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector2DType &vector) noexcept
                : NDArrayDynamicBase<DType>{Shape{static_cast<Size>(vector.size())}, vector} {
                if (!vector.empty()) {
                    NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0].size()));
                }
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector3DType &vector) noexcept
                : NDArrayDynamicBase<DType>{Shape{static_cast<Size>(vector.size())}, vector} {
                if (!vector.empty()) {
                    NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0].size()));
                    if (!vector[0].empty()) {
                        NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0][0].size()));
                    }
                }
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(const StdVector4DType &vector) noexcept
                : NDArrayDynamicBase<DType>{Shape{static_cast<Size>(vector.size())}, vector} {
                if (!vector.empty()) {
                    NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0].size()));
                    if (!vector[0].empty()) {
                        NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0][0].size()));
                        if (!vector[0][0].empty()) {
                            NDArrayDynamicBase<DType>::m_shape.addDim(static_cast<Size>(vector[0][0][0].size()));
                        }
                    }
                }
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::NDArrayDynamic(std::initializer_list<DType> init_list) noexcept
                : NDArrayDynamicBase<DType>{Shape{static_cast<Size>(init_list.size())}, init_list} {
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const DType &value) noexcept {
                NDArrayDynamicBase<DType>::operator=(value);
                return *this;
            }

            template<typename DType>
            inline NDArrayDynamic<DType>::~NDArrayDynamic() noexcept = default;

            template<typename DType>
            inline NDArrayDynamic<DType> &
            NDArrayDynamic<DType>::operator=(const NDArrayDynamic<DType> &another) noexcept {
                if (&another != this) {
                    NDArrayDynamicBase<DType>::operator=(another);
                }
                return *this;
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &
            NDArrayDynamic<DType>::operator=(NDArrayDynamic<DType> &&another) noexcept {
                NDArrayDynamicBase<DType>::operator=(another);
                return *this;
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(CArray1DType<Size1T> array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(CArray2DType<Size1T, Size2T> array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdArray1DType<Size1T> &array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
                return *this;
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType> &
            NDArrayDynamic<DType>::operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                NDArrayDynamicBase<DType>::operator=(array);
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdVector1DType &vector) noexcept {
                NDArrayDynamicBase<DType>::operator=(vector);
                return *this;
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdVector2DType &vector) noexcept {
                NDArrayDynamicBase<DType>::operator=(vector);
                return *this;
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdVector3DType &vector) noexcept {
                NDArrayDynamicBase<DType>::operator=(vector);
                return *this;
            }

            template<typename DType>
            inline NDArrayDynamic<DType> &NDArrayDynamic<DType>::operator=(const StdVector4DType &vector) noexcept {
                NDArrayDynamicBase<DType>::operator=(vector);
                return *this;
            }
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

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
#include <random>

#include <np/Shape.hpp>

#include <np/ndarray/dynamic/NDArrayDynamicDecl.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // Dynamic array creators
            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic() noexcept
                : m_ArrayImpl{} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const DType &value) noexcept
                : m_ArrayImpl{value} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(Shape shape) noexcept
                : m_ArrayImpl{std::move(shape)} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const CArray1DType<Size1T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const CArray1DType<Size1T> &array,
                                                                  bool isColumnVector) noexcept
                : m_ArrayImpl{array, isColumnVector} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const CArray2DType<Size1T, Size2T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(
                    const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdArray1DType<Size1T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdArray1DType<Size1T> &array,
                                                                  bool isColumnVector) noexcept
                : m_ArrayImpl{array, isColumnVector} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdArray2DType<Size1T, Size2T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(
                    const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(
                    const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector1DType &vector) noexcept
                : m_ArrayImpl{vector} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector1DType &vector, bool isColumnVector) noexcept
                : m_ArrayImpl{vector, isColumnVector} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector2DType &vector) noexcept
                : m_ArrayImpl{vector} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector3DType &vector) noexcept
                : m_ArrayImpl{vector} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector4DType &vector) noexcept
                : m_ArrayImpl{vector} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const StdVector1DType &vector, Shape shape) noexcept
                : m_ArrayImpl{vector, shape} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(const NDArrayDynamic<DType, Storage> &another) noexcept
                : m_ArrayImpl{another.m_ArrayImpl} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(NDArrayDynamic<DType, Storage> &&another) noexcept
                : m_ArrayImpl{std::move(another.m_ArrayImpl)} {
            }

            template<typename DType, typename Storage>
            template<typename InternalStorage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(
                    const internal::NDArrayDynamicInternal<DType, InternalStorage> &array) noexcept
                : m_ArrayImpl{array} {
            }

            template<typename DType, typename Storage>
            template<typename InternalStorage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(
                    internal::NDArrayDynamicInternal<DType, InternalStorage> &&array) noexcept
                : m_ArrayImpl{std::move(array)} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::NDArrayDynamic(std::initializer_list<DType> init_list) noexcept
                : m_ArrayImpl{init_list} {
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>::~NDArrayDynamic() noexcept = default;

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage> &
            NDArrayDynamic<DType, Storage>::operator=(const NDArrayDynamic<DType, Storage> &another) noexcept {
                m_ArrayImpl = another.m_ArrayImpl;
                return *this;
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage> &
            NDArrayDynamic<DType, Storage>::operator=(NDArrayDynamic<DType, Storage> &&another) noexcept {
                m_ArrayImpl = std::move(another.m_ArrayImpl);
                return *this;
            }
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

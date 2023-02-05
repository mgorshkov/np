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

#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            // Array creators
            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic() noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{SizeT}} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(Shape shape) noexcept
                : NDArrayStaticBase<DType, SizeT>{std::move(shape)} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const DType &value) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{SizeT}, value} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(Shape shape, const DType &value) noexcept
                : NDArrayStaticBase<DType, SizeT>{std::move(shape), value} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const CArray1DType<Size1T> &data) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size1T}, data} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept
                : NDArrayStaticBase<DType, SizeT>{isColumnVector, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const CArray2DType<Size1T, Size2T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size4T, Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(
                    const NDArrayStatic<DType, SizeT> &another) noexcept
                : NDArrayStaticBase<DType, SizeT>{another} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(
                    NDArrayStatic<DType, SizeT> &&another) noexcept
                : NDArrayStaticBase<DType, SizeT>{another} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdArray1DType<Size1T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdArray1DType<Size1T> &array, bool isColumnVector) noexcept
                : NDArrayStaticBase<DType, SizeT>{isColumnVector, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdArray2DType<Size1T, Size2T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{Size4T, Size3T, Size2T, Size1T}, array} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector1DType &vector) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{vector.size()}, vector} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector1DType &vector, const Shape &shape) noexcept
                : NDArrayStaticBase<DType, SizeT>{shape, vector} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector1DType &vector, bool isColumnVector) noexcept
                : NDArrayStaticBase<DType, SizeT>{isColumnVector, vector} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector2DType &vector) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{vector.size()}, vector} {
                if (!vector.empty()) {
                    NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0].size());
                }
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector3DType &vector) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{vector.size()}, vector} {
                if (!vector.empty()) {
                    NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0].size());
                    if (!vector[0].empty()) {
                        NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0][0].size());
                    }
                }
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(const StdVector4DType &vector) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{vector.size()}, vector} {
                if (!vector.empty()) {
                    NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0].size());
                    if (!vector[0].empty()) {
                        NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0][0].size());
                        if (!vector[0][0].empty()) {
                            NDArrayStaticBase<DType, SizeT>::m_shape.addDim(vector[0][0][0].size());
                        }
                    }
                }
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT>::NDArrayStatic(std::initializer_list<DType> init_list) noexcept
                : NDArrayStaticBase<DType, SizeT>{Shape{init_list.size()}, init_list} {
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT> &NDArrayStatic<DType, SizeT>::operator=(const DType &another) noexcept {
                if (this != &another) {
                    NDArrayStaticBase<DType, SizeT>::operator=(another);
                }
                return *this;
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT> &NDArrayStatic<DType, SizeT>::operator=(
                    const NDArrayStatic<DType, SizeT> &another) noexcept {
                if (this != &another) {
                    NDArrayStaticBase<DType, SizeT>::operator=(another);
                }
                return *this;
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(NDArrayStatic<DType, SizeT> &&another) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(another);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(CArray1DType<Size1T> array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(CArray2DType<Size1T, Size2T> array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(const StdArray1DType<Size1T> &array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::
            operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(array);
                return *this;
            }

            template<typename DType, Size SizeT>
            inline NDArrayStatic<DType, SizeT> &
            NDArrayStatic<DType, SizeT>::operator=(const NDArrayStatic<DType, SizeT>::StdVector1DType &vector) noexcept {
                NDArrayStaticBase<DType, SizeT>::operator=(vector);
                return *this;
            }
        }// namespace array_static
    }    // namespace ndarray
}// namespace np

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

#include <array>
#include <cstddef>
#include <optional>
#include <ostream>
#include <vector>

#include <np/Axis.hpp>
#include <np/DType.hpp>
#include <np/Shape.hpp>

#include <np/internal/Tools.hpp>
#include <np/ndarray/static/internal/NDArrayStaticStorage.hpp>
#include <np/ndarray/static/internal/NDArrayStaticStorageStreamIo.hpp>
#include <np/ndarray/static/internal/Using.hpp>

#include <np/ndarray/internal/NDArrayShaped.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            template<typename DType, Size SizeT>
            class NDArrayStatic;

            template<typename DType, Size SizeT>
            using NDArrayStaticStorage = internal::NDArrayStaticStorage<DType, SizeT>;

            template<typename DType, Size SizeT>
            using NDArrayStaticBase = ndarray::internal::NDArrayShaped<DType, NDArrayStatic<DType, SizeT>, NDArrayStaticStorage<DType, SizeT>>;

            template<typename DType, Size SizeT>
            class NDArrayStatic final : public NDArrayStaticBase<DType, SizeT> {
            public:
                template<std::size_t Size1T>
                using CArray1DType = DType[Size1T];
                template<std::size_t Size1T, std::size_t Size2T>
                using CArray2DType = CArray1DType<Size1T>[Size2T];
                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                using CArray3DType = CArray2DType<Size1T, Size2T>[Size3T];
                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                using CArray4DType = CArray3DType<Size1T, Size2T, Size3T>[Size4T];

                template<std::size_t Size1T>
                using StdArray1DType = std::array<DType, Size1T>;
                template<std::size_t Size1T, std::size_t Size2T>
                using StdArray2DType = std::array<StdArray1DType<Size1T>, Size2T>;
                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                using StdArray3DType = std::array<StdArray2DType<Size1T, Size2T>, Size3T>;
                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                using StdArray4DType = std::array<StdArray3DType<Size1T, Size2T, Size3T>, Size4T>;

                using StdVector1DType = std::vector<DType>;
                using StdVector2DType = std::vector<StdVector1DType>;
                using StdVector3DType = std::vector<StdVector2DType>;
                using StdVector4DType = std::vector<StdVector3DType>;

                // Creating arrays
                inline NDArrayStatic() noexcept;

                inline explicit NDArrayStatic(const DType &value) noexcept;

                inline explicit NDArrayStatic(Shape shape) noexcept;

                inline explicit NDArrayStatic(Shape shape, const DType &value) noexcept;

                inline NDArrayStatic(const NDArrayStatic &another) noexcept;

                inline NDArrayStatic(NDArrayStatic &&another) noexcept;

                template<std::size_t Size1T>
                inline explicit NDArrayStatic(const CArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayStatic(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline explicit NDArrayStatic(const CArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline explicit NDArrayStatic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline explicit NDArrayStatic(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                template<std::size_t Size1T>
                inline explicit NDArrayStatic(const StdArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T>
                inline explicit NDArrayStatic(const StdArray1DType<Size1T> &array, bool isColumnVector) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline explicit NDArrayStatic(const StdArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline explicit NDArrayStatic(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline explicit NDArrayStatic(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                inline explicit NDArrayStatic(const StdVector1DType &vector) noexcept;

                inline explicit NDArrayStatic(const StdVector1DType &vector, const Shape &shape) noexcept;

                inline NDArrayStatic(const StdVector1DType &vector, bool isColumnVector) noexcept;

                inline explicit NDArrayStatic(const StdVector2DType &vector) noexcept;

                inline explicit NDArrayStatic(const StdVector3DType &vector) noexcept;

                inline explicit NDArrayStatic(const StdVector4DType &vector) noexcept;

                inline NDArrayStatic(std::initializer_list<DType> init_list) noexcept;

                inline ~NDArrayStatic() noexcept = default;

                inline NDArrayStatic &operator=(const DType &value) noexcept;

                inline NDArrayStatic &operator=(const NDArrayStatic &another) noexcept;

                inline NDArrayStatic &operator=(NDArrayStatic &&another) noexcept;

                template<std::size_t Size1T>
                inline NDArrayStatic &operator=(CArray1DType<Size1T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline NDArrayStatic &operator=(CArray2DType<Size1T, Size2T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline NDArrayStatic &operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline NDArrayStatic &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayStatic &operator=(const StdArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline NDArrayStatic &operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline NDArrayStatic &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline NDArrayStatic &
                operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                inline NDArrayStatic &operator=(const StdVector1DType &vector) noexcept;

                inline NDArrayStatic &operator=(const StdVector2DType &vector) noexcept;

                inline NDArrayStatic &operator=(const StdVector3DType &vector) noexcept;

                inline NDArrayStatic &operator=(const StdVector4DType &vector) noexcept;

                NDArrayStatic<bool_, SizeT> operator==(const NDArrayStatic &array) const;
                NDArrayStatic<bool_, SizeT> operator<(const NDArrayStatic &array) const;
                NDArrayStatic<bool_, SizeT> operator>(const NDArrayStatic &array) const;

                static constexpr int kDepth = 0;

            private:
                using Base = ndarray::internal::NDArrayBase<DType, NDArrayStatic<DType, SizeT>, NDArrayStaticStorage<DType, SizeT>>;
                using BasePtr = ndarray::internal::NDArrayBasePtr<DType, NDArrayStatic<DType, SizeT>, NDArrayStaticStorage<DType, SizeT>>;
                using BaseBoolPtr = ndarray::internal::NDArrayBasePtr<bool_, NDArrayStatic<bool_, SizeT>, NDArrayStaticStorage<bool_, SizeT>>;
                using BaseConstPtr = ndarray::internal::NDArrayBaseConstPtr<DType, NDArrayStatic<DType, SizeT>, NDArrayStaticStorage<DType, SizeT>>;
            };
        }// namespace array_static
    }    // namespace ndarray
}// namespace np
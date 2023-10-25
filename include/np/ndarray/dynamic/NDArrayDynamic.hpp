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

#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>

#include <np/Axis.hpp>
#include <np/Shape.hpp>

#include <np/ndarray/internal/NDArrayIndex.hpp>
#include <np/ndarray/internal/NDArrayShaped.hpp>

#include <np/ndarray/dynamic/internal/NDArrayDynamicStorage.hpp>
#include <np/ndarray/dynamic/internal/NDArrayDynamicStorageStreamIo.hpp>
#include <np/ndarray/dynamic/internal/Using.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // N-dimensional dynamic array
            template<typename DType>
            class NDArrayDynamic;

            template<typename DType>
            using NDArrayDynamicStorage = internal::NDArrayDynamicStorage<DType>;

            template<typename DType>
            using NDArrayDynamicBase = ndarray::internal::NDArrayShaped<DType, NDArrayDynamic<DType>, NDArrayDynamicStorage<DType>>;

            template<typename DType>
            using Base = ndarray::internal::NDArrayBase<DType, NDArrayDynamic<DType>, NDArrayDynamicStorage<DType>>;

            template<typename DType>
            using BasePtr = ndarray::internal::NDArrayBasePtr<DType, NDArrayDynamic<DType>, NDArrayDynamicStorage<DType>>;

            template<typename DType>
            using BaseConstPtr = ndarray::internal::NDArrayBaseConstPtr<DType, NDArrayDynamic<DType>, NDArrayDynamicStorage<DType>>;

            using BaseBoolPtr = ndarray::internal::NDArrayBasePtr<bool_, NDArrayDynamic<bool_>, NDArrayDynamicStorage<bool_>>;

            template<typename DType>
            class NDArrayDynamic final : public NDArrayDynamicBase<DType> {
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

                // Creating Arrays
                inline NDArrayDynamic() noexcept;

                inline explicit NDArrayDynamic(const Shape &shape, const DType &value = DType());

                NDArrayDynamic(const NDArrayDynamic &another) noexcept = default;

                NDArrayDynamic(NDArrayDynamic &&another) noexcept = default;

                template<std::size_t Size1T>
                inline explicit NDArrayDynamic(const CArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayDynamic(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline explicit NDArrayDynamic(const CArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline explicit NDArrayDynamic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline explicit NDArrayDynamic(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                template<std::size_t Size1T>
                inline explicit NDArrayDynamic(const StdArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T>
                inline explicit NDArrayDynamic(const StdArray1DType<Size1T> &array, bool isColumnVector) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline explicit NDArrayDynamic(const StdArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline explicit NDArrayDynamic(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline explicit NDArrayDynamic(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                inline explicit NDArrayDynamic(const StdVector1DType &vector) noexcept;

                inline explicit NDArrayDynamic(const StdVector1DType &vector, const Shape &shape) noexcept;

                inline NDArrayDynamic(DType *data, Size size) noexcept;
                inline NDArrayDynamic(DType *data, const Shape &shape) noexcept;

                inline NDArrayDynamic(const StdVector1DType &vector, bool isColumnVector) noexcept;

                inline explicit NDArrayDynamic(const StdVector2DType &vector) noexcept;

                inline explicit NDArrayDynamic(const StdVector3DType &vector) noexcept;

                inline explicit NDArrayDynamic(const StdVector4DType &vector) noexcept;

                inline NDArrayDynamic(std::initializer_list<DType> init_list) noexcept;

                inline ~NDArrayDynamic() noexcept;

                inline NDArrayDynamic &operator=(const DType &value) noexcept;

                inline NDArrayDynamic &operator=(const NDArrayDynamic &another) noexcept;

                inline NDArrayDynamic &operator=(NDArrayDynamic &&another) noexcept;

                template<std::size_t Size1T>
                inline NDArrayDynamic &operator=(CArray1DType<Size1T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline NDArrayDynamic &operator=(CArray2DType<Size1T, Size2T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline NDArrayDynamic &operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline NDArrayDynamic &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayDynamic &operator=(const StdArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline NDArrayDynamic &operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline NDArrayDynamic &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline NDArrayDynamic &
                operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

                inline NDArrayDynamic &operator=(const StdVector1DType &vector) noexcept;

                inline NDArrayDynamic &operator=(const StdVector2DType &vector) noexcept;

                inline NDArrayDynamic &operator=(const StdVector3DType &vector) noexcept;

                inline NDArrayDynamic &operator=(const StdVector4DType &vector) noexcept;

                inline bool operator!=(const DType &value) const;

                NDArrayDynamic<bool_> operator==(const NDArrayDynamic &array) const;
                NDArrayDynamic<bool_> operator<(const NDArrayDynamic &array) const;
                NDArrayDynamic<bool_> operator>(const NDArrayDynamic &array) const;

                template<typename DTypeNew>
                inline NDArrayDynamic<DTypeNew> astype() {
                    NDArrayDynamic<DTypeNew> result{NDArrayDynamicBase<DType>::shape()};
                    for (Size i = 0; i < result.size(); ++i) {
                        result.set(i, static_cast<DTypeNew>(NDArrayDynamicBase<DType>::get(i)));
                    }
                    return result;
                }
            };

            using NDArrayDynamicBool = NDArrayDynamic<bool_>;

            struct NDArrayDynamicHasher {
                template<typename DType>
                auto operator()(const NDArrayDynamic<DType> &array) const -> std::size_t {
                    std::size_t h{0};
                    for (const auto &element: array) {
                        h ^= std::hash<DType>{}(element);
                    }
                    return h;
                }
            };

            template<typename DType, typename ValueType, typename Hasher = ndarray::internal::NDArrayBaseHasher, typename EqualTo = ndarray::internal::NDArrayBaseEqualTo>
            using NDArrayDynamicMap = std::unordered_map<NDArrayDynamic<DType>, ValueType, Hasher, EqualTo>;

            template<typename DType>
            using NDArrayDynamicIndexKeyType = ndarray::internal::IndexParent<DType, NDArrayDynamic<DType>, internal::NDArrayDynamicStorage<DType>, BasePtr<DType>>;

            template<typename DType>
            using NDArrayDynamicIndexConstKeyType = ndarray::internal::IndexParent<DType, NDArrayDynamic<DType>, internal::NDArrayDynamicStorage<DType>, BaseConstPtr<DType>>;

            template<typename DType, typename ValueType, typename Hasher = ndarray::internal::NDArrayBaseHasher, typename EqualTo = ndarray::internal::NDArrayBaseEqualTo>
            using NDArrayDynamicIndexMap = std::unordered_map<NDArrayDynamicIndexKeyType<DType>, ValueType, Hasher, EqualTo>;

            template<typename DType, typename ValueType, typename Hasher = ndarray::internal::NDArrayBaseHasher, typename EqualTo = ndarray::internal::NDArrayBaseEqualTo>
            using NDArrayDynamicIndexConstMap = std::unordered_map<NDArrayDynamicIndexConstKeyType<DType>, ValueType, Hasher, EqualTo>;

        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

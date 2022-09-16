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

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>

#include <np/Axis.hpp>
#include <np/Shape.hpp>

#include <np/ndarray/dynamic/internal/NDArrayDynamicInternal.hpp>
#include <np/ndarray/internal/Indexing.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // N-dimensional dynamic array
            template<typename DType, typename Storage = internal::NDArrayDynamicInternalStorageVector<DType>>
            class NDArrayDynamic {
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

                inline explicit NDArrayDynamic(const DType &value) noexcept;

                inline explicit NDArrayDynamic(Shape shape) noexcept;

                inline NDArrayDynamic(const NDArrayDynamic &another) noexcept;

                template<typename StorageOther>
                inline NDArrayDynamic(const NDArrayDynamic<DType, StorageOther> &another) noexcept;

                inline NDArrayDynamic(NDArrayDynamic &&another) noexcept;

                template<typename StorageOther>
                inline NDArrayDynamic(NDArrayDynamic<DType, StorageOther> &&another) noexcept;

                template<typename InternalStorage>
                inline explicit NDArrayDynamic(const internal::NDArrayDynamicInternal<DType, InternalStorage> &array) noexcept;

                template<typename InternalStorage>
                inline explicit NDArrayDynamic(internal::NDArrayDynamicInternal<DType, InternalStorage> &&array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayDynamic(const CArray1DType<Size1T> &array) noexcept;

                template<std::size_t Size1T>
                inline NDArrayDynamic(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept;

                template<std::size_t Size1T, std::size_t Size2T>
                inline NDArrayDynamic(const CArray2DType<Size1T, Size2T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                inline NDArrayDynamic(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept;

                template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                inline NDArrayDynamic(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept;

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

                inline explicit NDArrayDynamic(const StdVector1DType &vector, Shape shape) noexcept;

                inline NDArrayDynamic(const StdVector1DType &vector, bool isColumnVector) noexcept;

                inline explicit NDArrayDynamic(const StdVector2DType &vector) noexcept;

                inline explicit NDArrayDynamic(const StdVector3DType &vector) noexcept;

                inline explicit NDArrayDynamic(const StdVector4DType &vector) noexcept;

                inline NDArrayDynamic(std::initializer_list<DType> init_list) noexcept;

                inline ~NDArrayDynamic() noexcept;

                inline NDArrayDynamic &operator=(const NDArrayDynamic &another) noexcept;

                inline NDArrayDynamic &operator=(NDArrayDynamic &&another) noexcept;

                // Indexing arrays
                inline void set(std::size_t i,
                                const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array);

                inline void set(std::size_t i,
                                const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array);

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageConstSpan<DType>>
                operator[](std::size_t i) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> at(std::size_t i);

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageConstSpan<DType>>
                at(std::size_t i) const;

                inline NDArrayDynamic<DType> operator[](const std::string &cond) const;

                inline typename Storage::const_reference get(std::size_t i) const;

                inline typename Storage::reference get(std::size_t i);

                inline void set(std::size_t i, typename Storage::value_type value);

                inline friend std::ostream &operator<<(std::ostream &stream, const NDArrayDynamic &array) {
                    return stream << array.m_ArrayImpl;
                }

                // Save and load
                inline void save(const char *filename);

                inline void savez(const char *filename);

                inline static NDArrayDynamic load(const char *filename);

                // Saving & Loading Text Files
                inline static NDArrayDynamic loadtxt(const char *filename);

                inline NDArrayDynamic genfromtxt(const char *filename);

                inline void savetxt(const char *filename);

                // Array dimensions
                inline Shape shape() const;

                // Array length
                inline Size len() const;

                // Number of array dimensions
                inline Size ndim() const;

                // Number of array elements
                inline Size size() const;

                inline constexpr DType dtype();

                template<typename DTypeNew>
                inline NDArrayDynamic<DTypeNew, internal::NDArrayDynamicInternalStorageVector<DTypeNew>> astype();

                // Array mathematics
                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                operator+(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                add(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                operator-(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                subtract(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                operator*(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                multiply(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                operator/(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>>
                divide(const NDArrayDynamic &array) const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> exp() const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> sqrt() const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> sin() const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> cos() const;

                inline NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> log() const;

                // Dot product
                inline DType dot(const NDArrayDynamic &array) const;

                // Elementwise comparison
                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator==(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const;

                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator<(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const;

                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator>(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const;

                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator==(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const;

                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator<(const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const;

                inline NDArrayDynamic<bool_, internal::NDArrayDynamicInternalStorageVector<bool_>>
                operator>(const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const;

                // Array-wise comparison
                inline bool array_equal(const DType &value) const;

                inline bool array_equal(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageVector<DType>> &array) const;

                inline bool array_equal(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageSpan<DType>> &array) const;

                inline bool array_equal(
                        const NDArrayDynamic<DType, internal::NDArrayDynamicInternalStorageConstSpan<DType>> &array) const;

                // Aggregate functions
                // Array-wise sum
                inline DType sum() const;

                // Array-wise minimum value
                inline DType min() const;

                // Maximum value of an Array row
                inline DType max() const;

                // Cumulative sum of the elements
                inline NDArrayDynamic<DType, Storage> cumsum() const;

                // Mean
                inline DType mean() const;

                // Median
                inline DType median() const;

                // Covariance
                inline NDArrayDynamic<DType, Storage> cov() const;

                // Correlation coefficient
                inline NDArrayDynamic<DType, Storage> corrcoef() const;

                // Compute the standard deviation along the specified axis.
                inline DType std_() const;

                // Create a view of the array with the same data
                inline NDArrayDynamic<DType, Storage> view() const;

                // Create a deep copy of the array
                inline NDArrayDynamic<DType, Storage> copy() const;

                // Sort an array
                inline void sort();

                // template<std::size_t N = -1>
                // inline void sort(std::optional<Axis<N>> axis=std::optional<Axis<N>>{});

                // Permute array dimensions
                inline NDArrayDynamic<DType> transpose() const;

                // Flatten the array
                inline NDArrayDynamic<DType, Storage> ravel() const;

                // Reshape, but don’t change data
                inline NDArrayDynamic<DType, Storage> reshape(const Shape &shape) const;

                // Adding and removing elements
                // Return a new array with the given shape
                inline NDArrayDynamic<DType, Storage> resize(const Shape &shape) const;

                // Append items to the array
                inline NDArrayDynamic<DType, Storage> append(const NDArrayDynamic &array) const;

                // Insert items in the array
                inline NDArrayDynamic<DType, Storage> insert(Size index, const NDArrayDynamic &array) const;

                // Delete items from the array
                inline NDArrayDynamic<DType, Storage> del(Size index) const;

                // Concatenate arrays
                inline NDArrayDynamic<DType, Storage> concatenate(const NDArrayDynamic &array, std::optional<std::size_t> axis = std::nullopt) const;

                // Stack arrays vertically (row-wise)
                inline NDArrayDynamic<DType, Storage> vstack(const NDArrayDynamic &array) const;

                // Stack arrays vertically (row-wise)
                inline NDArrayDynamic<DType, Storage> r_(const NDArrayDynamic &array) const;

                // Stack arrays horizontally (column-wise)
                inline NDArrayDynamic<DType, Storage> hstack(const NDArrayDynamic &array) const;

                // Create stacked column-wise arrays
                inline NDArrayDynamic<DType, Storage> column_stack(const NDArrayDynamic &array) const;

                // Create stacked column-wise arrays
                inline NDArrayDynamic<DType, Storage> c_(const NDArrayDynamic &array) const;

                // Split the array horizontally
                inline std::vector<NDArrayDynamic<DType, Storage>> hsplit(std::size_t sections) const;

                // Split the array vertically
                inline std::vector<NDArrayDynamic<DType, Storage>> vsplit(std::size_t sections) const;

                typename internal::NDArrayDynamicInternal<DType, Storage>::iterator begin() {
                    return m_ArrayImpl.begin();
                }

                typename internal::NDArrayDynamicInternal<DType, Storage>::iterator end() {
                    return m_ArrayImpl.end();
                }

                typename internal::NDArrayDynamicInternal<DType, Storage>::const_iterator cbegin() const {
                    return m_ArrayImpl.cbegin();
                }

                typename internal::NDArrayDynamicInternal<DType, Storage>::const_iterator cend() const {
                    return m_ArrayImpl.cend();
                }

            private:
                inline void save(std::ostream &stream);

                inline static NDArrayDynamic load(std::istream &stream);

                std::size_t getMaxElementSize() const {
                    std::size_t size = sizeof(DType);
                    if constexpr (std::is_same<string_, DType>::value || std::is_same<unicode_, DType>::value) {
                        size = 1;
                        for (auto it = m_ArrayImpl.cbegin(); it != m_ArrayImpl.cend(); ++it) {
                            if ((*it).size() > size) {
                                size = (*it).size();
                            }
                        }
                    }
                    return size;
                }

                inline NDArrayDynamic<DType> booleanIndexing(const std::string &cond) const;
                inline NDArrayDynamic<DType> slicing(const std::string &cond) const;

                internal::NDArrayDynamicInternal<DType, Storage> m_ArrayImpl;

                static constexpr long kIndexingHandlersSize{static_cast<long>(ndarray::internal::IndexingMode::None)};

                const ndarray::internal::IndexingHandler<NDArrayDynamic<DType>> m_IndexingHandlers[kIndexingHandlersSize] = {
                        {ndarray::internal::IndexingMode::Slicing,
                         ndarray::internal::isSlicing,
                         std::bind(&NDArrayDynamic::slicing, this, std::placeholders::_1)},
                        {ndarray::internal::IndexingMode::BooleanIndexing,
                         ndarray::internal::isBooleanIndexing<DType>,
                         std::bind(&NDArrayDynamic::booleanIndexing, this, std::placeholders::_1)}};

                template<typename DTypeOther, typename StorageOther>
                friend class NDArrayDynamic;
            };
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

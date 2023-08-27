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
#include <memory>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

#include <np/Axis.hpp>
#include <np/Shape.hpp>
#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/internal/IndexStorage.hpp>
#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage>
            class NDArrayBase;

            template<typename DType, typename Derived, typename Storage>
            using NDArrayBasePtr = NDArrayBase<DType, Derived, Storage> *;

            template<typename Derived, typename Storage>
            using NDArrayBaseBoolPtr = NDArrayBase<bool_, Derived, Storage> *;

            template<typename DType, typename Derived, typename Storage>
            using NDArrayBaseConstPtr = const NDArrayBase<DType, Derived, Storage> *;

            template<typename DType, typename Derived, typename Storage, typename ParentStorage, typename Parent>
            class Index;

            template<typename DType, typename Storage, typename Parent>
            class IndexStorage;

            template<typename DType, typename Derived, typename ParentStorage, typename Parent>
            using IndexParent = Index<DType, Derived, IndexStorage<DType, ParentStorage, Parent>, ParentStorage, Parent>;

            // N-dimensional array
            template<typename DType, typename Derived, typename Storage>
            class NDArrayBase {
            public:
                template<typename... Args>
                explicit NDArrayBase(Args &&...args);

                NDArrayBase(const NDArrayBase &another);

                NDArrayBase(NDArrayBase &&another) noexcept;

                virtual ~NDArrayBase() = default;

                NDArrayBase &operator=(const NDArrayBase &another);

                NDArrayBase &operator=(NDArrayBase &&another) noexcept;

                // Save and load
                void save(const char *filename);

                void savez(const char *filename);

                auto load(const char *filename);

                // Saving & Loading Text Files
                auto loadtxt(const char *filename);

                auto genfromtxt(const char *filename);

                void savetxt(const char *filename, const char *delimiter);

                // Array dimensions
                [[nodiscard]] virtual Shape shape() const { return m_storage.shape(); };
                virtual void setShape(const Shape &shape) { m_storage.setShape(shape); };

                [[nodiscard]] bool empty() const {
                    return shape().empty();
                }

                // Array length
                [[nodiscard]] Size len() const {
                    return shape().empty() ? 0 : shape()[0];
                }

                // Number of array dimensions
                [[nodiscard]] Size ndim() const {
                    return static_cast<Size>(shape().size());
                }

                // Number of array elements
                [[nodiscard]] Size size() const {
                    return shape().calcSizeByShape();
                }

                inline constexpr DType dtype();

                // Array mathematics
                template<Arithmetic DType2, typename Derived2, typename Storage2>
                auto add(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<Arithmetic DType2>
                auto add(const DType2 &value) const;

                template<Arithmetic DType2, typename Derived2, typename Storage2>
                auto subtract(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<Arithmetic DType2>
                auto subtract(const DType2 &value) const;

                template<Arithmetic DType2, typename Derived2, typename Storage2>
                auto multiply(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<Arithmetic DType2>
                auto multiply(const DType2 &value) const;

                template<Arithmetic DType2, typename Derived2, typename Storage2>
                auto divide(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<Arithmetic DType2>
                auto divide(const DType2 &value) const;

                auto exp() const;

                auto sqrt() const;

                auto sin() const;

                auto cos() const;

                auto log() const;

                auto abs() const;

                // Dot product
                template<typename DType2, typename Derived2, typename Storage2>
                auto dot(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Elementwise comparison
                template<typename DType2, typename Derived2, typename Storage2>
                auto operator==(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<typename DType2, typename Derived2, typename Storage2>
                auto operator<(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                template<typename DType2, typename Derived2, typename Storage2>
                auto operator>(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Array-wise comparison
                bool array_equal(const DType &element) const {
                    auto sh = shape();
                    return sh.size() == 1 && sh[0] == 1 && get(0) == element;
                }

                template<typename DType2, typename Derived2, typename Storage2>
                bool array_equal(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                    auto shape1 = shape();
                    auto shape2 = array.shape();
                    if (shape1 != shape2)
                        return false;
                    for (Size index = 0; index < size(); ++index) {
                        if (!np::internal::element_equal(get(index), array.get(index)))
                            return false;
                    }
                    return true;
                }

                // Aggregate functions
                // Array-wise sum
                DType sum() const;

                // Array-wise sum treating NaNs as zeros
                DType nansum() const;

                // Array-wise minimum value
                DType min() const;

                // Maximum value of an Array row
                DType max() const;

                // Cumulative sum of the elements
                auto cumsum() const;

                // Cumulative sum of the elements treating NaNs as zeros
                auto nancumsum() const;

                // Mean
                [[nodiscard]] float_ mean() const;

                // Mean excluding NaNs
                [[nodiscard]] float_ nanmean() const;

                // Median
                [[nodiscard]] float_ median() const;

                // Median excluding NaNs
                [[nodiscard]] float_ nanmedian() const;

                // Covariance
                auto cov() const;

                // Correlation coefficient
                auto corrcoef() const;

                // Compute the standard deviation along the specified axis.
                [[nodiscard]] float_ std_() const;

                // Compute the standard deviation along the specified axis excluding NaNs.
                [[nodiscard]] float_ nanstd() const;

                // Compute the standard variance along the specified axis.
                [[nodiscard]] float_ var() const;

                // Compute the standard variance along the specified axis excluding NaNs.
                [[nodiscard]] float_ nanvar() const;

                // Create a view of the array with the same data
                auto view() const;

                // Create a deep copy of the array
                auto copy() const;

                // Sort an array
                void sort();

                // template<std::size_t N = -1>
                // inline void sort(std::optional<Axis<N>> axis=std::optional<Axis<N>>{});

                // Permute array dimensions
                auto transpose() const;

                // Flatten the array
                auto ravel() const;

                // Reshape, but don’t change data
                auto reshape(const Shape &shape) const;

                // Adding and removing elements
                // Return a new array with the given shape
                auto resize(const Shape &shape) const;

                // Append items to the array
                template<typename DType2, typename Derived2, typename Storage2>
                auto append(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Insert items in the array
                template<typename DType2, typename Derived2, typename Storage2>
                auto insert(Size index, const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Delete items from the array
                auto del(Size index) const;

                // Concatenate arrays
                template<typename DType2, typename Derived2, typename Storage2>
                auto concatenate(const NDArrayBase<DType2, Derived2, Storage2> &array, std::optional<std::size_t> axis = std::nullopt) const;

                // Stack arrays vertically (row-wise)
                template<typename DType2, typename Derived2, typename Storage2>
                auto vstack(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Stack arrays vertically (row-wise)
                template<typename DType2, typename Derived2, typename Storage2>
                auto r_(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Stack arrays horizontally (column-wise)
                template<typename DType2, typename Derived2, typename Storage2>
                auto hstack(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Create stacked column-wise arrays
                template<typename DType2, typename Derived2, typename Storage2>
                auto c_(const NDArrayBase<DType2, Derived2, Storage2> &array) const;

                // Split the array horizontally
                auto hsplit(std::size_t sections) const;

                // Split the array vertically
                auto vsplit(std::size_t sections) const;

                // Expand the shape of an array.
                auto expand_dims(Size axis) const;

                const DType &get(Size i) const {
                    return m_storage.get(i);
                }

                DType &get(Size i) {
                    return m_storage.get(i);
                }

                void set(Size i, const DType &value) {
                    m_storage.set(i, value);
                }

                Storage &getStorage() {
                    return m_storage;
                }

                const Storage &getStorage() const {
                    return m_storage;
                }

                [[nodiscard]] Size index(Size i) const {
                    return m_storage.index(i);
                }

                auto begin() {
                    return m_storage.begin();
                }

                auto end() {
                    return m_storage.end();
                }

                auto cbegin() const {
                    return m_storage.cbegin();
                }

                auto cend() const {
                    return m_storage.cend();
                }

                void push(const DType &value) {
                    m_storage.push_back(value);
                }

                // Indexing arrays
                using IndexParentType = IndexParent<DType, Derived, Storage, NDArrayBasePtr<DType, Derived, Storage>>;
                using IndexParentConstType = IndexParent<DType, Derived, Storage, NDArrayBaseConstPtr<DType, Derived, Storage>>;

                IndexParentConstType operator[](SignedSize i) const;
                IndexParentType operator[](SignedSize i);

                // Subsetting
                // a[2] Select the element at the 2nd index
                // b[1,2] Select the element at row 1 column 2 (equivalent to b[1][2])
                // Boolean indexing
                // a[a < 2] Select elements from a less than 2
                // Slicing
                // a[0:2] Select items at index 0 and 1
                // b[0:2,1] Select items at rows 0 and 1 in column 1
                IndexParentConstType operator[](const std::string &cond) const;
                IndexParentType operator[](const std::string &cond);

            protected:
                [[nodiscard]] std::size_t getMaxElementSize() const {
                    std::size_t size = sizeof(DType);
                    if constexpr (std::is_same<string_, DType>::value || std::is_same<unicode_, DType>::value) {
                        size = 1;
                        for (Size i = 0; i < len(); ++i) {
                            if (get(i).size() > size) {
                                size = get(i).size();
                            }
                        }
                    }
                    return size;
                }

                void dumpToStreamAsBinary(std::ostream &stream) {
                    for (std::size_t index = 0; index < size(); ++index) {
                        ndarray::internal::dumpObject(stream, get(index));
                    }
                }

                IndexType<DType> runHandlers(Size dimIndex, const std::string &dimCond) const;
                IndexType<DType> none(Size dimIndex, const std::string &dimCond) const;
                IndexType<DType> subsetting(Size dimIndex, const std::string &dimCond) const;
                IndexType<DType> slicing(Size dimIndex, const std::string &dimCond) const;
                IndexType<DType> booleanIndexing(Size dimIndex, const std::string &dimCond) const;

                void save(std::ostream &stream);
                auto load(std::istream &stream);

            private:
                Storage m_storage;
            };

            struct NDArrayBaseHasher {
                template<typename DType, typename Derived, typename Storage>
                auto operator()(const NDArrayBase<DType, Derived, Storage> &array) const -> std::size_t {
                    std::size_t h{0};
                    for (Size i = 0; i < array.size(); ++i) {
                        h ^= std::hash<DType>{}(array.get(i));
                    }
                    return h;
                }
            };

            struct NDArrayBaseEqualTo {
                template<typename DType, typename Derived, typename Storage>
                auto operator()(const NDArrayBase<DType, Derived, Storage> &x,
                                const NDArrayBase<DType, Derived, Storage> &y) const -> bool {
                    return x.array_equal(y);
                }
            };

        }// namespace internal
    }    // namespace ndarray
}// namespace np

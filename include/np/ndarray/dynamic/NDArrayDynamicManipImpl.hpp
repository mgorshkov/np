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

#include <np/ndarray/dynamic/NDArrayDynamicDecl.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            // Returns a new Array with axes transposed.
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::transpose() const {
                auto shapeTransposed = shape();
                if (shapeTransposed.empty()) {
                    return *this;
                }
                shapeTransposed.transpose();
                NDArrayDynamic<DType, Storage> result{shapeTransposed};

                Size row = 0;
                Size column = 0;
                Size last = shapeTransposed[0];
                for (Size i = 0; i < size(); ++i) {
                    if (row * last + column >= size()) {
                        ++column;
                        row = 0;
                    }
                    result.set(i, get(row * last + column));
                    ++row;
                }

                return result;
            }

            // Flatten the array
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::ravel() const {
                auto sh = shape();
                sh.flatten();
                return reshape(sh);
            }

            // Reshape, but don’t change data
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::reshape(const Shape& shape) const {
                if (size() != ndarray::internal::calcSizeByShape(shape))
                    throw std::runtime_error("Sizes of new and current arrays must be equal");
                auto result = copy();
                result.m_ArrayImpl.setShape(shape);
                return result;
            }

            // Return a new array with new size
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::resize(const Shape& shape) const {
                Size newSize = ndarray::internal::calcSizeByShape(shape);
                Size copySize = std::min(size(), newSize);
                NDArrayDynamic<DType, Storage> result{shape};
                for (Size offset = 0; offset < newSize; offset += copySize) {
                    std::copy(m_ArrayImpl.cbegin(), m_ArrayImpl.cbegin() + copySize, result.m_ArrayImpl.begin() + offset);
                }
                return result;
            }

            // Append items to an array
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::append(const NDArrayDynamic<DType, Storage>& array) const {
                if (array.size() == 0)
                    return *this;
                auto size1 = ndarray::internal::calcSizeByShape(shape());
                auto size2 = ndarray::internal::calcSizeByShape(array.shape());
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result(sh);
                std::copy(m_ArrayImpl.cbegin(), m_ArrayImpl.cend(), result.m_ArrayImpl.begin());
                std::copy(array.m_ArrayImpl.cbegin(), array.m_ArrayImpl.cend(), result.m_ArrayImpl.begin() + m_ArrayImpl.size());
                return result;
            }

            // Insert items in an array
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::insert(Size index, const NDArrayDynamic<DType, Storage>& array) const {
                auto size1 = size();
                if (index > size1) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                if (array.size() == 0)
                    return *this;
                auto size2 = array.size();
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result(sh);
                std::copy(m_ArrayImpl.cbegin(), m_ArrayImpl.cbegin() + index, result.m_ArrayImpl.begin());
                std::copy(array.m_ArrayImpl.cbegin(), array.m_ArrayImpl.cend(), result.m_ArrayImpl.begin() + index);
                std::copy(m_ArrayImpl.cbegin() + index, m_ArrayImpl.cend(), result.m_ArrayImpl.begin() + index + array.m_ArrayImpl.size());
                return result;
            }

            // Delete an item from an array
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::del(Size index) const {
                auto sz = size();
                if (sz == 0)
                    throw std::runtime_error("Cannot del from an empty array");
                if (index > sz) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                Shape sh{sz - 1};
                NDArrayDynamic<DType, Storage> result(sh);
                std::copy(m_ArrayImpl.cbegin(), m_ArrayImpl.cbegin() + index, result.begin());
                if (index < sz - 1) {
                    std::copy(m_ArrayImpl.cbegin() + index + 1, m_ArrayImpl.cend(), result.begin() + index);
                }
                return result;
            }

            // Concatenate arrays
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::concatenate(const NDArrayDynamic<DType, Storage>& array) const {
                if (array.size() == 0)
                    return *this;
                auto size1 = ndarray::internal::calcSizeByShape(shape());
                auto size2 = ndarray::internal::calcSizeByShape(array.shape());
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result(sh);
                std::copy(m_ArrayImpl.cbegin(), m_ArrayImpl.cend(), result.m_ArrayImpl.begin());
                std::copy(array.m_ArrayImpl.cbegin(), array.m_ArrayImpl.cend(), result.m_ArrayImpl.begin() + m_ArrayImpl.size());
                return result;
            }

            // Stack arrays vertically (rowwise)
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::vstack(const NDArrayDynamic<DType, Storage>& array) const {
                if (size() == 0) {
                    return array;
                }
                if (array.size() == 0) {
                    return *this;
                }
                // Both are not empty
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                // All the dims except the first should be equal
                for (std::size_t i = 1; i < sh1.size(); ++i) {
                    if (sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the first should be equal");
                }
                Shape sh = shape();
                sh[0] = sh1[0] + sh2[0];
                NDArrayDynamic<DType, Storage> result{array};
                result.m_ArrayImpl.setShape(sh);
                return result.insert(0, *this);
            }

            // Stack arrays vertically (rowwise)
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::r_(const NDArrayDynamic<DType, Storage>& array) const {
                return vstack(array);
            }

            // Stack arrays horizontally (columnwise)
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::hstack(const NDArrayDynamic<DType, Storage>& array) const {
                if (size() == 0) {
                    return array;
                }
                if (array.size() == 0) {
                    return *this;
                }
                // Both are not empty
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                if (sh1.size() == 1) {
                    //concatenation along 1st axis
                    return concatenate(array);
                }
                // All the dims except the first should be equal
                for (std::size_t i = 0; i < sh1.size(); ++i) {
                    if (i != 1 && sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the second should be equal");
                }
                Shape sh = shape();
                sh[1] = sh1[1] + sh2[1];
                NDArrayDynamic<DType, Storage> result{sh};
                std::size_t destOffset = 0;
                for (Size y = 0; y < sh[0]; ++y) {
                    std::copy(m_ArrayImpl.cbegin() + y * sh1[1],
                              m_ArrayImpl.cbegin() + (y + 1) * sh1[1],
                              result.m_ArrayImpl.begin() + destOffset);
                    destOffset += sh1[1];
                    std::copy(array.m_ArrayImpl.cbegin() + y * sh2[1],
                              array.m_ArrayImpl.cbegin() + (y + 1) * sh2[1],
                              result.m_ArrayImpl.begin() + destOffset);
                    destOffset += sh2[1];
                }
                return result;
            }

            // Create stacked columnwise arrays
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::column_stack(const NDArrayDynamic<DType, Storage>& array) const {
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != 1 || sh2.size() != 1)
                    throw std::runtime_error("Arrays must be 1D");
                if (sh1[0] != sh2[0])
                    throw std::runtime_error("Arrays have different dimensions");
                Size length = sh1[0];
                Shape sh{length, 2};
                NDArrayDynamic<DType, Storage> result{sh};
                std::size_t index1 = 0;
                std::size_t index2 = 0;
                Size i = 0;
                while (i < size() + array.size()) {
                    result.set(i++, get(index1++));
                    result.set(i++, array.get(index2++));
                }
                return result;
            }

            // Stack arrays horizontally (columnwise)
            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::c_(const NDArrayDynamic<DType, Storage>& array) const {
                return hstack(array);
            }

            // Splitting arrays
            // Split the array horizontally
            template<typename DType, typename Storage>
            std::vector<NDArrayDynamic<DType, Storage>> NDArrayDynamic<DType, Storage>::hsplit(Size index) const {
                auto sz = size();
                if (sz == 0)
                    throw std::runtime_error("Cannot hsplit an empty array");
                if (index > sz) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                Shape sh = shape();
                std::size_t splitIndex;
                if (sh.size() == 1) {
                    splitIndex = 0;
                } else {
                    splitIndex = 1;
                }
                Shape sh1{sh};
                sh1[splitIndex] = index;
                Shape sh2{sh};
                auto rest = sh[splitIndex] - index;
                sh2[splitIndex] = rest;
                NDArrayDynamic<DType, Storage> result1{sh1};
                NDArrayDynamic<DType, Storage> result2{sh2};
                Size i = 0;
                Size i1 = 0;
                Size i2 = 0;
                while (i < size()) {
                    std::copy(m_ArrayImpl.cbegin() + i,
                              m_ArrayImpl.cbegin() + i + index,
                              result1.m_ArrayImpl.begin() + i1);
                    i1 += index;
                    i += index;
                    std::copy(m_ArrayImpl.cbegin() + i,
                              m_ArrayImpl.cbegin() + i + rest,
                              result2.m_ArrayImpl.begin() + i2);
                    i2 += rest;
                    i += rest;
                }
                return {result1, result2};
            }

            // Split the array vertically
            template<typename DType, typename Storage>
            std::vector<NDArrayDynamic<DType, Storage>> NDArrayDynamic<DType, Storage>::vsplit(Size index) const {
                auto sz = size();
                if (sz == 0)
                    throw std::runtime_error("Cannot vsplit an empty array");
                Shape sh = shape();
                if (sh.size() < 2) {
                    throw std::runtime_error("vsplit only works on arrays of 2 or more dimensions");
                }
                if (index > len() - 1) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                Shape sh1{sh};
                sh1[0] = index;
                Shape sh2{sh};
                auto rest = sh[0] - index;
                sh2[0] = rest;
                NDArrayDynamic<DType, Storage> result1{sh1};
                NDArrayDynamic<DType, Storage> result2{sh2};
                Size i = 0;
                std::copy(m_ArrayImpl.cbegin(),
                    m_ArrayImpl.cbegin() + sh[1] * index,
                    result1.m_ArrayImpl.begin());
                i += sh[1] * index;
                std::copy(m_ArrayImpl.cbegin() + i,
                    m_ArrayImpl.cbegin() + i + sh[1] * rest,
                    result2.m_ArrayImpl.begin());
                return {result1, result2};
            }
        }
    }
}

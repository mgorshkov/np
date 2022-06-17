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
            template<typename DType, typename Storage>
            NDArrayDynamic<DType> NDArrayDynamic<DType, Storage>::transpose() const {
                auto sh = shape();
                if (sh.empty()) {
                    return NDArrayDynamic<DType>{};
                }
                sh.transpose();
                NDArrayDynamic<DType> result{sh};
                if (sh.size() == 1) {
                    for (Size i = 0; i < size(); ++i) {
                        result.set(i, get(i));
                    }
                } else if (sh.size() > 1) {
                    Size dim1 = shape()[0];
                    std::vector<NDArrayDynamic<DType>> subarrays;
                    for (Size i = 0; i < dim1; ++i) {
                        auto tr = at(i);
                        subarrays.push_back(tr.transpose());
                    }
                    std::size_t index = 0;
                    for (Size i = 0; i < subarrays[0].size(); ++i) {
                        for (Size d = 0; d < dim1; ++d) {
                            result.set(index++, subarrays[d].get(i));
                        }
                    }
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::ravel() const {
                auto sh = shape();
                sh.flatten();
                return reshape(sh);
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::reshape(const Shape &shape) const {
                if (size() != ndarray::internal::calcSizeByShape(shape))
                    throw std::runtime_error("Sizes of new and current arrays must be equal");
                auto result = copy();
                result.m_ArrayImpl.setShape(shape);
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::resize(const Shape &shape) const {
                Size newSize = ndarray::internal::calcSizeByShape(shape);
                Size copySize = std::min(size(), newSize);
                NDArrayDynamic<DType, Storage> result{shape};
                for (Size offset = 0; offset < newSize; offset += copySize) {
                    for (auto i = 0; i < copySize && i + offset < newSize; ++i) {
                        result.set(i + offset, get(i));
                    }
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::append(const NDArrayDynamic<DType, Storage> &array) const {
                if (array.size() == 0)
                    return copy();
                auto size1 = ndarray::internal::calcSizeByShape(shape());
                auto size2 = ndarray::internal::calcSizeByShape(array.shape());
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result{sh};
                for (auto i = 0; i < size(); ++i) {
                    result.set(i, get(i));
                }
                for (auto i = 0; i < array.size(); ++i) {
                    result.set(i + size(), array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::insert(Size index, const NDArrayDynamic<DType, Storage> &array) const {
                auto size1 = size();
                if (index > size1) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                if (array.size() == 0)
                    return copy();
                auto size2 = array.size();
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result(sh);
                for (auto i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                for (auto i = 0; i < array.size(); ++i) {
                    result.set(i + index, array.get(i));
                }
                for (auto i = index; i < size(); ++i) {
                    result.set(i + array.size(), get(i));
                }
                return result;
            }

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
                for (auto i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                if (index < sz - 1) {
                    for (auto i = index + 1; i < size(); ++i) {
                        result.set(i - 1, get(i));
                    }
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::concatenate(const NDArrayDynamic<DType, Storage> &array) const {
                if (array.size() == 0)
                    return copy();
                auto size1 = ndarray::internal::calcSizeByShape(shape());
                auto size2 = ndarray::internal::calcSizeByShape(array.shape());
                Shape sh{size1 + size2};
                NDArrayDynamic<DType, Storage> result(sh);
                for (auto i = 0; i < size(); ++i) {
                    result.set(i, get(i));
                }
                for (auto i = 0; i < array.size(); ++i) {
                    result.set(i + size(), array.get(i));
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::vstack(const NDArrayDynamic<DType, Storage> &array) const {
                if (size() == 0) {
                    return array.copy();
                }
                if (array.size() == 0) {
                    return copy();
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

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::r_(const NDArrayDynamic<DType, Storage> &array) const {
                return vstack(array);
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::hstack(const NDArrayDynamic<DType, Storage> &array) const {
                if (size() == 0) {
                    return array.copy();
                }
                if (array.size() == 0) {
                    return copy();
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
                // All the dims except the second should be equal
                for (std::size_t i = 0; i < sh1.size(); ++i) {
                    if (i != 1 && sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the second should be equal");
                }
                Shape sh = shape();
                sh[1] = sh1[1] + sh2[1];
                NDArrayDynamic<DType, Storage> result{sh};
                std::size_t destOffset = 0;
                for (Size y = 0; y < sh[0]; ++y) {
                    for (auto i = 0; i < sh1[1]; ++i) {
                        result.set(i + destOffset, get(i + y * sh1[1]));
                    }
                    destOffset += sh1[1];
                    for (auto i = 0; i < sh2[1]; ++i) {
                        result.set(i + destOffset, array.get(i + y * sh2[1]));
                    }
                    destOffset += sh2[1];
                }
                return result;
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::column_stack(const NDArrayDynamic<DType, Storage> &array) const {
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

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::c_(const NDArrayDynamic<DType, Storage> &array) const {
                if (size() == 0) {
                    return array.copy();
                }
                if (array.size() == 0) {
                    return copy();
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
                // All the dims except the last should be equal
                Size last = sh1.size() - 1;
                Size sizes = 1;
                for (auto i = 0; i < last; ++i) {
                    if (sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the last should be equal");
                    sizes *= sh1[i];
                }
                Shape sh = shape();
                sh[last] = sh1[last] + sh2[last];
                NDArrayDynamic<DType, Storage> result{sh};
                std::size_t destOffset = 0;
                for (Size y = 0; y < sizes; ++y) {
                    for (auto i = 0; i < sh1[last]; ++i) {
                        result.set(i + destOffset, get(i + y * sh1[last]));
                    }
                    destOffset += sh1[last];
                    for (auto i = 0; i < sh2[last]; ++i) {
                        result.set(i + destOffset, array.get(i + y * sh2[last]));
                    }
                    destOffset += sh2[last];
                }
                return result;
            }

            template<typename DType, typename Storage>
            std::vector<NDArrayDynamic<DType, Storage>> NDArrayDynamic<DType, Storage>::hsplit(std::size_t sections) const {
                if (sections == 0) {
                    throw std::runtime_error("Sections must not be 0");
                }
                Shape sh = shape();
                Size size = sh.size() < 2 ? 0 : sh[1];
                if (size % sections != 0) {
                    throw std::runtime_error("Array split does not result in an equal division");
                }
                std::size_t splitSize = size / sections;
                std::vector<NDArrayDynamic<DType, Storage>> results;
                Size i = 0;
                Size i1 = 0;
                while (i < size()) {
                    NDArrayDynamic<DType, Storage> result;
                    for (int j = 0; j < splitSize; ++j) {
                        result.set(j + i1, get(i + j));
                    }
                    i1 += splitSize;
                    i += splitSize;
                    results.push_back(result);
                }
                return results;
            }

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
                for (int j = 0; j < sh[1] * index; ++j) {
                    result1.set(j, get(j));
                }
                i += sh[1] * index;
                for (int j = 0; j < sh[1] * rest; ++j) {
                    result2.set(j, get(j + i));
                }
                return {result1, result2};
            }
        }// namespace array_dynamic
    }// namespace ndarray
}// namespace np

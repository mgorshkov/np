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
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::concatenate(const NDArrayDynamic<DType, Storage> &array, std::optional<std::size_t> axis) const {
                if (array.size() == 0)
                    return copy();
                if (axis == std::nullopt) {
                    return concatenate(array, 0);
                }
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                // All the dims except the 'axis' should be equal
                for (std::size_t i = 0; i < sh1.size(); ++i) {
                    if (i != *axis && sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the first should be equal");
                }
                if (*axis >= sh1.size())
                    throw std::runtime_error("axis : " + std::to_string(*axis) + " is out of bounds for array of dimension " + std::to_string(sh1.size()));
                Shape sh{sh1};
                auto size1 = ndarray::internal::calcSizeByShape(sh1);
                auto size2 = ndarray::internal::calcSizeByShape(sh2);
                if (*axis == 0) {
                    sh[*axis] = sh1[*axis] + sh2[*axis];
                    NDArrayDynamic<DType, Storage> result{sh};
                    for (auto i = 0; i < size1; ++i) {
                        result.set(i, get(i));
                    }
                    for (auto i = 0; i < size2; ++i) {
                        result.set(i + size1, array.get(i));
                    }
                    return result;
                }
                if (*axis == 1) {
                    Size SizeT2 = std::accumulate(sh.begin() + 1, sh.end(), 1, std::multiplies<Size>());
                    sh[*axis] = sh1[*axis] + sh2[*axis];
                    NDArrayDynamic<DType> result{sh};
                    Size destOffset = 0;
                    Size src1Offset = 0;
                    Size src2Offset = 0;
                    for (Size x = 0; x < sh1[0]; ++x) {
                        for (Size y = 0; y < SizeT2; ++y) {
                            result.set(destOffset++, get(src1Offset++));
                        }
                        for (Size y = 0; y < SizeT2; ++y) {
                            result.set(destOffset++, array.get(src2Offset++));
                        }
                    }
                    return result;
                }

                throw std::runtime_error("axis > 1 are not supported");
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::vstack(const NDArrayDynamic<DType, Storage> &array) const {
                return concatenate(array);
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::r_(const NDArrayDynamic<DType, Storage> &array) const {
                return concatenate(array);
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
                return sh1.size() == 1 ?
                                       //concatenation along 1st axis
                               concatenate(array)
                                       :
                                       //concatenation along 2nd axis
                               concatenate(array, 1);
            }

            template<typename DType, typename Storage>
            NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::column_stack(const NDArrayDynamic<DType, Storage> &array) const {
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                if (sh1.size() == 0 && sh2.size() == 0) {
                    return NDArrayDynamic<DType>{};
                }
                if (sh1.size() == 1) {
                    Size length = sh1[0];
                    Shape sh{length, 2};
                    NDArrayDynamic<DType> result{sh};
                    std::size_t index1 = 0;
                    std::size_t index2 = 0;
                    Size i = 0;
                    while (i < size() + array.size()) {
                        result.set(i++, get(index1++));
                        result.set(i++, array.get(index2++));
                    }
                    return result;
                }
                //concatenation along 2nd axis
                return concatenate(array, 1);
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
                if (sh.size() == 0) {
                    std::vector<NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.push_back(NDArrayDynamic<DType>{});
                    }
                    return results;
                } else if (sh.size() == 1) {
                    Shape sh1{sh};
                    if (sh[0] % sections != 0) {
                        throw std::runtime_error("Array split does not result in an equal division");
                    }
                    Size sectionSize = sh[0] / sections;
                    sh1[0] = sectionSize;
                    std::vector<NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.push_back(NDArrayDynamic<DType>{sh1});
                    }
                    std::vector<Size> sectionIndexes(sections);
                    for (auto i = 0; i < size(); ++i) {
                        Size section = i / sectionSize;
                        results[section].set(sectionIndexes[section]++, get(i));
                    }
                    return results;
                }
                Shape sh1{sh};
                if (sh[1] % sections != 0) {
                    throw std::runtime_error("Array split does not result in an equal division");
                }
                Size rest = 1;
                for (std::size_t i = 2; i < sh1.size(); ++i) {
                    rest *= sh1[i];
                }
                Size sectionSize = sh[1] / sections;
                sh1[1] = sectionSize;
                std::vector<NDArrayDynamic<DType>> results;
                for (std::size_t section = 0; section < sections; ++section) {
                    results.push_back(NDArrayDynamic<DType>{sh1});
                }
                std::vector<Size> sectionIndexes(sections);
                Size i = 0;
                std::size_t section = 0;
                while (i < size()) {
                    for (auto j = 0; j < rest; ++j) {
                        results[section].set(sectionIndexes[section]++, get(i++));
                    }
                    ++section;
                    if (section >= sections) {
                        section = 0;
                    }
                }
                return results;
            }

            template<typename DType, typename Storage>
            std::vector<NDArrayDynamic<DType, Storage>> NDArrayDynamic<DType, Storage>::vsplit(std::size_t sections) const {
                if (sections == 0) {
                    throw std::runtime_error("Sections must not be 0");
                }
                Shape sh = shape();
                if (sh.size() == 0) {
                    std::vector<NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.push_back(NDArrayDynamic<DType>{});
                    }
                    return results;
                } else if (sh.size() == 1) {
                    Shape sh1{sh};
                    if (sh[0] % sections != 0) {
                        throw std::runtime_error("Array split does not result in an equal division");
                    }
                    Size sectionSize = sh[0] / sections;
                    sh1[0] = sectionSize;
                    std::vector<NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.push_back(NDArrayDynamic<DType>{sh1});
                    }
                    std::vector<Size> sectionIndexes(sections);
                    for (auto i = 0; i < size(); ++i) {
                        Size section = i / sectionSize;
                        results[section].set(sectionIndexes[section]++, get(i));
                    }
                    return results;
                }
                Shape sh0{sh};
                if (sh[0] % sections != 0) {
                    throw std::runtime_error("Array split does not result in an equal division");
                }
                Size rest = 1;
                for (std::size_t i = 1; i < sh0.size(); ++i) {
                    rest *= sh0[i];
                }
                Size sectionSize = sh[0] / sections;
                sh0[0] = sectionSize;
                std::vector<NDArrayDynamic<DType>> results;
                for (std::size_t section = 0; section < sections; ++section) {
                    results.push_back(NDArrayDynamic<DType>{sh0});
                }
                std::vector<Size> sectionIndexes(sections);
                Size i = 0;
                std::size_t section = 0;
                while (i < size()) {
                    for (auto j = 0; j < rest; ++j) {
                        results[section].set(sectionIndexes[section]++, get(i++));
                    }
                    ++section;
                    if (section >= sections) {
                        section = 0;
                    }
                }
                return results;
            }
        }// namespace array_dynamic
    }    // namespace ndarray
}// namespace np

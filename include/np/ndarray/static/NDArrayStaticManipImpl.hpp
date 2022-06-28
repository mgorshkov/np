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

#include <np/ndarray/static/NDArrayStaticDecl.hpp>
#include <np/ndarray/static/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            // Returns a new Array with axes transposed.
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::transpose() const {
                Shape sh = shape();
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
                        auto subarray = at(i);
                        subarrays.push_back(subarray.transpose());
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

            // Flatten the array
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::ravel() const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                NDArrayStatic<DType, size> result;
                for (auto i = 0; i < size; ++i) {
                    result.set(i, get(i));
                }
                return result;
            }

            // Reshape, but don’t change data
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::reshape(const Shape &shape) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                auto newSize = calcSizeByShape(shape);
                if (size != newSize) {
                    throw std::runtime_error("New size does not equal to the old one");
                }
                NDArrayDynamic<DType> result(shape);
                for (auto i = 0; i < size; ++i) {
                    result.set(i, get(i));
                }
                return result;
            }

            // Adding and removing elements
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::resize(const Shape &shape) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                Size newSize = calcSizeByShape(shape);
                auto copySize = std::min(size, newSize);
                NDArrayDynamic<DType> result{shape};
                for (Size offset = 0; offset < newSize; offset += copySize) {
                    for (auto i = 0; i < copySize && i + offset < newSize; ++i) {
                        result.set(i + offset, get(i));
                    }
                }
                return result;
            }

            // Append items to an array
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::append(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                NDArrayStatic<DType, 2 * size> result;
                for (auto i = 0; i < size; ++i) {
                    result.set(i, get(i));
                    result.set(i + size, array.get(i));
                }
                return result;
            }

            // Insert items in an array
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::insert(Size index, const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                NDArrayStatic<DType, 2 * size> result;
                for (auto i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                for (auto i = 0; i < size; ++i) {
                    result.set(i + index, array.get(i));
                }
                for (auto i = index; i < size; ++i) {
                    result.set(i + size, get(i));
                }
                return result;
            }

            // Delete items from an array
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, (SizeT * ... * SizeTs) - 1> NDArrayStatic<DType, SizeT, SizeTs...>::del(Size index) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                NDArrayStatic<DType, size - 1> result;
                for (auto i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                if (index < size - 1) {
                    for (auto i = index + 1; i < size; ++i) {
                        result.set(i - 1, get(i));
                    }
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::concatenate(const NDArrayStatic<DType, SizeT, SizeTs...> &array, std::optional<std::size_t> axis) const {
                if (array.size() == 0)
                    return NDArrayDynamic<DType>{};
                if (axis == std::nullopt) {
                    return concatenate(array, 0);
                }
                auto v = internal::to_vector(std::tuple(SizeT, SizeTs...));
                if (*axis >= v.size())
                    throw std::runtime_error("axis : " + std::to_string(*axis) + " is out of bounds for array of dimension " + std::to_string(v.size()));
                if (*axis == 0) {
                    v[*axis] = 2 * v[*axis];
                    NDArrayDynamic<DType> result{Shape{v}};
                    for (Size i = 0; i < size(); ++i) {
                        result.set(i, get(i));
                    }
                    for (Size i = 0; i < size(); ++i) {
                        result.set(i + size(), array.get(i));
                    }
                    return result;
                }
                if (*axis == 1) {
                    Size SizeT2 = std::accumulate(v.begin() + 1, v.end(), 1, std::multiplies<Size>());
                    v[*axis] = 2 * v[*axis];
                    NDArrayDynamic<DType> result{Shape{v}};
                    Size destOffset = 0;
                    Size src1Offset = 0;
                    Size src2Offset = 0;
                    for (Size x = 0; x < SizeT; ++x) {
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

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::vstack(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return concatenate(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::r_(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return concatenate(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::hstack(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return sizeof...(SizeTs) == 0 ?
                    concatenate(array) : concatenate(array, 1U);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::column_stack(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
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
                return concatenate(array, 1U);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::c_(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
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
                NDArrayDynamic<DType> result{sh};
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

            template<typename DType, Size SizeT, Size... SizeTs>
            inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::hsplit(std::size_t sections) const {
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

            // Split the array vertically
            template<typename DType, Size SizeT, Size... SizeTs>
            inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::vsplit(std::size_t sections) const {
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
        }// namespace array_static
    }    // namespace ndarray
}// namespace np

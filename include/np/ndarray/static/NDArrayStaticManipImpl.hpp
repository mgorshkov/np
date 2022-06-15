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

            // Concatenate arrays
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::concatenate(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                NDArrayStatic<DType, 2 * size> result;
                for (auto i = 0; i < size; ++i) {
                    result.set(i, get(i));
                }
                for (auto i = 0; i < size; ++i) {
                    result.set(i + size, array.get(i));
                }
                return result;
            }

            // Stack arrays vertically (rowwise)
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::vstack(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                auto result = array.ravel();
                return result.insert(0, ravel());
            }

            // Stack arrays vertically (rowwise)
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::r_(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return vstack(array);
            }

            // Stack arrays horizontally (columnwise)
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::hstack(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                static constexpr Size size = (SizeT * ... * SizeTs);
                if constexpr (sizeof...(SizeTs) == 0) {
                    NDArrayDynamic<DType> result{Shape{2 * size}};
                    for (auto i = 0; i < size; ++i) {
                        result.set(i, get(i));
                    }
                    for (auto i = 0; i < size; ++i) {
                        result.set(i + size, array.get(i));
                    }
                    return result;
                }
                if constexpr (sizeof...(SizeTs) > 0) {
                    auto v = internal::to_vector(std::tuple(SizeT, SizeTs...));
                    Size SizeT2 = v[1];
                    v[1] = 2 * v[1];
                    NDArrayDynamic<DType> result{Shape{v}};
                    std::size_t destOffset = 0;
                    for (Size y = 0; y < SizeT; ++y) {
                        for (auto i = 0; i < SizeT2; ++i) {
                            result.set(i + destOffset, get(i + y * SizeT2));
                        }
                        destOffset += SizeT2;
                        for (auto i = 0; i < SizeT2; ++i) {
                            result.set(i + destOffset, array.get(i + y * SizeT2));
                        }
                        destOffset += SizeT2;
                    }
                    return result;
                }
            }

            // Create stacked columnwise arrays
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::column_stack(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != 1 || sh2.size() != 1)
                    throw std::runtime_error("Arrays must be 1D");
                if (sh1[0] != sh2[0])
                    throw std::runtime_error("Arrays have different dimensions");
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

            // Stack arrays horizontally (columnwise)
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::c_(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return hstack(array);
            }

            // Splitting arrays
            // Split the array horizontally
            template<typename DType, Size SizeT, Size... SizeTs>
            inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::hsplit(Size index) const {
                std::size_t splitIndex;
                if constexpr (sizeof...(SizeTs) == 0) {
                    splitIndex = 0;
                } else {
                    splitIndex = 1;
                }
                Shape sh{SizeT, SizeTs...};
                Shape sh1{SizeT, SizeTs...};
                sh1[splitIndex] = index;
                Shape sh2{SizeT, SizeTs...};
                auto rest = sh[splitIndex] - index;
                sh2[splitIndex] = rest;
                NDArrayDynamic<DType> result1{sh1};
                NDArrayDynamic<DType> result2{sh2};
                Size i = 0;
                Size i1 = 0;
                Size i2 = 0;
                while (i < size()) {
                    for (int j = 0; j < index; ++j) {
                        result1.set(j + i1, get(i + j));
                    }
                    i1 += index;
                    i += index;
                    for (int j = 0; j < rest; ++j) {
                        result2.set(j + i2, get(i + j));
                    }
                    i2 += rest;
                    i += rest;
                }
                return {result1, result2};
            }

            // Split the array vertically
            template<typename DType, Size SizeT, Size... SizeTs>
            inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::vsplit(Size index) const {
                Shape sh{SizeT, SizeTs...};
                Shape sh1{sh};
                sh1[0] = index;
                Shape sh2{sh};
                auto rest = sh[0] - index;
                sh2[0] = rest;
                NDArrayDynamic<DType> result1{sh1};
                NDArrayDynamic<DType> result2{sh2};
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
        }// namespace array_static
    }// namespace ndarray
}// namespace np

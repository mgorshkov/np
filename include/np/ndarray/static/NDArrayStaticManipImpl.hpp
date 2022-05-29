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

namespace np::ndarray::array_static {
    // Returns a new Array with axes transposed.
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::transpose() const {
        static constexpr auto types = np::internal::ReverseAppendToArgList<SizeT, SizeTs...>::type;
        NDArrayDynamic<DType> result{Shape{np::ndarray::array_static::internal::to_vector(types)}};
        static constexpr Size size = (SizeT * ... * SizeTs);
        Size offset = 0;
        std::size_t column = 0;
        for (Size i = 0; i < size; ++i) {
            result.set(i, get(offset + column));
            offset += SizeT;
            if (offset >= size) {
                offset = 0;
                column++;
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
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::reshape(const Shape& shape) const {
        static constexpr Size size = (SizeT * ... * SizeTs);
        auto newSize = shape.size();
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
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::resize(const Shape& shape) const {
        static constexpr Size size = (SizeT * ... * SizeTs);
        Size newSize = static_cast<Size>(shape.size());
        auto copySize = std::min(size, newSize);
        NDArrayDynamic<DType> result(shape);
        for (Size offset = 0; offset < newSize; offset += copySize) {
            for (auto i = 0; i < size; ++i) {
                result.set(i + offset, get(i + copySize));
            }
        }
        return result;
    }

    // Append items to an array
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::append(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
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
    inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::insert(Size index, const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        static constexpr Size size = (SizeT * ... * SizeTs);
        NDArrayStatic<DType, 2 * size> result;
        for (auto i = 0; i < index; ++i) {
            result.set(i, get(i));
        }
        for (auto i = 0; i < size; ++i) {
            result.set(i + index, array.get(i));
        }
        for (auto i = 0; i < size - index; ++i) {
            result.set(i + index + size, array.get(i + index));
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
            for (auto i = 0; i < size - index; ++i) {
                result.set(i + index, get(i));
            }
        }
        return result;
    }

    // Concatenate arrays
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::concatenate(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
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
    inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::vstack(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        auto result = array.ravel();
        return result.insert(0, ravel());
    }

    // Stack arrays vertically (rowwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> NDArrayStatic<DType, SizeT, SizeTs...>::r_(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        return vstack(array);
    }

    // Stack arrays horizontally (columnwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::hstack(
            const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        static constexpr Size size = 2 * (SizeT * ... * SizeTs);
        if constexpr(size == 2) {
            return concatenate(array);
        }
        if constexpr(size > 2) {
            Size SizeT2 = std::get<1>(std::tuple(SizeT, SizeTs...));
            NDArrayDynamic<DType> result{Shape{size}};
            std::size_t destOffset = 0;
            for (Size y = 0; y < SizeT; ++y) {
                for (auto i = 0; i < SizeT2; ++i) {
                    result.set(i + destOffset, get(i + y * SizeT2));
                }
                destOffset += SizeT2;
                for (auto i = 0; i < SizeT2; ++i) {
                    result.set(i + destOffset, get(i + y * SizeT2));
                }
                destOffset += SizeT2;
            }
            return result;
        }
    }

    // Create stacked columnwise arrays
    template<typename DType, Size SizeT, Size... SizeTs>
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::column_stack(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        Shape sh1 = shape();
        Shape sh2 = array.shape();
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
    inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::c_(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        return hstack(array);
    }

    // Splitting arrays
    // Split the array horizontally
    template<typename DType, Size SizeT, Size... SizeTs>
    inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::hsplit(Size index) const {
        std::vector<NDArrayDynamic<DType>> result;
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
            std::copy(cbegin() + i,
            cbegin() + i + index,
            result1.begin() + i1);
            i1 += index;
            i += index;
            std::copy(cbegin() + i,
            cbegin() + i + rest,
            result2.begin() + i2);
            i2 += rest;
            i += rest;
        }
        return {result1, result2};
    }

    // Split the array vertically at the 2nd index
    template<typename DType, Size SizeT, Size... SizeTs>
    inline std::vector<NDArrayDynamic<DType>> NDArrayStatic<DType, SizeT, SizeTs...>::vsplit(Size index) const {
        std::vector<NDArrayDynamic<DType>> result;
        Shape sh{SizeT, SizeTs...};
        Shape sh1{sh};
        sh1[0] = index;
        Shape sh2{sh};
        auto rest = sh[0] - index;
        sh2[0] = rest;
        NDArrayDynamic<DType> result1{sh1};
        NDArrayDynamic<DType> result2{sh2};
        Size i = 0;
        std::copy(cbegin(),
                cbegin() + sh[1] * index,
        result1.begin());
        i += sh[1] * index;
        std::copy(cbegin() + i,
        cbegin() + i + sh[1] * rest,
        result2.begin());
        return result;
    }
} // namespace np::ndarray::array_static


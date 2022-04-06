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

#include <cstddef>
#include <iostream>
#include <iomanip>

#include <np/ndarray/dynamic/internal/NDArrayDynamicInternal.hpp>

namespace np::ndarray::array_dynamic::internal {

class SquareBracketsInserter {
public:
    explicit SquareBracketsInserter(std::ostream & stream) noexcept
        : m_Stream{stream}
    {
        m_Stream << "[";
    }

    ~SquareBracketsInserter() noexcept {
        m_Stream << "]";
    }

private:
    std::ostream& m_Stream;
};

template <typename DType, typename Storage>
std::ostream & operator<<(std::ostream &stream, const NDArrayDynamicInternal<DType, Storage> &array) {
    SquareBracketsInserter squareBracketsInserter(stream);

    if (!array.m_Shape.empty()) {
        if (array.m_Shape.size() == 1) {
            for (Size index = 0; index < array.m_Shape[0]; ++index) {
                if (index > 0)
                    stream << " ";
                if constexpr(std::is_floating_point<DType>::value) {
                    stream << std::setprecision(8);
                }
                if constexpr(std::is_same<DType, std::string>::value) {
                    stream << "\"";
                }
                stream << array.m_Impl[index];
                if constexpr(std::is_same<DType, std::string>::value) {
                    stream << "\"";
                }
            }
        } else {
            for (Size index = 0; index < array.m_Shape[0]; ++index) {
                if (index > 0) {
                    stream << std::endl << " ";
                }
                stream << array[index];
            }
        }
    }
    return stream;
}

}

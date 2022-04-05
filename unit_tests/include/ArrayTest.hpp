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

#include <iostream>
#include <gtest/gtest.h>

#include <np/ndarray/static/NDArrayStatic.hpp>
#include <np/ndarray/dynamic/NDArrayDynamic.hpp>

using namespace np;

class ArrayTest : public ::testing::Test {
protected:
    // dynamic arrays
    template <typename DType, typename Storage>
    inline void checkArrayRepr(const ndarray::array_dynamic::NDArrayDynamic<DType, Storage>& array, const char* repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(repr, ss.str());
    }

    template <typename DType, typename Storage>
    inline void checkArrayShape(const ndarray::array_dynamic::NDArrayDynamic<DType, Storage>& array, const Shape& shape) {
        EXPECT_EQ(shape, array.shape());
    }

    // static arrays
    template <typename DType, Size SizeT, Size... SizeTs>
    inline void checkArrayRepr(const ndarray::array_static::NDArrayStatic<DType, SizeT, SizeTs...>& array, const char* repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }

    template <typename DType, Size SizeT, Size... SizeTs>
    inline void checkArrayShape(const ndarray::array_static::NDArrayStatic<DType, SizeT, SizeTs...>& array, const Shape& shape) {
        EXPECT_EQ(shape, array.shape());
    }
};
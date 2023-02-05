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

#include <gtest/gtest.h>
#include <iostream>

class ArrayTest : public ::testing::Test {
protected:
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    static void compare(const np::ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &result, const np::ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &result_sample, bool shouldEqual = true) {
        bool equals = np::array_equal(result, result_sample);
        if (shouldEqual && !equals) {
            std::cerr << "Array " << result << " is not equal to " << result_sample << std::endl;
            EXPECT_TRUE(equals);
        } else if (!shouldEqual && equals) {
            std::cerr << "Array " << result << " is equal to " << result_sample << std::endl;
            EXPECT_FALSE(equals);
        }
    }

    template<typename DType, typename Derived, typename Storage>
    static void checkArrayRepr(const np::ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, const char *repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }

    template<typename DType, typename Derived, typename Storage>
    static void checkArrayShape(const np::ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, const np::Shape &shape) {
        EXPECT_EQ(shape, array.shape());
    }
};
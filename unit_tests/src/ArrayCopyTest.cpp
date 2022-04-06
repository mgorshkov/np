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

#include <np/Array.hpp>
#include <np/Copy.hpp>
#include <np/Comp.hpp>

using namespace np;

class ArrayCopyTest : public ::testing::Test {
protected:
    // dynamic arrays
    template <typename DType>
    inline void checkArrayCopy(const Array<DType>& array) {
        auto c = copy<DType>(array);
        bool equals = array_equal<DType>(c, array);
        EXPECT_TRUE(equals);
    }

    // static arrays
    template <typename DType, std::size_t SizeT, std::size_t... SizeTs>
    inline void checkArrayCopy(const Array<DType, SizeT, SizeTs...>& array) {
        auto c = copy<DType, SizeT, SizeTs...>(array);
        bool equals = array_equal<DType, SizeT, SizeTs...>(c, array);
        EXPECT_TRUE(equals);
    }
};

TEST_F(ArrayCopyTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    checkArrayCopy<int_>(array);
}

TEST_F(ArrayCopyTest, dynamicEmptyFloatArrayTest) {
    // dynamic
    Array<float_> array{};
    checkArrayCopy<float_>(array);
}

TEST_F(ArrayCopyTest, dynamicEmptyStringArrayTest) {
    // dynamic
    Array<string_> array{};
    checkArrayCopy<string_>(array);
}

TEST_F(ArrayCopyTest, dynamicEmptyUnicodeArrayTest) {
    // dynamic
    Array<unicode_> array{};
    checkArrayCopy<unicode_>(array);
}

TEST_F(ArrayCopyTest, static1DIntArrayTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    checkArrayCopy<int_, 3>(array);
}

TEST_F(ArrayCopyTest, static1DFloatArrayTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    checkArrayCopy<float_, 3>(array);
}

TEST_F(ArrayCopyTest, static1DStringArrayTest) {
    // static
    Array<string_, 3> array{"str1", "str2", "str3"};
    checkArrayCopy<string_, 3>(array);
}

TEST_F(ArrayCopyTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<float_> array{1, 2, 3};
    checkArrayCopy<float_>(array);
}

TEST_F(ArrayCopyTest, dynamic1DFloatArrayTest) {
    // dynamic
    Array<float_> array{1.1, 2.2, 3.3};
    checkArrayCopy<float_>(array);
}

TEST_F(ArrayCopyTest, dynamic1DStringArrayTest) {
    // dynamic
    Array<string_> array{"str1", "str2", "str3"};
    checkArrayCopy<string_>(array);
}

TEST_F(ArrayCopyTest, static2DIntArrayTest) {
    // static
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array{c_array_2d};
    checkArrayCopy<int_, 2, 3>(array);
}

TEST_F(ArrayCopyTest, static2DFloatArrayTest) {
    // static
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array{c_array_2d};
    checkArrayCopy<float_, 2, 3>(array);
}

TEST_F(ArrayCopyTest, static2DStringArrayTest) {
    // static
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array{c_array_2d};
    checkArrayCopy<string_, 2, 3>(array);
}

TEST_F(ArrayCopyTest, dynamic2DIntArrayTest) {
    // dynamic
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    checkArrayCopy<int_>(array);
}

TEST_F(ArrayCopyTest, dynamic2DFloatArrayTest) {
    // dynamic
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    checkArrayCopy<float_>(array);
}

TEST_F(ArrayCopyTest, dynamic2DStringArrayTest) {
    // dynamic
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    checkArrayCopy<string_>(array);
}

TEST_F(ArrayCopyTest, static3DIntArrayTest) {
    // static
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    checkArrayCopy<int_, 2, 2, 3>(array);
}

TEST_F(ArrayCopyTest, static3DFloatArrayTest) {
    // static
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    checkArrayCopy<float_, 2, 2, 3>(array);
}

TEST_F(ArrayCopyTest, static3DStringArrayTest) {
    // static
    string_ c_array_3d[2][4][3] = {
        {
            {"str1_1", "str1_2", "str1_3"},
            {"str2_1", "str2_2", "str2_3"},
            {"str3_1", "str3_2", "str3_3"},
            {"str4_1", "str4_2", "str4_3"}
        },
        {
            { "str5_1", "str5_2", "str5_3" },
            { "str6_1", "str6_2", "str6_3" },
            { "str7_1", "str7_2", "str7_3" },
            { "str8_1", "str8_2", "str8_3" }
        }
    };
    Array<string_, 2, 4, 3> array{c_array_3d};
    checkArrayCopy<string_, 2, 4, 3>(array);
}

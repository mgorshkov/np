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

#include <gtest/gtest.h>
#include <iostream>

#include <np/Array.hpp>
#include <np/Comp.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayComparisonTest : public ArrayTest {
protected:
};

TEST_F(ArrayComparisonTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<int_>(array1, array2);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamicEmptyFloatArrayTest) {
    // dynamic
    Array<float_> array1{};
    Array<float_> array2{};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<float_>(array1, array2);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamicEmptyStringArrayTest) {
    // dynamic
    Array<string_> array1{};
    Array<string_> array2{};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<string_>(array1, array2);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamicEmptyUnicodeArrayTest) {
    // dynamic
    Array<unicode_> array1{};
    Array<unicode_> array2{};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<unicode_>(array1, array2);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayComparisonTest, static1DIntArrayTest) {
    // static
    Array<int_, 3> array1{1, 3, 2};
    Array<int_, 3> array2{1, 2, 3};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 3>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 3>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 3>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<int_, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static1DFloatArrayTest) {
    // static
    Array<float_, 3> array1{1.1, 3.3, 2.2};
    Array<float_, 3> array2{1.1, 2.2, 3.3};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 3>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 3>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 3>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<float_, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static1DStringArrayTest) {
    Array<string_, 3> array1{"str1", "str3", "str2"};
    Array<string_, 3> array2{"str1", "str2", "str3"};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 3>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 3>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 3>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<string_, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array1{1, 3, 2};
    Array<int_> array2{1, 2, 3};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<int_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic1DFloatArrayTest) {
    Array<float_> array1{1.1, 3.3, 2.2};
    Array<float_> array2{1.1, 2.2, 3.3};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<float_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic1DStringArrayTest) {
    Array<string_> array1{"str1", "str3", "str2"};
    Array<string_> array2{"str1", "str2", "str3"};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[1 0 0]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[0 0 1]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[0 1 0]");
    }
    // Arraywise comparison
    {
        auto equal = array_equal<string_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static2DIntArrayTest) {
    long c_array_2d_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array1{c_array_2d_1};
    long c_array_2d_2[2][3] = {{1, 3, 2}, {6, 5, 4}};
    Array<int_, 2, 3> array2{c_array_2d_2};
    // Elementwise comparison
    {
        Array<bool_, 2, 3> equal = array1 == array2;
        checkArrayRepr<bool_, 2, 3>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 2, 3>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 2, 3>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<int_, 2, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static2DFloatArrayTest) {
    double c_array_2d_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array1{c_array_2d_1};
    double c_array_2d_2[2][3] = {{1.1, 3.3, 2.2}, {6.6, 5.5, 4.4}};
    Array<float_, 2, 3> array2{c_array_2d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 2, 3>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 2, 3>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 2, 3>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<float_, 2, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static2DStringArrayTest) {
    std::string c_array_2d_1[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array1{c_array_2d_1};
    std::string c_array_2d_2[2][3] = {{"str1", "str3", "str2"}, {"str6", "str5", "str4"}};
    Array<string_, 2, 3> array2{c_array_2d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 2, 3>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 2, 3>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 2, 3>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<string_, 2, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic2DIntArrayTest) {
    long c_array_2d_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1{c_array_2d_1};
    long c_array_2d_2[2][3] = {{1, 3, 2}, {6, 5, 4}};
    Array<int_> array2{c_array_2d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<int_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic2DFloatArrayTest) {
    double c_array_2d_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array1{c_array_2d_1};
    double c_array_2d_2[2][3] = {{1.1, 3.3, 2.2}, {6.6, 5.5, 4.4}};
    Array<float_> array2{c_array_2d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<float_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, dynamic2DStringArrayTest) {
    std::string c_array_2d_1[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array1{c_array_2d_1};
    std::string c_array_2d_2[2][3] = {{"str1", "str3", "str2"}, {"str6", "str5", "str4"}};
    Array<string_> array2{c_array_2d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_>(equal, "[[1 0 0]\n [0 1 0]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_>(less, "[[0 1 0]\n [1 0 0]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_>(more, "[[0 0 1]\n [0 0 1]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<string_>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static3DIntArrayTest) {
    long c_array_3d_1[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array1{c_array_3d_1};
    long c_array_3d_2[2][2][3] = {{{1, 3, 2}, {4, 5, 6}}, {{7, 9, 8}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array2{c_array_3d_2};
    // Elementwise comparison
    {
        Array<bool_, 2, 2, 3> equal = array1 == array2;
        checkArrayRepr<bool_, 2, 2, 3>(equal, "[[[1 0 0]\n [1 1 1]]\n [[1 0 0]\n [1 1 1]]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 2, 2, 3>(less, "[[[0 1 0]\n [0 0 0]]\n [[0 1 0]\n [0 0 0]]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 2, 2, 3>(more, "[[[0 0 1]\n [0 0 0]]\n [[0 0 1]\n [0 0 0]]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<int_, 2, 2, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static3DFloatArrayTest) {
    double c_array_3d_1[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                    {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array1{c_array_3d_1};
    double c_array_3d_2[2][2][3] = {{{1.1, 3.3, 2.2}, {5.5, 4.4, 6.6}},
                                    {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array2{c_array_3d_2};
    // Elementwise comparison
    {
        auto equal = array1 == array2;
        checkArrayRepr<bool_, 2, 2, 3>(equal, "[[[1 0 0]\n [0 0 1]]\n [[1 1 1]\n [1 1 1]]]");
    }
    {
        auto less = array1 < array2;
        checkArrayRepr<bool_, 2, 2, 3>(less, "[[[0 1 0]\n [1 0 0]]\n [[0 0 0]\n [0 0 0]]]");
    }
    {
        auto more = array1 > array2;
        checkArrayRepr<bool_, 2, 2, 3>(more, "[[[0 0 1]\n [0 1 0]]\n [[0 0 0]\n [0 0 0]]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<float_, 2, 2, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

TEST_F(ArrayComparisonTest, static3DStringArrayTest) {
    string_ c_array_3d_1[2][4][3] = {
            {{"str1_1", "str1_2", "str1_3"},
             {"str2_1", "str2_2", "str2_3"},
             {"str3_1", "str3_2", "str3_3"},
             {"str4_1", "str4_2", "str4_3"}},
            {{"str5_1", "str5_2", "str5_3"},
             {"str6_1", "str6_2", "str6_3"},
             {"str7_1", "str7_2", "str7_3"},
             {"str8_1", "str8_2", "str8_3"}}};
    Array<string_, 2, 4, 3> array1{c_array_3d_1};
    string_ c_array_3d_2[2][4][3] = {
            {{"str1_1", "str1_2", "str1_3"},
             {"str2_1", "str2_3", "str2_2"},
             {"str3_1", "str3_2", "str3_3"},
             {"str4_1", "str4_2", "str4_3"}},
            {{"str5_1", "str5_3", "str5_2"},
             {"str6_1", "str6_2", "str6_3"},
             {"str7_1", "str7_2", "str7_3"},
             {"str8_1", "str8_2", "str8_3"}}};
    Array<string_, 2, 4, 3> array2{c_array_3d_2};
    // Elementwise comparison
    {
        Array<bool_, 2, 4, 3> equal = array1 == array2;
        checkArrayRepr<bool_, 2, 4, 3>(equal, "[[[1 1 1]\n [1 0 0]\n [1 1 1]\n [1 1 1]]\n [[1 0 0]\n [1 1 1]\n [1 1 1]\n [1 1 1]]]");
    }
    {
        Array<bool_, 2, 4, 3> less = array1 < array2;
        checkArrayRepr<bool_, 2, 4, 3>(less, "[[[0 0 0]\n [0 1 0]\n [0 0 0]\n [0 0 0]]\n [[0 1 0]\n [0 0 0]\n [0 0 0]\n [0 0 0]]]");
    }
    {
        Array<bool_, 2, 4, 3> more = array1 > array2;
        checkArrayRepr<bool_, 2, 4, 3>(more, "[[[0 0 0]\n [0 0 1]\n [0 0 0]\n [0 0 0]]\n [[0 0 1]\n [0 0 0]\n [0 0 0]\n [0 0 0]]]");
    }
    // Arraywise comparison
    {
        bool equal = array_equal<string_, 2, 4, 3>(array1, array2);
        EXPECT_FALSE(equal);
    }
}

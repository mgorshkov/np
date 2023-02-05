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

#include <iostream>

#include <np/Array.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayMathTest : public ArrayTest {
protected:
};

TEST_F(ArrayMathTest, dynamicEmptyIntArraysTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = add(array1, array2);
    auto equal = array_equal(array, array1);
    EXPECT_TRUE(equal);
}

TEST_F(ArrayMathTest, dynamicEmptyFloatArraysTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = add(array1, array2);
    auto equal = array_equal(array, array1);
    EXPECT_TRUE(equal);
}

TEST_F(ArrayMathTest, static1DIntArraysTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    {
        Array<int_, 3> sum{5, 7, 9};
        auto array = add(array1, array2);
        auto equal = array_equal<int_>(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        auto result = array1.dot(array2);
        EXPECT_EQ(32, result);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysTest) {
    // dynamic
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    {
        Array<int_> sum{5, 7, 9};
        auto array = add(array1, array2);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        auto result = array1.dot(array2);
        EXPECT_EQ(32, result);
    }
}

TEST_F(ArrayMathTest, static1DFloatArraysTest) {
    // static
    Array<float_, 3> array1{1.1, 2.2, 3.3};
    Array<float_, 3> array2{4.4, 5.5, 6.6};
    {
        Array<float_, 3> sum{5.5, 7.7, 9.9};
        auto array = add(array1, array2);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        auto result = array1.dot(array2);
        EXPECT_EQ(38.72, result);
    }
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysTest) {
    // dynamic
    Array<float_> array1{1.1, 2.2, 3.3};
    Array<float_> array2{4.4, 5.5, 6.6};
    {
        Array<float_> sum{5.5, 7.7, 9.9};
        auto array = add(array1, array2);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        auto result = array1.dot(array2);
        EXPECT_EQ(38.72, result);
    }
}

TEST_F(ArrayMathTest, static2DIntArraysTest) {
    long c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array1(c_array_1);
    long c_array_2[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array2(c_array_2);
    auto array = add(array1, array2);
    long c_array_sum[2][3] = {{8, 10, 12}, {14, 16, 18}};
    Array<int_, 2 * 3> sum(c_array_sum);
    auto equal = array_equal(array, sum);
    EXPECT_TRUE(equal);
}

TEST_F(ArrayMathTest, dynamic2DIntArraysTest) {
    long c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1(c_array_1);
    long c_array_2[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array2(c_array_2);
    {
        long c_array_plus[2][3] = {{8, 10, 12}, {14, 16, 18}};
        auto array = add(array1, array2);
        Array<int_> sum{c_array_plus};
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        long c_array_minus[2][3] = {{-6, -6, -6}, {-6, -6, -6}};
        auto array = subtract(array1, array2);
        Array<int_> diff{c_array_minus};
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, static2DFloatArraysTest) {
    double c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array1{c_array_1};
    double c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array2{c_array_2};
    {
        double c_array_plus[2][3] = {{8.8, 11, 13.2}, {14.5, 16.61, 18.72}};
        auto array = add(array1, array2);
        Array<float_, 2 * 3> sum{c_array_plus};
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        double c_array_minus[2][3] = {{-6.6, -6.6, -6.6}, {-5.7, -5.61, -5.52}};
        auto array = subtract(array1, array2);
        Array<float_, 2 * 3> diff{c_array_minus};
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, dynamic2DFloatArraysTest) {
    double c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array1{c_array_1};
    double c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array2{c_array_2};
    {
        double c_array_plus[2][3] = {{8.8, 11, 13.2}, {14.5, 16.61, 18.72}};
        auto array = add(array1, array2);
        Array<float_> sum{c_array_plus};
        auto equal = array_equal<float_>(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        double c_array_minus[2][3] = {{-6.6, -6.6, -6.6}, {-5.7, -5.61, -5.52}};
        auto array = subtract(array1, array2);
        Array<float_> diff{c_array_minus};
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, static3DIntArraysTest) {
    long c_array_1[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array1 = createIntArray<2 * 2 * 3>(c_array_1);
    long c_array_2[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array2 = createIntArray<2 * 2 * 3>(c_array_2);
    {
        long c_array_plus[2][2][3] = {{{2, 4, 6}, {8, 10, 12}}, {{14, 16, 18}, {20, 22, 24}}};
        auto array = add(array1, array2);
        auto sum = createIntArray<2 * 2 * 3>(c_array_plus);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        long c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = subtract(array1, array2);
        auto diff = createIntArray<2 * 2 * 3>(c_array_minus);
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, dynamic3DIntArraysTest) {
    long c_array_1[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array1 = createIntArray<2 * 2 * 3>(c_array_1);
    long c_array_2[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array2 = createIntArray<2 * 2 * 3>(c_array_2);
    {
        long c_array_plus[2][2][3] = {{{2, 4, 6}, {8, 10, 12}}, {{14, 16, 18}, {20, 22, 24}}};
        auto array = add(array1, array2);
        auto sum = createIntArray(c_array_plus);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        long c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = subtract(array1, array2);
        auto diff = createIntArray(c_array_minus);
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, static3DFloatArraysTest) {
    double c_array_1[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array1 = createFloatArray<2 * 2 * 3>(c_array_1);
    double c_array_2[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array2 = createFloatArray<2 * 2 * 3>(c_array_2);
    {
        double c_array_plus[2][2][3] = {{{2.2, 4.4, 6.6}, {8.8, 11, 13.2}}, {{15.4, 17.6, 19.8}, {20.2, 22.22, 24.24}}};
        auto array = add(array1, array2);
        auto sum = createFloatArray(c_array_plus);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        double c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = subtract(array1, array2);
        auto diff = createFloatArray(c_array_minus);
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayMathTest, dynamic3DFloatArraysTest) {
    double c_array_1[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array1 = createFloatArray(c_array_1);
    double c_array_2[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array2 = createFloatArray<2 * 2 * 3>(c_array_2);
    {
        double c_array_plus[2][2][3] = {{{2.2, 4.4, 6.6}, {8.8, 11, 13.2}}, {{15.4, 17.6, 19.8}, {20.2, 22.22, 24.24}}};
        auto array = add(array1, array2);
        auto sum = createFloatArray(c_array_plus);
        auto equal = array_equal(array, sum);
        EXPECT_TRUE(equal);
    }
    {
        double c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = subtract(array1, array2);
        auto diff = createFloatArray(c_array_minus);
        auto equal = array_equal(array, diff);
        EXPECT_TRUE(equal);
    }
}

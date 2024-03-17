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

class ArrayMath3DTest : public ArrayTest {
protected:
};

TEST_F(ArrayMath3DTest, static3DIntArraysTest) {
    long c_array_1[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array1 = createIntArray<2 * 2 * 3>(c_array_1);
    long c_array_2[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array2 = createIntArray<2 * 2 * 3>(c_array_2);
    {
        long c_array_plus[2][2][3] = {{{2, 4, 6}, {8, 10, 12}}, {{14, 16, 18}, {20, 22, 24}}};
        auto array = array1 + array2;
        auto sum = createIntArray<2 * 2 * 3>(c_array_plus);
        compare(array, sum);
    }
    {
        long c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = array1 - array2;
        auto diff = createIntArray<2 * 2 * 3>(c_array_minus);
        compare(array, diff);
    }
    {
        EXPECT_THROW(array1.dot(array2), std::runtime_error);
    }
}

TEST_F(ArrayMath3DTest, dynamic3DIntArraysTest) {
    long c_array_1[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array1 = createIntArray<2 * 2 * 3>(c_array_1);
    long c_array_2[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    auto array2 = createIntArray<2 * 2 * 3>(c_array_2);
    {
        long c_array_plus[2][2][3] = {{{2, 4, 6}, {8, 10, 12}}, {{14, 16, 18}, {20, 22, 24}}};
        auto array = array1 + array2;
        auto sum = createIntArray(c_array_plus);
        compare(array, sum);
    }
    {
        long c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = array1 - array2;
        auto diff = createIntArray(c_array_minus);
        compare(array, diff);
    }
    {
        EXPECT_THROW(array1.dot(array2), std::runtime_error);
    }
}

TEST_F(ArrayMath3DTest, static3DFloatArraysTest) {
    double c_array_1[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array1 = createFloatArray<2 * 2 * 3>(c_array_1);
    double c_array_2[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array2 = createFloatArray<2 * 2 * 3>(c_array_2);
    {
        double c_array_plus[2][2][3] = {{{2.2, 4.4, 6.6}, {8.8, 11, 13.2}}, {{15.4, 17.6, 19.8}, {20.2, 22.22, 24.24}}};
        auto array = array1 + array2;
        auto sum = createFloatArray(c_array_plus);
        compare(array, sum);
    }
    {
        double c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = array1 - array2;
        auto diff = createFloatArray(c_array_minus);
        compare(array, diff);
    }
    {
        EXPECT_THROW(array1.dot(array2), std::runtime_error);
    }
}

TEST_F(ArrayMath3DTest, dynamic3DFloatArraysTest) {
    double c_array_1[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array1 = createFloatArray(c_array_1);
    double c_array_2[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}, {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    auto array2 = createFloatArray<2 * 2 * 3>(c_array_2);
    {
        double c_array_plus[2][2][3] = {{{2.2, 4.4, 6.6}, {8.8, 11, 13.2}}, {{15.4, 17.6, 19.8}, {20.2, 22.22, 24.24}}};
        auto array = array1 + array2;
        auto sum = createFloatArray(c_array_plus);
        compare(array, sum);
    }
    {
        double c_array_minus[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
        auto array = array1 - array2;
        auto diff = createFloatArray(c_array_minus);
        compare(array, diff);
    }
    {
        EXPECT_THROW(array1.dot(array2), std::runtime_error);
    }
}

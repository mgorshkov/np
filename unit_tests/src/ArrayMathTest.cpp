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

TEST_F(ArrayMathTest, dynamicEmptyIntArraysAddTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = add(array1, array2);
    compare(array, array1);
}

TEST_F(ArrayMathTest, dynamicEmptyIntArraysSubtractTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = subtract(array1, array2);
    compare(array, array1);
}

TEST_F(ArrayMathTest, dynamicEmptyFloatArraysAddTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = add(array1, array2);
    compare(array, array1);
}

TEST_F(ArrayMathTest, dynamicEmptyFloatArraysSubtractTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    auto array = subtract(array1, array2);
    compare(array, array1);
}

TEST_F(ArrayMathTest, static1DIntArraysAddTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    auto array = add(array1, array2);
    Array<int_> plus{5, 7, 9};
    compare(array, plus);
}

TEST_F(ArrayMathTest, static1DIntArraysSubtractTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    auto array = subtract(array1, array2);
    Array<float_> minus{-3, -3, -3};
    compare(array, minus);
}

TEST_F(ArrayMathTest, static1DIntArraysDotTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    auto array = array1.dot(array2);
    Array<int_> dot{32};
    compare(array, dot);
}

TEST_F(ArrayMathTest, static1DIntArraysInterpTest) {
    Array<float_, 1> x{2.5};
    Array<int_, 3> xp{1, 2, 3};
    Array<int_, 3> fp{3, 2, 0};
    auto array = interp(x, xp, fp);
    Array<float_> result{1.0};
    compare(array, result);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysAddTest) {
    // dynamic
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    auto array = add(array1, array2);
    Array<int_> plus{5, 7, 9};
    compare(array, plus);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysSubtractTest) {
    // dynamic
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    auto array = subtract(array1, array2);
    Array<int_> minus{-3, -3, -3};
    compare(array, minus);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysDotTest) {
    // dynamic
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};

    auto array = array1.dot(array2);
    Array<int_> dot{32};
    compare(array, dot);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysInterpTest) {
    Array<int_> x{0, 1, 2, 3, 4, 5};
    Array<int_> xp{0, 5};
    Array<float_> fp{-1.0, +1.0};
    auto array = interp(x, xp, fp);
    Array<float_> result{-1.0, -0.6, -0.2, 0.2, 0.6, 1.0};
    compare(array, result);
}

TEST_F(ArrayMathTest, static1DFloatArraysAddTest) {
    // static
    Array<float_, 3> array1{1.1, 2.2, 3.3};
    Array<float_, 3> array2{4.4, 5.5, 6.6};
    auto array = add(array1, array2);
    Array<float_> plus{5.5, 7.7, 9.9};
    compare(array, plus);
}

TEST_F(ArrayMathTest, static1DFloatArraysSubtractTest) {
    // static
    Array<float_, 3> array1{1.1, 2.2, 3.3};
    Array<float_, 3> array2{4.4, 5.5, 6.6};
    auto array = subtract(array1, array2);
    Array<float_> minus{-3.3, -3.3, -3.3};
    compare(array, minus);
}

TEST_F(ArrayMathTest, static1DFloatArraysDotTest) {
    // static
    Array<float_, 3> array1{1.1, 2.2, 3.3};
    Array<float_, 3> array2{4.4, 5.5, 6.6};
    auto array = array1.dot(array2);
    Array<float_> dot{38.72};
    compare(array, dot);
}

TEST_F(ArrayMathTest, static1DFloatArraysInterpTest) {
    {
        Array<float_, 1> x{2.5};
        Array<float_, 3> xp{1.0, 2.0, 3.0};
        Array<float_, 3> fp{3.0, 2.0, 0.0};
        auto array = interp(x, xp, fp);
        Array<float_> result{1.0};
        compare(array, result);
    }
    {
        Array<float_, 6> x{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        Array<float_, 2> xp{0.0, 5.0};
        Array<float_, 2> fp{-1.0, +1.0};
        auto array = interp(x, xp, fp);
        Array<float_> result{-1.0, -0.6, -0.2, 0.2, 0.6, 1.0};
        compare(array, result);
    }
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysAddTest) {
    // dynamic
    Array<float_> array1{1.1, 2.2, 3.3};
    Array<float_> array2{4.4, 5.5, 6.6};
    auto array = add(array1, array2);
    Array<float_> plus{5.5, 7.7, 9.9};
    compare(array, plus);
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysSubtractTest) {
    // dynamic
    Array<float_> array1{1.1, 2.2, 3.3};
    Array<float_> array2{4.4, 5.5, 6.6};
    auto array = subtract(array1, array2);
    Array<float_> minus{-3.3, -3.3, -3.3};
    compare(array, minus);
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysDotTest) {
    Array<float_> array1{1.1, 2.2, 3.3};
    Array<float_> array2{4.4, 5.5, 6.6};
    auto array = array1.dot(array2);
    Array<float_> dot{38.72};
    compare(array, dot);
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysInterpTest) {
    {
        Array<float_> x{2.5};
        Array<float_> xp{1.0, 2.0, 3.0};
        Array<float_> fp{3.0, 2.0, 0.0};
        auto array = interp(x, xp, fp);
        Array<float_> result{1.0};
        compare(array, result);
    }
    {
        Array<float_, 6> x{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        Array<float_> xp{0.0, 5.0};
        Array<float_> fp{-1.0, +1.0};
        auto array = interp(x, xp, fp);
        Array<float_> result{-1.0, -0.6, -0.2, 0.2, 0.6, 1.0};
        compare(array, result);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysAddWithBroadcastTest) {
    long c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1(c_array_1);
    Array<int_> array2{7, 8, 9};
    long c_array_plus[2][3] = {{8, 10, 12}, {11, 13, 15}};
    auto array = add(array1, array2);
    Array<int_> sum{c_array_plus};
    compare(array, sum);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysSubtractWithBroadcastTest) {
    long c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1(c_array_1);
    Array<int_> array2{7, 8, 9};
    long c_array_minus[2][3] = {{-6, -6, -6}, {-3, -3, -3}};
    auto array = subtract(array1, array2);
    Array<int_> diff{c_array_minus};
    compare(array, diff);
}

TEST_F(ArrayMathTest, static1DFloatArrayRoundTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    auto res = round(array);
    Array<float_> sample{1.0, 2.0, 3.0};
    compare(sample, res);
}

TEST_F(ArrayMathTest, dynamic1DFloatArrayRoundTest) {
    // dynamic
    Array<float_> array{1.1, 2.2, 3.3};
    auto res = round(array);
    Array<float_> sample{1.0, 2.0, 3.0};
    compare(sample, res);
}

TEST_F(ArrayMathTest, roundTest) {
    auto res = round(56294995342131.5, 3);
    np::float_ sample = 56294995342131.51;
    EXPECT_DOUBLE_EQ(sample, res);
}

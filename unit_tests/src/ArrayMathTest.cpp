/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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
    {
        auto array = array1 + array2;
        compare(array, array1);
    }
    {
        array1 += array2;
        Array<int_> array{};
        compare(array, array1);
    }
}

TEST_F(ArrayMathTest, dynamicEmptyIntArraysSubtractTest) {
    // dynamic
    Array<int_> array1{};
    Array<int_> array2{};
    {
        auto array = array1 - array2;
        compare(array, array1);
    }
    {
        array1 -= array2;
        Array<int_> array{};
        compare(array, array1);
    }
}

TEST_F(ArrayMathTest, dynamicEmptyFloatArraysAddTest) {
    // dynamic
    Array<float_> array1{};
    Array<float_> array2{};
    {
        auto array = array1 + array2;
        compare(array, array1);
    }
    {
        array1 += array2;
        Array<float_> array{};
        compare(array, array1);
    }
}

TEST_F(ArrayMathTest, dynamicEmptyFloatArraysSubtractTest) {
    // dynamic
    Array<float_> array1{};
    Array<float_> array2{};
    {
        auto array = array1 - array2;
        compare(array, array1);
    }
    {
        array1 -= array2;
        Array<float_> array{};
        compare(array, array1);
    }
}

TEST_F(ArrayMathTest, static1DIntArraysAddTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    Array<int_> plus{5, 7, 9};
    {
        auto array = array1 + array2;
        compare(array, plus);
    }
    {
        array1 += array2;
        compare(array1, plus);
    }
}

TEST_F(ArrayMathTest, static1DIntArraysSubtractTest) {
    // static
    Array<int_, 3> array1{1, 2, 3};
    Array<int_, 3> array2{4, 5, 6};
    Array<int_> minus{-3, -3, -3};
    {
        auto array = array1 - array2;
        compare(array, minus);
    }
    {
        array1 -= array2;
        compare(array1, minus);
    }
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
    Array<int_> array1{1, 2, 3};
    Array<int_> array2{4, 5, 6};
    Array<int_> plus{5, 7, 9};
    {
        auto array = array1 + array2;
        compare(array, plus);
    }
    {
        array1 += array2;
        compare(array1, plus);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysLargeAddTest) {
    // dynamic
    const size_t size = 100000;
    Array<int_> array1 = arange<int_>(size);
    Array<int_> array2 = arange<int_>(size);
    std::vector<int_> vector(size);
    std::iota(vector.begin(), vector.end(), 0);
    for (auto &x: vector) {
        x *= 2;
    }
    NDArrayDynamic<int_> plus{vector};
    {
        auto array = array1 + array2;
        compare(array, plus);
    }
    {
        array1 += array2;
        compare(array1, plus);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysSubtractTest) {
    // dynamic
    Array<int_> array1{1, 2, 3};
    Array<int_> array2{4, 5, 6};
    Array<int_> minus{-3, -3, -3};
    {
        auto array = array1 - array2;
        compare(array, minus);
    }
    {
        array1 -= array2;
        compare(array1, minus);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysLargeSubtractTest) {
    // dynamic
    const size_t size = 100000;
    Array<int_> array1 = arange<int_>(size);
    Array<int_> array2 = arange<int_>(size);
    std::vector<int_> vector(size);
    NDArrayDynamic<int_> minus{vector};
    {
        auto array = array1 - array2;
        compare(array, minus);
    }
    {
        array1 -= array2;
        compare(array1, minus);
    }
}

TEST_F(ArrayMathTest, dynamic1DIntArraysDotTest) {
    // dynamic
    Array<int_> array1{1, 2, 3};
    Array<int_> array2{4, 5, 6};

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
    Array<float_> plus{5.5, 7.7, 9.9};
    {
        auto array = array1 + array2;
        compare(array, plus);
    }
    {
        array1 += array2;
        compare(array1, plus);
    }
}

TEST_F(ArrayMathTest, static1DFloatArraysSubtractTest) {
    // static
    Array<float_, 3> array1{1.1, 2.2, 3.3};
    Array<float_, 3> array2{4.4, 5.5, 6.6};
    Array<float_> minus{-3.3, -3.3, -3.3};
    {
        auto array = array1 - array2;
        compare(array, minus);
    }
    {
        array1 -= array2;
        compare(array1, minus);
    }
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
    Array<float_> plus{5.5, 7.7, 9.9};
    {
        auto array = array1 + array2;
        compare(array, plus);
    }
    {
        array1 += array2;
        compare(array1, plus);
    }
}

TEST_F(ArrayMathTest, dynamic1DFloatArraysSubtractTest) {
    // dynamic
    Array<float_> array1{1.1, 2.2, 3.3};
    Array<float_> array2{4.4, 5.5, 6.6};
    Array<float_> minus{-3.3, -3.3, -3.3};
    {
        auto array = array1 - array2;
        compare(array, minus);
    }
    {
        array1 -= array2;
        compare(array1, minus);
    }
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
    auto array = array1 + array2;
    Array<int_> sum{c_array_plus};
    compare(array, sum);
}

TEST_F(ArrayMathTest, dynamic1DIntArraysSubtractWithBroadcastTest) {
    long c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1(c_array_1);
    Array<int_> array2{7, 8, 9};
    long c_array_minus[2][3] = {{-6, -6, -6}, {-3, -3, -3}};
    auto array = array1 - array2;
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

// Test that expression boolean indexing works correctly.
// This tests the pattern used in monte-carlo PI estimation:
//   dist = rx * rx + ry * ry;
//   inside = dist["dist<1"];
// The expression dist = rx*rx + ry*ry is a BinaryExpression tree.
// operator[]("dist<1") must evaluate the expression and return a valid
// indexed view without dangling pointers.
TEST_F(ArrayMathTest, expressionBooleanIndexingTest) {
    // Create arrays matching the monte-carlo pattern
    const Size n = 10;
    Array<double> rx(Shape{n});
    Array<double> ry(Shape{n});
    for (Size i = 0; i < n; ++i) {
        rx.set(i, (i + 1) * 0.1);
        ry.set(i, (i + 1) * 0.1);
    }

    // rx = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    EXPECT_DOUBLE_EQ(rx.get(0), 0.1);
    EXPECT_DOUBLE_EQ(rx.get(2), 0.3);

    // Test individual operations: rx*rx = [0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0]
    auto rx_sq = rx * rx;
    EXPECT_DOUBLE_EQ(rx_sq.get(0), 0.01);
    EXPECT_DOUBLE_EQ(rx_sq.get(2), 0.09);

    // Test the full expression: dist = rx*rx + ry*ry
    // dist = [0.02, 0.08, 0.18, 0.32, 0.5, 0.72, 0.98, 1.28, 1.62, 2.0]
    auto dist = rx * rx + ry * ry;
    EXPECT_DOUBLE_EQ(dist.get(0), 0.02);
    EXPECT_DOUBLE_EQ(dist.get(2), 0.18);

    // Test boolean indexing on the expression: dist["dist<1"]
    // This is the key test - operator[] on a BinaryExpression must work correctly
    auto inside = dist["dist<1"];

    // dist values: [0.02, 0.08, 0.18, 0.32, 0.5, 0.72, 0.98, 1.28, 1.62, 2.0]
    // dist<1: indices 0, 1, 2, 3, 4, 5, 6 (7 elements)
    EXPECT_EQ(inside.size(), 7);
    EXPECT_DOUBLE_EQ(inside.get(0), 0.02);
    EXPECT_DOUBLE_EQ(inside.get(1), 0.08);
    EXPECT_DOUBLE_EQ(inside.get(2), 0.18);
    EXPECT_DOUBLE_EQ(inside.get(3), 0.32);
    EXPECT_DOUBLE_EQ(inside.get(4), 0.5);
    EXPECT_DOUBLE_EQ(inside.get(5), 0.72);
    EXPECT_DOUBLE_EQ(inside.get(6), 0.98);

    // Test that the expression can be reused after boolean indexing
    // (the cache should not interfere with subsequent operations)
    auto dist2 = rx * rx + ry * ry;
    EXPECT_DOUBLE_EQ(dist2.get(0), 0.02);
    EXPECT_DOUBLE_EQ(dist2.get(9), 2.0);
}

// This tests np::sum with a string condition on an expression tree.
// np::sum("dist<1", dist) should use the fused AVX2 count_if() path
// with no intermediate array allocations, returning the count directly.
TEST_F(ArrayMathTest, expressionSumWithStringTest) {
    // Create arrays matching the monte-carlo pattern
    const Size n = 10;
    Array<double> rx(Shape{n});
    Array<double> ry(Shape{n});
    for (Size i = 0; i < n; ++i) {
        rx.set(i, (i + 1) * 0.1);
        ry.set(i, (i + 1) * 0.1);
    }

    // dist = rx*rx + ry*ry
    // dist = [0.02, 0.08, 0.18, 0.32, 0.5, 0.72, 0.98, 1.28, 1.62, 2.0]
    auto dist = rx * rx + ry * ry;

    // np::sum with condition "dist<1" should count elements < 1.0
    // dist<1: indices 0, 1, 2, 3, 4, 5, 6 (7 elements)
    auto inside = sum("dist<1", dist);
    EXPECT_DOUBLE_EQ(inside, 7.0);

    // Verify it matches the operator[] approach
    auto inside2 = dist["dist<1"];
    EXPECT_EQ(inside2.size(), 7);
    EXPECT_DOUBLE_EQ(inside, static_cast<double>(inside2.size()));

    // Test with a different threshold
    // dist<0.5: indices 0, 1, 2, 3 (4 elements: 0.02, 0.08, 0.18, 0.32)
    auto inside3 = sum("dist<0.5", dist);
    EXPECT_DOUBLE_EQ(inside3, 4.0);

    // Test that the expression can be reused after sum
    auto dist3 = rx * rx + ry * ry;
    EXPECT_DOUBLE_EQ(dist3.get(0), 0.02);
    EXPECT_DOUBLE_EQ(dist3.get(9), 2.0);
}

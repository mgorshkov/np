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

class ArrayMath2DTest : public ArrayTest {
protected:
};

TEST_F(ArrayMath2DTest, dynamic2DIntArraysTest) {
    int_ c_array_1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array1(c_array_1);
    int_ c_array_2[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array2(c_array_2);
    {
        int_ c_array_plus[2][3] = {{8, 10, 12}, {14, 16, 18}};
        auto array = add(array1, array2);
        Array<int_> sum{c_array_plus};
        compare(array, sum);
    }
    {
        int_ c_array_minus[2][3] = {{-6, -6, -6}, {-6, -6, -6}};
        auto array = subtract(array1, array2);
        Array<int_> diff{c_array_minus};
        compare(array, diff);
    }
    {
        int_ c_array_3[3][4] = {{7, 8, 9, 10}, {11, 12, 13, 14}, {15, 16, 17, 18}};
        Array<int_> array3(c_array_3);
        int_ c_array_dot[2][4] = {{74, 80, 86, 92}, {173, 188, 203, 218}};
        Array<int_> dotProduct{c_array_dot};
        auto array = array1.dot(array3);
        compare(array, dotProduct);
    }
}

TEST_F(ArrayMath2DTest, static2DFloatArraysTest) {
    float_ c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array1{c_array_1};
    float_ c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array2{c_array_2};
    {
        float_ c_array_plus[2][3] = {{8.8, 11, 13.2}, {14.5, 16.61, 18.72}};
        auto array = add(array1, array2);
        Array<float_, 2 * 3> plus{c_array_plus};
        compare(array, plus);
    }
    {
        float_ c_array_minus[2][3] = {{-6.6, -6.6, -6.6}, {-5.7, -5.61, -5.52}};
        auto array = subtract(array1, array2);
        Array<float_, 2 * 3> diff{c_array_minus};
        compare(array, diff);
    }
    {
        float_ c_array_3[3][4] = {{7.7, 8.8, 9.9, 10.1}, {11.11, 12.12, 13.13, 14.14}, {15.15, 16.16, 17.17, 18.18}};
        Array<float_, 3 * 4> array3(c_array_3);
        float_ c_array_dot[2][4] = {{82.907, 89.672, 96.437, 102.212}, {194.975, 212.036, 229.097, 242.198}};
        Array<float_> dotProduct{c_array_dot};
        auto array = array1.dot(array3);
        compare(array, dotProduct);
    }
}

TEST_F(ArrayMath2DTest, dynamic2DFloatArraysTest) {
    {
        float_ c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
        Array<float_, 2 * 3> array1{c_array_1};
        float_ c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
        Array<float_, 2 * 3> array2{c_array_2};
        float_ c_array_plus[2][3] = {{8.8, 11, 13.2}, {14.5, 16.61, 18.72}};
        auto array = add(array1, array2);
        Array<float_> plus{c_array_plus};
        compare<float_>(array, plus);
    }
    {
        float_ c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
        Array<float_, 2 * 3> array1{c_array_1};
        float_ c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
        Array<float_, 2 * 3> array2{c_array_2};
        float_ c_array_minus[2][3] = {{-6.6, -6.6, -6.6}, {-5.7, -5.61, -5.52}};
        auto array = subtract(array1, array2);
        Array<float_> diff{c_array_minus};
        compare(array, diff);
    }
    {
        float_ c_array_1[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
        Array<float_, 2 * 3> array1{c_array_1};
        float_ c_array_2[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
        Array<float_, 2 * 3> array2{c_array_2};
        float_ c_array_3[3][4] = {{7.7, 8.8, 9.9, 10.1}, {11.11, 12.12, 13.13, 14.14}, {15.15, 16.16, 17.17, 18.18}};
        Array<float_> array3(c_array_3);
        float_ c_array_dot[2][4] = {{82.907, 89.672, 96.437, 102.212}, {194.975, 212.036, 229.097, 242.198}};
        Array<float_> dotProduct{c_array_dot};
        auto array = array1.dot(array3);
        compare(array, dotProduct);
    }
    {
        float_ c_array_1[3][2] = {{0, -0.5},
                                  {1.4, 1.3},
                                  {2.1, 2.2}};
        Array<float_> array1(c_array_1);
        float_ c_array_2[2] = {0.10731819, 0.21252231};
        Array<float_> array2(c_array_2);
        float_ c_array_dot[3] = {-0.106261155, 0.426524469, 0.692917281};
        Array<float_> dotProduct{c_array_dot};
        auto array = array1.dot(array2);
        compare(array, dotProduct);
    }
    {
        float_ c_array_1[2] = {0.10731819, 0.21252231};
        Array<float_> array1(c_array_1);
        float_ c_array_2[2][3] = {{0, -0.5, 1.4},
                                  {1.3, 2.1, 2.2}};
        Array<float_> array2(c_array_2);
        float_ c_array_dot[3] = {0.276279003, 0.392637756, 0.617794548};
        Array<float_> dotProduct{c_array_dot};
        auto array = array1.dot(array2);
        compare(array, dotProduct);
    }
}

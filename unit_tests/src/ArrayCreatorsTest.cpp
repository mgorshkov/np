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
#include <np/Math.hpp>
#include <np/Creators.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayCreatorsTest : public ArrayTest {
protected:
};

TEST_F(ArrayCreatorsTest, defaultDynamicCreationTest) {
    // dynamic
    checkArrayRepr<int_>(Array<int_>{}, "[]");

    checkArrayRepr<float_>(Array<float_>{}, "[]");

    checkArrayRepr<string_>(Array<string_>{}, "[]");
}

TEST_F(ArrayCreatorsTest, fromInitializerListStaticCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    // static
    checkArrayRepr<int_, 3>(Array<int_, 3>{1, 2, 3}, "[1 2 3]");

    checkArrayRepr<float_, 3>(Array<float_, 3>{1.1, 2.2, 3.3}, "[1.1 2.2 3.3]");

    checkArrayRepr<string_, 3>(Array<string_, 3>{"str1", "str2", "str3"}, "[\"str1\" \"str2\" \"str3\"]");
}

TEST_F(ArrayCreatorsTest, fromInitializerListDynamicCreationTest) {
    // dynamic
    checkArrayRepr<int_>(Array<int_>{1, 2, 3}, "[1 2 3]");

    checkArrayRepr<float_>(Array<float_>{1.1, 2.2, 3.3}, "[1.1 2.2 3.3]");

    checkArrayRepr<string_>(Array<string_>{"str1", "str2", "str3"}, "[\"str1\" \"str2\" \"str3\"]");
}

TEST_F(ArrayCreatorsTest, from1DCArrayStaticCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    // static
    long c_array_1d_int[3] = {1, 2, 3};
    Array<int_, 3> array_1d_int = c_array_1d_int;
    checkArrayRepr<int_, 3>(array_1d_int, "[1 2 3]");

    double c_array_1d_double[3] = {1.1, 2.2, 3.3};
    Array<float_, 3> array_1d_double = c_array_1d_double;
    checkArrayRepr<float_, 3>(array_1d_double, "[1.1 2.2 3.3]");
}

TEST_F(ArrayCreatorsTest, from1DCArrayDynamicCreationTest) {
    // dynamic
    long c_array_1d_int[3] = {1, 2, 3};
    Array<int_> array_1d_int = c_array_1d_int;
    checkArrayRepr<int_>(array_1d_int, "[1 2 3]");

    double c_array_1d_double[3] = {1.1, 2.2, 3.3};
    Array<float_> array_1d_double = c_array_1d_double;
    checkArrayRepr<float_>(array_1d_double, "[1.1 2.2 3.3]");
}

TEST_F(ArrayCreatorsTest, from2DCArrayCreationTest) {
    /*
    print(np.array(([1,2,3],[4,5,6])))
    [[1 2 3]
     [4 5 6]]
    */
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array_2d{c_array_2d};
    checkArrayRepr<int_, 2, 3>(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(ArrayCreatorsTest, from3DCArrayCreationTest) {
    /*
     >>> print(np.array((([1,2,3],[4,5,6]),([7,8,9],[10,11,12]))))
     [[[ 1  2  3]
       [ 4  5  6]]

      [[ 7  8  9]
       [10 11 12]]]
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array_3d{c_array_3d};
    checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, from1DStdArrayCreationTest) {
    std::array std_array_1d{1L, 2L, 3L};
    Array<int_, 3> array_1d{std_array_1d};
    checkArrayRepr<int_, 3>(array_1d, "[1 2 3]");
}

TEST_F(ArrayCreatorsTest, from2DStdArrayCreationTest) {
    std::array<std::array<long, 3>, 2> std_array_2d = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_, 2, 3> array_2d{std_array_2d};
    checkArrayRepr<int_, 2, 3>(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(ArrayCreatorsTest, from3DStdArrayCreationTest) {
    std::array<std::array<long, 3>, 2> std_array_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::array<std::array<long, 3>, 2> std_array_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::array<std::array<std::array<long, 3>, 2>, 2> std_array_3d = {std_array_2d_1, std_array_2d_2};
    Array<int_, 2, 2, 3> array_3d{std_array_3d};
    checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, from1DStdVectorCreationTest) {
    Array<int_, 3> array_1d{std::vector{1L, 2L, 3L}};
    checkArrayRepr<int_, 3>(array_1d, "[1 2 3]");
}

TEST_F(ArrayCreatorsTest, from2DStdVectorCreationTest) {
    Array<int_, 2, 3> array_2d{std::vector{{std::vector{1L, 2L, 3L}, std::vector{4L, 5L, 6L}}}};
    checkArrayRepr<int_, 2, 3>(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(ArrayCreatorsTest, from3DStdVectorCreationTest) {
    std::vector<std::vector<long>> std_vector_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::vector<std::vector<long>> std_vector_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::vector<std::vector<std::vector<int_>>> std_vector_3d = {std_vector_2d_1, std_vector_2d_2};
    Array<int_, 2, 2, 3> array_3d{std_vector_3d};
    checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, fillCreationTest) {
    {
        Array<int_, 3> array_1d{42};
        checkArrayRepr<int_, 3>(array_1d, "[42 42 42]");
    }
    {
        Array<int_, 3> array_1d(42);
        checkArrayRepr<int_, 3>(array_1d, "[42 42 42]");
    }
    {
        Array<int_, 2, 3> array_2d{42};
        checkArrayRepr<int_, 2, 3>(array_2d, "[[42 42 42]\n [42 42 42]]");
    }
    {
        Array<int_, 2, 3> array_2d(42);
        checkArrayRepr<int_, 2, 3>(array_2d, "[[42 42 42]\n [42 42 42]]");
    }
    {
        Array<int_, 2, 2, 3> array_3d{42};
        checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[42 42 42]\n [42 42 42]]\n [[42 42 42]\n [42 42 42]]]");
    }
    {
        Array<int_, 2, 2, 3> array_3d(42);
        checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[42 42 42]\n [42 42 42]]\n [[42 42 42]\n [42 42 42]]]");
    }
}

// Create an NDArray of zeros
TEST_F(ArrayCreatorsTest, testZeros) {
    auto array_1d = zeros<int_, 3>();
    checkArrayRepr<int_, 3>(array_1d, "[0 0 0]");
    
    auto array_2d = zeros<int_, 2, 3>();
    checkArrayRepr<int_, 2, 3>(array_2d, "[[0 0 0]\n [0 0 0]]");
    
    auto array_3d = zeros<int_, 2, 2, 3>();
    checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[0 0 0]\n [0 0 0]]\n [[0 0 0]\n [0 0 0]]]");
}

// Create an NDArray of ones
TEST_F(ArrayCreatorsTest, testOnes) {
    auto array_1d = ones<int_, 3>();
    checkArrayRepr<int_, 3>(array_1d, "[1 1 1]");
    
    auto array_2d = ones<int_, 2, 3>();
    checkArrayRepr<int_, 2, 3>(array_2d, "[[1 1 1]\n [1 1 1]]");
    
    auto array_3d = ones<int_, 2, 2, 3>();
    checkArrayRepr<int_, 2, 2, 3>(array_3d, "[[[1 1 1]\n [1 1 1]]\n [[1 1 1]\n [1 1 1]]]");
}

// Create 1D array with regularly incrementing values
TEST_F(ArrayCreatorsTest, testARange) {
    auto array_1 = arange<int_, 10>();
    checkArrayRepr<int_, 10>(array_1, "[0 1 2 3 4 5 6 7 8 9]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_2 = arange<int_, 10, 25>();
    checkArrayRepr<int_, 15>(array_2, "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_3 = arange<int_, 10, 25, 5>();
    checkArrayRepr<int_, 3>(array_3, "[10 15 20]");
}

// Create an NDArray of evenly spaced values (number of samples)
TEST_F(ArrayCreatorsTest, testLinspace) {
    auto array_1 = linspace<int_, 11>(0, 10);
    checkArrayRepr<int_, 11>(array_1, "[0 1 2 3 4 5 6 7 8 9 10]");
    auto array_2 = linspace<float_, 10>(-3.0, 3.0);
    checkArrayRepr<float_, 10>(array_2, "[-3 -2.3333333 -1.6666667 -1 -0.33333333 0.33333333 1 1.6666667 2.3333333 3]");
}

// Create a constant NDArray
TEST_F(ArrayCreatorsTest, testFullStatic) {
    auto array = full<int_, 2, 2>(7);
    checkArrayRepr<int_, 2, 2>(array, "[[7 7]\n [7 7]]");
}

TEST_F(ArrayCreatorsTest, testFullDynamic) {
    Shape shape{2, 2};
    auto array = full<int_>(7, shape);
    checkArrayRepr<int_>(array, "[[7 7]\n [7 7]]");
}

// Create an identity matrix
TEST_F(ArrayCreatorsTest, testEyeStatic) {
    auto array = eye<int_, 2>();
    checkArrayRepr<int_, 2, 2>(array, "[[1 0]\n [0 1]]");
}

TEST_F(ArrayCreatorsTest, testEyeDynamic) {
    auto array = eye<int_>(2);
    checkArrayRepr<int_>(array, "[[1 0]\n [0 1]]");
}

// Create an NDArray with random values
TEST_F(ArrayCreatorsTest, testRandomStatic) {
    auto array = static_cast<NDArrayStatic<int_, 2, 2>>(random::rand<int_, 2, 2>());
    Shape shape{2, 2};
    checkArrayShape<int_, 2, 2>(array, shape);
}

TEST_F(ArrayCreatorsTest, testRandomDynamic) {
    Shape shape{2, 2};
    auto array = random::rand<int_>(shape);
    checkArrayShape<int_>(array, shape);
}

// Create an empty NDArray
TEST_F(ArrayCreatorsTest, testEmptyStatic) {
    auto array = empty<int_, 2, 2>();
    Shape shape{2, 2};
    checkArrayShape<int_, 2, 2>(array, shape);
}

TEST_F(ArrayCreatorsTest, testEmptyDynamic) {
    Shape shape{2, 2};
    auto array = empty<int_>(shape);
    checkArrayShape<int_>(array, shape);
}

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

#include <np/Array.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayCreatorsTest : public ArrayTest {
protected:
};

TEST_F(ArrayCreatorsTest, defaultDynamicCreationTest) {
    // dynamic
    auto ar = Array<int_>{};
    checkArrayRepr<int_>(ar, "[]");

    checkArrayRepr<float_>(Array<float_>{}, "[]");

    checkArrayRepr<string_>(Array<string_>{}, "[]");
}

TEST_F(ArrayCreatorsTest, fromInitializerListStaticCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    // static
    checkArrayRepr(Array<int_, 3>{1, 2, 3}, "[1 2 3]");

    checkArrayRepr(Array<float_, 3>{1.1, 2.2, 3.3}, "[1.1 2.2 3.3]");

    checkArrayRepr(Array<string_, 3>{"str1", "str2", "str3"}, R"(["str1" "str2" "str3"])");
}

TEST_F(ArrayCreatorsTest, fromInitializerListDynamicCreationTest) {
    // dynamic
    checkArrayRepr(Array<int_>{1, 2, 3}, "[1 2 3]");

    checkArrayRepr(Array<float_>{1.1, 2.2, 3.3}, "[1.1 2.2 3.3]");

    checkArrayRepr(Array<string_>{"str1", "str2", "str3"}, R"(["str1" "str2" "str3"])");
}

TEST_F(ArrayCreatorsTest, from1DCArrayStaticCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    // static
    long c_array_1d_int[3] = {1, 2, 3};
    Array<int_, 3> array_1d_int{c_array_1d_int};
    checkArrayRepr(array_1d_int, "[1 2 3]");

    double c_array_1d_double[3] = {1.1, 2.2, 3.3};
    Array<float_, 3> array_1d_double{c_array_1d_double};
    checkArrayRepr(array_1d_double, "[1.1 2.2 3.3]");
}

TEST_F(ArrayCreatorsTest, from1DCArrayDynamicCreationTest) {
    // dynamic
    long c_array_1d_int[3] = {1, 2, 3};
    Array<int_> array_1d_int{c_array_1d_int};
    checkArrayRepr<int_>(array_1d_int, "[1 2 3]");

    double c_array_1d_double[3] = {1.1, 2.2, 3.3};
    Array<float_> array_1d_double{c_array_1d_double};
    checkArrayRepr<float_>(array_1d_double, "[1.1 2.2 3.3]");
}

TEST_F(ArrayCreatorsTest, from2DCArrayCreationTest) {
    /*
    print(np.array(([1,2,3],[4,5,6])))
    [[1 2 3]
     [4 5 6]]
    */
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array_2d{c_array_2d};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
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
    Array<int_, 2 * 2 * 3> array_3d{c_array_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, from4DCArrayCreationTest) {
    /*
     >>> print(np.array((([[1, 101], [2, 102], [3, 103]],[[4, 104], [5, 105], [6, 106]]),([[7, 107], [8, 108], [9, 109]],[[10, 110], [11, 111], [12, 112]]))))
     [[[[  1 101]
        [  2 102]
        [  3 103]]

       [[  4 104]
        [  5 105]
        [  6 106]]]

      [[[  7 107]
        [  8 108]
        [  9 109]]

       [[ 10 110]
        [ 11 111]
        [ 12 112]]]]
     */
    long c_array_4d[2][2][3][2] = {{{{1, 101}, {2, 102}, {3, 103}}, {{4, 104}, {5, 105}, {6, 106}}},
                                   {{{7, 107}, {8, 108}, {9, 109}}, {{10, 110}, {11, 111}, {12, 112}}}};
    Array<int_, 2 * 2 * 3 * 2> array_4d{c_array_4d};
    checkArrayRepr(array_4d, "[[[[1 101]\n [2 102]\n [3 103]]\n [[4 104]\n [5 105]\n [6 106]]]\n [[[7 107]\n [8 108]\n [9 109]]\n [[10 110]\n [11 111]\n [12 112]]]]");
}

TEST_F(ArrayCreatorsTest, from1DStdArrayCreationTest) {
    std::array std_array_1d{1L, 2L, 3L};
    Array<int_, 3> array_1d{std_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(ArrayCreatorsTest, from2DStdArrayCreationTest) {
    std::array<std::array<long, 3>, 2> std_array_2d = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_, 2 * 3> array_2d{std_array_2d};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(ArrayCreatorsTest, from3DStdArrayCreationTest) {
    std::array<std::array<long, 3>, 2> std_array_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::array<std::array<long, 3>, 2> std_array_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::array<std::array<std::array<long, 3>, 2>, 2> std_array_3d = {std_array_2d_1, std_array_2d_2};
    Array<int_, 2 * 2 * 3> array_3d{std_array_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, from4DStdArrayCreationTest) {
    std::array<long, 2> a1{1, 101};
    std::array<long, 2> a2{2, 102};
    std::array<long, 2> a3{3, 103};
    std::array<long, 2> a4{4, 104};
    std::array<long, 2> a5{5, 105};
    std::array<long, 2> a6{6, 106};
    std::array<long, 2> a7{7, 107};
    std::array<long, 2> a8{8, 108};
    std::array<long, 2> a9{9, 109};
    std::array<long, 2> a10{10, 110};
    std::array<long, 2> a11{11, 111};
    std::array<long, 2> a12{12, 112};
    std::array<std::array<std::array<long, 2>, 3>, 2> std_array_3d_1 = {{{a1, a2, a3}, {a4, a5, a6}}};
    std::array<std::array<std::array<long, 2>, 3>, 2> std_array_3d_2 = {{{a7, a8, a9}, {a10, a11, a12}}};
    std::array<std::array<std::array<std::array<long, 2>, 3>, 2>, 2> std_array_4d = {std_array_3d_1, std_array_3d_2};
    Array<int_, 2 * 2 * 3 * 2> array_4d{std_array_4d};
    checkArrayRepr(array_4d, "[[[[1 101]\n [2 102]\n [3 103]]\n [[4 104]\n [5 105]\n [6 106]]]\n [[[7 107]\n [8 108]\n [9 109]]\n [[10 110]\n [11 111]\n [12 112]]]]");
}

TEST_F(ArrayCreatorsTest, from1DStdVectorCreationTest) {
    Array<int_, 3> array_1d{std::vector{1L, 2L, 3L}};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(ArrayCreatorsTest, from2DStdVectorCreationTest) {
    Array<int_, 2 * 3> array_2d{std::vector{{std::vector{1L, 2L, 3L}, std::vector{4L, 5L, 6L}}}};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(ArrayCreatorsTest, from3DStdVectorCreationTest) {
    std::vector<std::vector<long>> std_vector_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::vector<std::vector<long>> std_vector_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::vector<std::vector<std::vector<int_>>> std_vector_3d = {std_vector_2d_1, std_vector_2d_2};
    Array<int_, 2 * 2 * 3> array_3d{std_vector_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(ArrayCreatorsTest, from4DStdVectorCreationTest) {
    std::vector<long> a1{1, 101};
    std::vector<long> a2{2, 102};
    std::vector<long> a3{3, 103};
    std::vector<long> a4{4, 104};
    std::vector<long> a5{5, 105};
    std::vector<long> a6{6, 106};
    std::vector<long> a7{7, 107};
    std::vector<long> a8{8, 108};
    std::vector<long> a9{9, 109};
    std::vector<long> a10{10, 110};
    std::vector<long> a11{11, 111};
    std::vector<long> a12{12, 112};
    std::vector<std::vector<std::vector<long>>> std_vector_3d_1 = {{{a1, a2, a3}, {a4, a5, a6}}};
    std::vector<std::vector<std::vector<long>>> std_vector_3d_2 = {{{a7, a8, a9}, {a10, a11, a12}}};
    std::vector<std::vector<std::vector<std::vector<long>>>> std_vector_4d = {std_vector_3d_1, std_vector_3d_2};
    Array<int_, 2 * 2 * 3 * 2> array_4d{std_vector_4d};
    checkArrayRepr(array_4d, "[[[[1 101]\n [2 102]\n [3 103]]\n [[4 104]\n [5 105]\n [6 106]]]\n [[[7 107]\n [8 108]\n [9 109]]\n [[10 110]\n [11 111]\n [12 112]]]]");
}

TEST_F(ArrayCreatorsTest, fill1DCreationTest) {
    {
        Array<int_> array_1d{Shape{3}, 42};
        checkArrayRepr(array_1d, "[42 42 42]");
    }
    {
        Array<int_, 3> array_1d{Shape{3}, 42};
        checkArrayRepr(array_1d, "[42 42 42]");
    }
}

TEST_F(ArrayCreatorsTest, fill2DCreationTest) {
    {
        Array<int_> array_2d{Shape{2, 3}, 42};
        checkArrayRepr(array_2d, "[[42 42 42]\n [42 42 42]]");
    }
    {
        Array<int_, 2 * 3> array_2d{Shape{2, 3}, 42};
        checkArrayRepr(array_2d, "[[42 42 42]\n [42 42 42]]");
    }
}

TEST_F(ArrayCreatorsTest, fill3DCreationTest) {
    {
        Array<int_> array_3d{Shape{2, 2, 3}, 42};
        checkArrayRepr(array_3d, "[[[42 42 42]\n [42 42 42]]\n [[42 42 42]\n [42 42 42]]]");
    }
    {
        Array<int_, 2 * 2 * 3> array_3d{Shape{2, 2, 3}, 42};
        checkArrayRepr(array_3d, "[[[42 42 42]\n [42 42 42]]\n [[42 42 42]\n [42 42 42]]]");
    }
}

// Create an NDArray of zeros
TEST_F(ArrayCreatorsTest, test1DZerosStatic) {
    auto array_1d = zeros<int_, 3>();
    checkArrayRepr(array_1d, "[0 0 0]");
}

TEST_F(ArrayCreatorsTest, test1DZerosDynamic) {
    auto array_1d = zeros<int_>(Shape{3});
    checkArrayRepr(array_1d, "[0 0 0]");
}

TEST_F(ArrayCreatorsTest, test2DZerosStatic) {
    auto array_2d = zeros<int_, 2, 3>();
    checkArrayRepr(array_2d, "[[0 0 0]\n [0 0 0]]");
}

TEST_F(ArrayCreatorsTest, test2DZerosDynamic) {
    auto array_2d = zeros<int_>(Shape{2, 3});
    checkArrayRepr(array_2d, "[[0 0 0]\n [0 0 0]]");
}

TEST_F(ArrayCreatorsTest, test3DZerosStatic) {
    auto array_3d = zeros<int_, 2, 2, 3>();
    checkArrayRepr(array_3d, "[[[0 0 0]\n [0 0 0]]\n [[0 0 0]\n [0 0 0]]]");
}

TEST_F(ArrayCreatorsTest, test3DZerosDynamic) {
    auto array_3d = zeros<int_>(Shape{2, 2, 3});
    checkArrayRepr(array_3d, "[[[0 0 0]\n [0 0 0]]\n [[0 0 0]\n [0 0 0]]]");
}

// Create an NDArray of ones
TEST_F(ArrayCreatorsTest, test1DOnesStatic) {
    auto array_1d = ones<int_, 3>();
    checkArrayRepr(array_1d, "[1 1 1]");
}

TEST_F(ArrayCreatorsTest, test1DOnesDynamic) {
    auto array_1d = ones<int_>(Shape{3});
    checkArrayRepr(array_1d, "[1 1 1]");
}

TEST_F(ArrayCreatorsTest, test2DOnesStatic) {
    auto array_2d = ones<int_, 2, 3>();
    checkArrayRepr(array_2d, "[[1 1 1]\n [1 1 1]]");
}

TEST_F(ArrayCreatorsTest, test2DOnesDynamic) {
    auto array_2d = ones<int_>(Shape{2, 3});
    checkArrayRepr(array_2d, "[[1 1 1]\n [1 1 1]]");
}

TEST_F(ArrayCreatorsTest, test3DOnesStatic) {
    auto array_3d = ones<int_, 2, 2, 3>();
    checkArrayRepr(array_3d, "[[[1 1 1]\n [1 1 1]]\n [[1 1 1]\n [1 1 1]]]");
}

TEST_F(ArrayCreatorsTest, test3DOnesDynamic) {
    auto array_3d = ones<int_>(Shape{2, 2, 3});
    checkArrayRepr(array_3d, "[[[1 1 1]\n [1 1 1]]\n [[1 1 1]\n [1 1 1]]]");
}

// Create 1D array with regularly incrementing values
TEST_F(ArrayCreatorsTest, testARangeStatic) {
    auto array_1 = arange<int_, 10>();
    checkArrayRepr(array_1, "[0 1 2 3 4 5 6 7 8 9]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_2 = arange<int_, 10, 25>();
    checkArrayRepr(array_2, "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_3 = arange<int_, 10, 25, 5>();
    checkArrayRepr(array_3, "[10 15 20]");
}

TEST_F(ArrayCreatorsTest, testARangeDynamic) {
    auto array_1 = arange<int_>(10);
    checkArrayRepr(array_1, "[0 1 2 3 4 5 6 7 8 9]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_2 = arange<int_>(10, 25);
    checkArrayRepr(array_2, "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]");
    // Create an NDArray of evenly spaced values (step value)
    auto array_3 = arange<int_>(10, 25, 5);
    checkArrayRepr(array_3, "[10 15 20]");
}

// Create an NDArray of evenly spaced values (number of samples)
TEST_F(ArrayCreatorsTest, testLinspaceStatic) {
    auto array_1 = linspace<int_, 11UL>(0, 10);
    checkArrayRepr(array_1, "[0 1 2 3 4 5 6 7 8 9 10]");
    auto array_2 = linspace<float_, 10UL>(-3.0, 3.0);
    checkArrayRepr(array_2, "[-3 -2.3333333 -1.6666667 -1 -0.33333333 0.33333333 1 1.6666667 2.3333333 3]");
}

TEST_F(ArrayCreatorsTest, testLinspaceDynamic) {
    auto array_1 = linspace<int_>(0UL, 10UL, 11);
    checkArrayRepr(array_1, "[0 1 2 3 4 5 6 7 8 9 10]");
    auto array_2 = linspace<float_>(-3.0, 3.0, 10);
    checkArrayRepr(array_2, "[-3 -2.3333333 -1.6666667 -1 -0.33333333 0.33333333 1 1.6666667 2.3333333 3]");
}

// Create a constant NDArray
TEST_F(ArrayCreatorsTest, testFullStatic) {
    auto array = full<int_, 2, 2>(7);
    checkArrayRepr(array, "[[7 7]\n [7 7]]");
}

TEST_F(ArrayCreatorsTest, testFullDynamic) {
    Shape shape{2, 2};
    auto array = full<int_>(7, shape);
    checkArrayRepr(array, "[[7 7]\n [7 7]]");
}

// Create an identity matrix
TEST_F(ArrayCreatorsTest, testEyeStatic) {
    auto array = eye<int_, 2>();
    checkArrayRepr(array, "[[1 0]\n [0 1]]");
}

TEST_F(ArrayCreatorsTest, testEyeDynamic) {
    auto array = eye<int_>(2);
    checkArrayRepr(array, "[[1 0]\n [0 1]]");
}

// Create an NDArray with random values
TEST_F(ArrayCreatorsTest, testRandomStatic) {
    auto array = random::rand<float_, 2, 2>();
    Shape shape{2, 2};
    checkArrayShape(array, shape);
}

TEST_F(ArrayCreatorsTest, testRandomDynamic) {
    Shape shape{2, 2};
    auto array = random::rand<float_>(shape);
    checkArrayShape(array, shape);
}

// Create an empty NDArray
TEST_F(ArrayCreatorsTest, testEmptyStatic) {
    auto array = empty<int_, 2, 2>();
    Shape shape{2, 2};
    checkArrayShape(array, shape);
}

TEST_F(ArrayCreatorsTest, testEmptyDynamic) {
    Shape shape{2, 2};
    auto array = empty<int_>(shape);
    checkArrayShape(array, shape);
}

TEST_F(ArrayCreatorsTest, testExtendArray) {
    Shape shape{2, 2};
    auto array = empty<int_>(shape);
    array.push_back(1);
    checkArrayRepr(array, "[0 0 0 0 1]");
}

TEST_F(ArrayCreatorsTest, testTransformArray) {
    auto array = eye<int_>(2);
    checkArrayRepr(array, "[[1 0]\n [0 1]]");
    Array<int_> output{};
    std::transform(array.cbegin(), array.cend(),
                   std::back_inserter(output), [](int_ i) { return i + 1; });
    checkArrayRepr(output, "[2 1 1 2]");
}

TEST_F(ArrayCreatorsTest, testDiagEmpty) {
    {
        auto v = Array<int_>{};
        int k = 0;
        auto array = diag(v, k);
        checkArrayRepr(array, "[]");
    }
    {
        auto v = Array<int_>{};
        int k = 3;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[0 0 0]\n"
                              " [0 0 0]\n"
                              " [0 0 0]]");
    }
    {
        auto v = Array<int_>{};
        int k = -4;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[0 0 0 0]\n"
                              " [0 0 0 0]\n"
                              " [0 0 0 0]\n"
                              " [0 0 0 0]]");
    }
}

TEST_F(ArrayCreatorsTest, testDiag1D) {
    {
        auto v = Array<int_>{1, 2, 3};
        int k = 0;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[1 0 0]\n"
                              " [0 2 0]\n"
                              " [0 0 3]]");
    }
    {
        auto v = Array<int_>{1, 2, 3};
        int k = 1;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[0 1 0 0]\n"
                              " [0 0 2 0]\n"
                              " [0 0 0 3]\n"
                              " [0 0 0 0]]");
    }
    {
        auto v = Array<int_>{1, 2, 3};
        int k = 2;
        auto array = diag(v, k);
        checkArrayRepr(array,
                       "[[0 0 1 0 0]\n"
                       " [0 0 0 2 0]\n"
                       " [0 0 0 0 3]\n"
                       " [0 0 0 0 0]\n"
                       " [0 0 0 0 0]]");
    }
    {
        auto v = Array<int_>{1, 2, 3};
        int k = -1;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[0 0 0 0]\n"
                              " [1 0 0 0]\n"
                              " [0 2 0 0]\n"
                              " [0 0 3 0]]");
    }
    {
        auto v = Array<int_>{1, 2, 3};
        int k = -2;
        auto array = diag(v, k);
        checkArrayRepr(array,
                       "[[0 0 0 0 0]\n"
                       " [0 0 0 0 0]\n"
                       " [1 0 0 0 0]\n"
                       " [0 2 0 0 0]\n"
                       " [0 0 3 0 0]]");
    }
    {
        auto v = Array<int_>{1, 2, 3};
        int k = -20;
        auto array = diag(v, k);
        checkArrayRepr(array, "[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
                              " [0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]");
    }
}

TEST_F(ArrayCreatorsTest, testDiag2D) {
    {
        int_ array[2][4] = {{1, 2, 3, 4},
                            {5, 6, 7, 8}};
        auto v = Array<int_>{array};
        int k = 0;
        checkArrayRepr(diag(v, k), "[1 6]");
    }
    {
        int_ array[2][4] = {{1, 2, 3, 4},
                            {5, 6, 7, 8}};
        auto v = Array<int_>{array};
        int k = 2;
        checkArrayRepr(diag(v, k), "[3 8]");
    }
    {
        int_ array[2][4] = {{1, 2, 3, 4},
                            {5, 6, 7, 8}};
        auto v = Array<int_>{array};
        int k = -1;
        checkArrayRepr(diag(v, k), "[5]");
    }
    {
        int_ array[2][4] = {{1, 2, 3, 4},
                            {5, 6, 7, 8}};
        auto v = Array<int_>{array};
        int k = -2;
        checkArrayRepr(diag(v, k), "[]");
    }
}

TEST_F(ArrayCreatorsTest, testDiag3D) {
    int_ array[2][2][4] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
    auto v = Array<int_>{array};
    int k = 2;
    EXPECT_THROW(diag(v, k), std::runtime_error);
}

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

#include <sstream>
#include <gtest/gtest.h>

#include <np/Shape.hpp>

#include <np/ndarray/dynamic/internal/NDArrayDynamicInternal.hpp>
#include <np/ndarray/dynamic/internal/NDArrayDynamicInternalStreamIoImpl.hpp>

using namespace np::ndarray::array_dynamic::internal;

class NDArrayDynamicInternalTest : public ::testing::Test {
protected:
    template <typename DType>
    static void checkArrayRepr(const NDArrayDynamicInternal<DType>& array, const char* repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }
};

TEST_F(NDArrayDynamicInternalTest, defaultCreationTest) {
    /*
     >> print(np.array([]))
    []
     */
    checkArrayRepr(NDArrayDynamicInternal<int>{}, "[]");
}

TEST_F(NDArrayDynamicInternalTest, fromInitializerListCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    checkArrayRepr(NDArrayDynamicInternal<int>{1, 2, 3}, "[1 2 3]");
}

TEST_F(NDArrayDynamicInternalTest, assignmentOperatorTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    NDArrayDynamicInternal<int> array{1, 2, 3};
    NDArrayDynamicInternal<int> arrayCopy;
    arrayCopy = array;
    checkArrayRepr(arrayCopy, "[1 2 3]");
}

TEST_F(NDArrayDynamicInternalTest, fromCArray1DCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    int c_array_1d[3] = {1, 2, 3};
    NDArrayDynamicInternal<int> array_1d{c_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicInternalTest, fromCArray2DCreationTest) {
    /*
    print(np.array(([1,2,3],[4,5,6])))
    [[1 2 3]
     [4 5 6]]
    */
    int c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    NDArrayDynamicInternal<int> array_2d{c_array_2d};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(NDArrayDynamicInternalTest, fromCArray3DCreationTest) {
    /*
     >>> print(np.array((([1,2,3],[4,5,6]),([7,8,9],[10,11,12]))))
     [[[ 1  2  3]
       [ 4  5  6]]

      [[ 7  8  9]
       [10 11 12]]]
     */
    int c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    NDArrayDynamicInternal<int> array_3d{c_array_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdArray1DCreationTest) {
    std::array std_array_1d{1, 2, 3};
    NDArrayDynamicInternal<int> array_1d{std_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdArray2DCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d = {{{1, 2, 3}, {4, 5, 6}}};
    NDArrayDynamicInternal<int> array_2d{std_array_2d};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdArray3DCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::array<std::array<int, 3>, 2> std_array_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::array<std::array<std::array<int, 3>, 2>, 2> std_array_3d = {std_array_2d_1, std_array_2d_2};
    NDArrayDynamicInternal<int> array_3d{std_array_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdVector1DCreationTest) {
    NDArrayDynamicInternal<int> array_1d{std::vector{1, 2, 3}};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdVector2DCreationTest) {
    NDArrayDynamicInternal<int> array_2d{std::vector{{std::vector{1, 2, 3}, std::vector{4, 5, 6}}}};
    checkArrayRepr(array_2d, "[[1 2 3]\n [4 5 6]]");
}

TEST_F(NDArrayDynamicInternalTest, fromStdVector3DCreationTest) {
    std::vector<std::vector<int>> std_vector_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::vector<std::vector<int>> std_vector_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::vector<std::vector<std::vector<int>>> std_vector_3d = {std_vector_2d_1, std_vector_2d_2};
    NDArrayDynamicInternal<int> array_3d{std_vector_3d};
    checkArrayRepr(array_3d, "[[[1 2 3]\n [4 5 6]]\n [[7 8 9]\n [10 11 12]]]");
}

TEST_F(NDArrayDynamicInternalTest, fill1DCreationTest) {
    {
        NDArrayDynamicInternal<int> array_1d{42};
        checkArrayRepr(array_1d, "[42]");
    }
    {
        NDArrayDynamicInternal<int> array_1d(np::Shape{3}, 42);
        checkArrayRepr(array_1d, "[42 42 42]");
    }
}

TEST_F(NDArrayDynamicInternalTest, fill2DCreationTest) {
    {
        NDArrayDynamicInternal<int> array_2d{np::Shape{2, 3}, 42};
        checkArrayRepr(array_2d, "[[42 42 42]\n [42 42 42]]");
    }
    {
        NDArrayDynamicInternal<int> array_2d(np::Shape{4, 4}, 42);
        checkArrayRepr(array_2d, "[[42 42 42 42]\n [42 42 42 42]\n [42 42 42 42]\n [42 42 42 42]]");
    }
}

TEST_F(NDArrayDynamicInternalTest, fill3DCreationTest) {
    {
        NDArrayDynamicInternal<int> array_3d{np::Shape{2, 2, 3}, 42};
        checkArrayRepr(array_3d, "[[[42 42 42]\n [42 42 42]]\n [[42 42 42]\n [42 42 42]]]");
    }
    {
        NDArrayDynamicInternal<int> array_3d{np::Shape{2, 4, 2}, 42};
        checkArrayRepr(array_3d, "[[[42 42]\n [42 42]\n [42 42]\n [42 42]]\n [[42 42]\n [42 42]\n [42 42]\n [42 42]]]");
    }
}

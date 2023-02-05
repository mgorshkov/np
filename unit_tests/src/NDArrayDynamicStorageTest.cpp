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

#include <sstream>

using namespace np::ndarray::array_dynamic::internal;

class NDArrayDynamicStorageTest : public ArrayTest {
protected:
    template<typename DType>
    static void checkArrayRepr(const NDArrayDynamicStorage<DType> &array, const char *repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }
};

TEST_F(NDArrayDynamicStorageTest, defaultCreationTest) {
    /*
     >> print(np.array([]))
    []
     */
    checkArrayRepr(NDArrayDynamicStorage<int>{}, "[]");
}

TEST_F(NDArrayDynamicStorageTest, fromInitializerListCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    checkArrayRepr(NDArrayDynamicStorage<int>{1, 2, 3}, "[1 2 3]");
}

TEST_F(NDArrayDynamicStorageTest, assignmentOperatorTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    NDArrayDynamicStorage<int> array{1, 2, 3};
    NDArrayDynamicStorage<int> arrayCopy;
    arrayCopy = array;
    checkArrayRepr(arrayCopy, "[1 2 3]");
}

TEST_F(NDArrayDynamicStorageTest, fromCArray1DCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    int c_array_1d[3] = {1, 2, 3};
    NDArrayDynamicStorage<int> array_1d{c_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicStorageTest, fromCArray2DCreationTest) {
    /*
    print(np.array(([1,2,3],[4,5,6])))
    [[1 2 3]
     [4 5 6]]
    */
    int c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    NDArrayDynamicStorage<int> array_2d{c_array_2d};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayDynamicStorageTest, fromCArray3DCreationTest) {
    /*
     >>> print(np.array((([1,2,3],[4,5,6]),([7,8,9],[10,11,12]))))
     [[[ 1  2  3]
       [ 4  5  6]]

      [[ 7  8  9]
       [10 11 12]]]
     */
    int c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    NDArrayDynamicStorage<int> array_3d{c_array_3d};
    checkArrayRepr(array_3d, "[1 2 3 4 5 6 7 8 9 10 11 12]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdArray1DCreationTest) {
    std::array std_array_1d{1, 2, 3};
    NDArrayDynamicStorage<int> array_1d{std_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdArray2DCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d = {{{1, 2, 3}, {4, 5, 6}}};
    NDArrayDynamicStorage<int> array_2d{std_array_2d};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdArray3DCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::array<std::array<int, 3>, 2> std_array_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::array<std::array<std::array<int, 3>, 2>, 2> std_array_3d = {std_array_2d_1, std_array_2d_2};
    NDArrayDynamicStorage<int> array_3d{std_array_3d};
    checkArrayRepr(array_3d, "[1 2 3 4 5 6 7 8 9 10 11 12]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdVector1DCreationTest) {
    NDArrayDynamicStorage<int> array_1d{std::vector{1, 2, 3}};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdVector2DCreationTest) {
    NDArrayDynamicStorage<int> array_2d{std::vector{{std::vector{1, 2, 3}, std::vector{4, 5, 6}}}};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayDynamicStorageTest, fromStdVector3DCreationTest) {
    std::vector<std::vector<int>> std_vector_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::vector<std::vector<int>> std_vector_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::vector<std::vector<std::vector<int>>> std_vector_3d = {std_vector_2d_1, std_vector_2d_2};
    NDArrayDynamicStorage<int> array_3d{std_vector_3d};
    checkArrayRepr(array_3d, "[1 2 3 4 5 6 7 8 9 10 11 12]");
}

TEST_F(NDArrayDynamicStorageTest, fill1DCreationTest) {
    {
        NDArrayDynamicStorage<int> array_1d{42};
        checkArrayRepr(array_1d, "[42]");
    }
    {
        NDArrayDynamicStorage<int> array_1d(3, {42});
        checkArrayRepr(array_1d, "[42 42 42]");
    }
}

TEST_F(NDArrayDynamicStorageTest, fill2DCreationTest) {
    {
        NDArrayDynamicStorage<int> array_2d(2 * 3, 42);
        checkArrayRepr(array_2d, "[42 42 42 42 42 42]");
    }
    {
        NDArrayDynamicStorage<int> array_2d(4 * 4, 42);
        checkArrayRepr(array_2d, "[42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42]");
    }
}

TEST_F(NDArrayDynamicStorageTest, fill3DCreationTest) {
    {
        NDArrayDynamicStorage<int> array_3d(2 * 2 * 3, 42);
        checkArrayRepr(array_3d, "[42 42 42 42 42 42 42 42 42 42 42 42]");
    }
    {
        NDArrayDynamicStorage<int> array_3d(2 * 4 * 2, 42);
        checkArrayRepr(array_3d, "[42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42]");
    }
}

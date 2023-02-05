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

using namespace np::ndarray::array_static::internal;
using np::Size;

class NDArrayStaticStorageTest : public ArrayTest {
protected:
    template<typename DType, Size SizeT>
    static void checkArrayRepr(const NDArrayStaticStorage<DType, SizeT> &array, const char *repr) {
        std::ostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }

    template<typename DType, Size SizeT>
    static void checkArrayRepr(const NDArrayStaticStorage<DType, SizeT> &array, const wchar_t *repr) {
        std::wostringstream ss;
        ss << array;
        EXPECT_EQ(ss.str(), repr);
    }
};

TEST_F(NDArrayStaticStorageTest, fromInitializerListCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    checkArrayRepr(NDArrayStaticStorage<int, 3>{1, 2, 3}, "[1 2 3]");
}

TEST_F(NDArrayStaticStorageTest, assignmentOperatorTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    NDArrayStaticStorage<int, 3> array{1, 2, 3};
    NDArrayStaticStorage<int, 3> arrayCopy;
    arrayCopy = array;
    checkArrayRepr(arrayCopy, "[1 2 3]");
}

TEST_F(NDArrayStaticStorageTest, from1DIntCArrayCreationTest) {
    /*
    >>> print(np.array([1,2,3]))
    [1 2 3]
     */
    NDArrayStaticStorage<int, 3> array_1d{1, 2, 3};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayStaticStorageTest, from1DFloatCArrayCreationTest) {
    /*
    >>> print(np.array([1.1,2.2,3.3]))
    [1.1 2.2 3.3]
     */
    NDArrayStaticStorage<float, 3> array_1d{1.1f, 2.2f, 3.3f};
    checkArrayRepr(array_1d, "[1.1 2.2 3.3]");
}

TEST_F(NDArrayStaticStorageTest, from1DStringCArrayCreationTest) {
    /*
    >>> print(np.array(['1','2','3']))
    ['1' '2' '3']
     */
    NDArrayStaticStorage<std::string, 3> array_1d{"1", "2", "3"};
    checkArrayRepr(array_1d, R"(["1" "2" "3"])");
}

TEST_F(NDArrayStaticStorageTest, from2DIntCArrayCreationTest) {
    /*
    print(np.array(([1,2,3],[4,5,6])))
    [[1 2 3]
     [4 5 6]]
    */
    int c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    NDArrayStaticStorage<int, 2 * 3> array_2d{c_array_2d};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayStaticStorageTest, from2DFloatCArrayCreationTest) {
    /*
    print(np.array(([1.1,2.2,3.3],[4.4,5.5,6.6])))
    [[1.1 2.2 3.3]
     [4.4 5.5 6.6]]
    */
    float c_array_2d[2][3] = {{1.1f, 2.2f, 3.3f}, {4.4f, 5.5f, 6.6f}};
    NDArrayStaticStorage<float, 2 * 3> array_2d{c_array_2d};
    checkArrayRepr(array_2d, "[1.1 2.2 3.3 4.4000001 5.5 6.5999999]");
}

TEST_F(NDArrayStaticStorageTest, from2DStringCArrayCreationTest) {
    /*
    print(np.array((["str1", "str2", "str3"], ["str4", "str5", "str6"])))
    [["str1" "str2" "str3"]
     ["str4" "str5" "str6"]]
    */
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"},
                                    {"str4", "str5", "str6"}};
    NDArrayStaticStorage<std::string, 2 * 3> array_2d{c_array_2d};
    checkArrayRepr(array_2d, R"(["str1" "str2" "str3" "str4" "str5" "str6"])");
}

TEST_F(NDArrayStaticStorageTest, from2DUnicodeCArrayCreationTest) {
    /*
    print(np.array((["str1", "str2", "str3"], ["str4", "str5", "str6"])))
    [["str1" "str2" "str3"]
     ["str4" "str5" "str6"]]
    */
    std::wstring c_array_2d[2][3] = {{L"str1", L"str2", L"str3"},
                                     {L"str4", L"str5", L"str6"}};
    NDArrayStaticStorage<std::wstring, 2 * 3> array_2d{c_array_2d};
    checkArrayRepr(array_2d, R"(["str1" "str2" "str3" "str4" "str5" "str6"])");
}

TEST_F(NDArrayStaticStorageTest, from3DIntCArrayCreationTest) {
    /*
     >>> print(np.array((([1,2,3],[4,5,6]),([7,8,9],[10,11,12]))))
     [[[ 1  2  3]
       [ 4  5  6]]

      [[ 7  8  9]
       [10 11 12]]]
     */
    int c_array_3d[2][2][3] = {{{1, 2, 3},
                                {4, 5, 6}},
                               {{7, 8, 9},
                                {10, 11, 12}}};
    NDArrayStaticStorage<int, 2 * 2 * 3> array_3d{c_array_3d};
    checkArrayRepr(array_3d, R"([1 2 3 4 5 6 7 8 9 10 11 12])");
}

TEST_F(NDArrayStaticStorageTest, from3DFloatCArrayCreationTest) {
    /*
     >>> print(np.array((([1.1,2.2,3.3],[4.4,5.5,6.6]),([7.7,8.8,9.9],[10.1,11.11,12.12]))))
     [[[ 1.1  2.2  3.3]
       [ 4.4  5.5  6.6]]

      [[ 7.7  8.8  9.9]
       [10.1 11.11 12.12]]]
     */
    float c_array_3d[2][2][3] = {{{1.1f, 2.2f, 3.3f},
                                  {4.4f, 5.5f, 6.6f}},
                                 {{7.7f, 8.8f, 9.9f},
                                  {10.1f, 11.11f, 12.12f}}};
    NDArrayStaticStorage<float, 2 * 2 * 3> array_3d{c_array_3d};
    checkArrayRepr(array_3d, R"([1.1 2.2 3.3 4.4000001 5.5 6.5999999 7.6999998 8.8000002 9.8999996 10.1 11.11 12.12])");
}

TEST_F(NDArrayStaticStorageTest, from3DStringCArrayCreationTest) {
    std::string c_array_3d[2][4][3] = {{{"str1_1", "str1_2", "str1_3"},
                                        {"str2_1", "str2_2", "str2_3"},
                                        {"str3_1", "str3_2", "str3_3"},
                                        {"str4_1", "str4_2", "str4_3"}},
                                       {{"str5_1", "str5_2", "str5_3"},
                                        {"str6_1", "str6_2", "str6_3"},
                                        {"str7_1", "str7_2", "str7_3"},
                                        {"str8_1", "str8_2", "str8_3"}}};
    NDArrayStaticStorage<std::string, 2 * 4 * 3> array_3d{c_array_3d};
    checkArrayRepr(array_3d, R"(["str1_1" "str1_2" "str1_3" "str2_1" "str2_2" "str2_3" "str3_1" "str3_2" "str3_3" "str4_1" "str4_2" "str4_3" "str5_1" "str5_2" "str5_3" "str6_1" "str6_2" "str6_3" "str7_1" "str7_2" "str7_3" "str8_1" "str8_2" "str8_3"])");
}

TEST_F(NDArrayStaticStorageTest, from3DUnicodeCArrayCreationTest) {
    std::wstring c_array_3d[2][4][3] = {{{L"str1_1", L"str1_2", L"str1_3"},
                                         {L"str2_1", L"str2_2", L"str2_3"},
                                         {L"str3_1", L"str3_2", L"str3_3"},
                                         {L"str4_1", L"str4_2", L"str4_3"}},
                                        {{L"str5_1", L"str5_2", L"str5_3"},
                                         {L"str6_1", L"str6_2", L"str6_3"},
                                         {L"str7_1", L"str7_2", L"str7_3"},
                                         {L"str8_1", L"str8_2", L"str8_3"}}};
    NDArrayStaticStorage<std::wstring, 2 * 4 * 3> array_3d{c_array_3d};
    checkArrayRepr(array_3d, R"(["str1_1" "str1_2" "str1_3" "str2_1" "str2_2" "str2_3" "str3_1" "str3_2" "str3_3" "str4_1" "str4_2" "str4_3" "str5_1" "str5_2" "str5_3" "str6_1" "str6_2" "str6_3" "str7_1" "str7_2" "str7_3" "str8_1" "str8_2" "str8_3"])");
    // check wostream
    checkArrayRepr(array_3d, LR"(["str1_1" "str1_2" "str1_3" "str2_1" "str2_2" "str2_3" "str3_1" "str3_2" "str3_3" "str4_1" "str4_2" "str4_3" "str5_1" "str5_2" "str5_3" "str6_1" "str6_2" "str6_3" "str7_1" "str7_2" "str7_3" "str8_1" "str8_2" "str8_3"])");
}

TEST_F(NDArrayStaticStorageTest, from1DIntStdArrayCreationTest) {
    std::array std_array_1d{1, 2, 3};
    NDArrayStaticStorage<int, 3> array_1d{std_array_1d};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayStaticStorageTest, from2DIntStdArrayCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d = {{{1, 2, 3}, {4, 5, 6}}};
    NDArrayStaticStorage<int, 2 * 3> array_2d{std_array_2d};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayStaticStorageTest, from3DIntStdArrayCreationTest) {
    std::array<std::array<int, 3>, 2> std_array_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::array<std::array<int, 3>, 2> std_array_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::array<std::array<std::array<int, 3>, 2>, 2> std_array_3d = {std_array_2d_1, std_array_2d_2};
    NDArrayStaticStorage<int, 2 * 2 * 3> array_3d{std_array_3d};
    checkArrayRepr(array_3d, "[1 2 3 4 5 6 7 8 9 10 11 12]");
}

TEST_F(NDArrayStaticStorageTest, from1DIntStdVectorCreationTest) {
    NDArrayStaticStorage<int, 3> array_1d{std::vector{1, 2, 3}};
    checkArrayRepr(array_1d, "[1 2 3]");
}

TEST_F(NDArrayStaticStorageTest, from2DIntStdVectorCreationTest) {
    NDArrayStaticStorage<int, 2 * 3> array_2d{std::vector{{std::vector{1, 2, 3}, std::vector{4, 5, 6}}}};
    checkArrayRepr(array_2d, "[1 2 3 4 5 6]");
}

TEST_F(NDArrayStaticStorageTest, from3DIntStdVectorCreationTest) {
    std::vector<std::vector<int>> std_vector_2d_1 = {{{1, 2, 3}, {4, 5, 6}}};
    std::vector<std::vector<int>> std_vector_2d_2 = {{{7, 8, 9}, {10, 11, 12}}};
    std::vector<std::vector<std::vector<int>>> std_vector_3d = {std_vector_2d_1, std_vector_2d_2};
    NDArrayStaticStorage<int, 2 * 2 * 3> array_3d{std_vector_3d};
    checkArrayRepr(array_3d, "[1 2 3 4 5 6 7 8 9 10 11 12]");
}

TEST_F(NDArrayStaticStorageTest, fill1DIntCreationTest) {
    {
        NDArrayStaticStorage<int, 3> array_1d{42};
        checkArrayRepr(array_1d, "[42 42 42]");
    }
    {
        NDArrayStaticStorage<int, 3> array_1d({42});
        checkArrayRepr(array_1d, "[42 42 42]");
    }
}

TEST_F(NDArrayStaticStorageTest, fill2DIntCreationTest) {
    {
        NDArrayStaticStorage<int, 2 * 3> array_2d{42};
        checkArrayRepr(array_2d, "[42 42 42 42 42 42]");
    }
    {
        NDArrayStaticStorage<int, 2 * 3> array_2d({42});
        checkArrayRepr(array_2d, "[42 42 42 42 42 42]");
    }
}

TEST_F(NDArrayStaticStorageTest, fill3DIntCreationTest) {
    {
        NDArrayStaticStorage<int, 2 * 2 * 3> array_3d{42};
        checkArrayRepr(array_3d, "[42 42 42 42 42 42 42 42 42 42 42 42]");
    }
    {
        NDArrayStaticStorage<int, 2 * 2 * 3> array_3d({42});
        checkArrayRepr(array_3d, "[42 42 42 42 42 42 42 42 42 42 42 42]");
    }
}

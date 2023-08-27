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

#include <gtest/gtest.h>

#include <np/Array.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayManip2DTest : public ArrayTest {
protected:
};

TEST_F(ArrayManip2DTest, static2DIntArrayTransposeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayRavelTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayReshapeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<int_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayResizeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
    Array<int_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayAppendTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_2d_array_append[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_append{c_array_2d_array_append};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayInsertTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_2d_array_insert[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_insert{c_array_2d_array_insert};
    Array<int_> result_sample{1, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5, 6};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayDelTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    Array<int_> result_sample{1, 3, 4, 5, 6};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayConcatenateTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_2d_array_concatenate[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_concatenate{c_array_2d_array_concatenate};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayVStackTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_vstack_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_vstack{c_array_vstack_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayR_Test) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_r_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_r_{c_array_r_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayHStackTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ c_array_hstack[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array_hstack{c_array_hstack};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayColumnStackTest) {
    int_ c_array_2d1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array1{c_array_2d1};
    int_ c_array_2d2[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array2{c_array_2d2};
    int_ c_array_2d_result[2][6] = {{1, 2, 3, 7, 8, 9},
                                    {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = column_stack(array1, array2);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayC_Test) {
    // static
    int_ c_array_2d1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array1{c_array_2d1};
    int_ c_array_2d2[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2 * 3> array2{c_array_2d2};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_result};
    auto result = c_(array1, array2);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayHSplitTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto result = hsplit(array, 3);
    int_ c_array_2d_0[2][1] = {{1}, {4}};
    Array<int_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    int_ c_array_2d_1[2][1] = {{2}, {5}};
    Array<int_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
    int_ c_array_2d_2[2][1] = {{3}, {6}};
    Array<int_> result2_sample{c_array_2d_2};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayVSplitTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto result = vsplit(array, 2);
    int_ c_array_2d_0[1][3] = {{1, 2, 3}};
    Array<int_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    int_ c_array_2d_1[1][3] = {{4, 5, 6}};
    Array<int_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, static2DIntArrayExpandDimsTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto result = expand_dims(array, 0);
    int_ c_array_3d_result[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_> result_sample{c_array_3d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayTransposeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayRavelTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayReshapeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
    Array<float_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayResizeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
    Array<float_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayAppendTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_2d_array_append[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array_append{c_array_2d_array_append};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayInsertTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_2d_array_insert[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array_insert{c_array_2d_array_insert};
    Array<float_> result_sample{1.1, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayDelTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    Array<float_> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayConcatenateTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_2d_array_concatenate[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array_concatenate{c_array_2d_array_concatenate};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayVStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_vstack_2d[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array_vstack{c_array_vstack_2d};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayR_Test) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_r_2d[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2 * 3> array_r_{c_array_r_2d};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayHStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2 * 3> array_hstack{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayColumnStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_2d1[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2 * 3> array1{c_array_2d1};
    float_ c_array_2d_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayC_Test) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2 * 3> array_c_{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_result};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayHSplitTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto result = hsplit(array, 3);
    float_ c_array_2d_0[2][1] = {{1.1}, {4.4}};
    Array<float_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    float_ c_array_2d_1[2][1] = {{2.2}, {5.5}};
    Array<float_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
    float_ c_array_2d_2[2][1] = {{3.3}, {6.6}};
    Array<float_> result2_sample{c_array_2d_2};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayVSplitTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto result = vsplit(array, 2);
    float_ c_array_2d_0[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    float_ c_array_2d_1[1][3] = {{4.4, 5.5, 6.6}};
    Array<float_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, static2DFloatArrayExpandDimsTest) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto result = expand_dims(array, 0);
    float_ c_array_3d_result[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample{c_array_3d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayTransposeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_result_2d[3][2] = {{"str1", "str4"},
                                       {"str2", "str5"},
                                       {"str3", "str6"}};
    Array<string_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayRavelTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayReshapeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_result[3][2] = {
            {"str1", "str2"},
            {"str3", "str4"},
            {"str5", "str6"}};
    Array<string_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayResizeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_result[3][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"},
            {"str1", "str2", "str3"}};
    Array<string_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayAppendTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_append[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_append{c_array_2d_append};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9", "str10", "str11", "str12"};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayInsertTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_insert[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_insert{c_array_2d_insert};
    Array<string_> result_sample{"str1", "str7", "str8", "str9", "str10", "str11", "str12", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayDelTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayConcatenateTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_array_concatenate[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_concatenate{c_array_2d_array_concatenate};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayVStackTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_vstack_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_vstack{c_array_vstack_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayR_Test) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_r_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_r_{c_array_r_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayHStackTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_hstack[2][3] = {{"str7", "str8", "str9"},
                                    {"str10", "str11", "str12"}};
    ;
    Array<string_, 2 * 3> array_hstack{c_array_hstack};
    string_ c_array_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                    {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayColumnStackTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d1[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array1{c_array_2d1};
    string_ c_array_2d_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                       {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayC_Test) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_c_[2][3] = {{"str7", "str8", "str9"},
                                {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_c_{c_array_c_};
    string_ c_array_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                    {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_result};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayHSplitTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto result = hsplit(array, 3);
    string_ c_array_2d_0[2][1] = {{"str1"}, {"str4"}};
    Array<string_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    string_ c_array_2d_1[2][1] = {{"str2"}, {"str5"}};
    Array<string_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
    string_ c_array_2d_2[2][1] = {{"str3"}, {"str6"}};
    Array<string_> result2_sample{c_array_2d_2};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayVSplitTest) {
    // static
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto result = vsplit(array, 2);
    string_ c_array_2d_0[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    string_ c_array_2d_2[1][3] = {{"str4", "str5", "str6"}};
    Array<string_> result1_sample{c_array_2d_2};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, static2DStringArrayExpandDimsTest) {
    // static
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto result = expand_dims(array, 1);
    string_ c_array_2d_sample[2][1][3] = {
            {{"str1", "str2", "str3"}},
            {{"str4", "str5", "str6"}}};
    Array<string_> result_sample{c_array_2d_sample};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayTransposeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayRavelTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayReshapeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<int_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayResizeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
    Array<int_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayAppendTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    Array<int_> array_append{7, 8, 9};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayInsertTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    Array<int_> array_insert{7, 8, 9};
    Array<int_> result_sample{1, 7, 8, 9, 2, 3, 4, 5, 6};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayDelTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    Array<int_> result_sample{1, 3, 4, 5, 6};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayConcatenateTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_2d_array_concatenate[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_concatenate{c_array_2d_array_concatenate};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayVStackTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_vstack_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_vstack{c_array_vstack_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayR_Test) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_r_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_r_{c_array_r_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayHStackTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_hstack[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_hstack{c_array_hstack};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayColumnStackTest) {
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_2d1[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array1{c_array_2d1};
    int_ c_array_2d_result[2][6] = {{1, 2, 3, 7, 8, 9},
                                    {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayC_Test) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_hstack[2][2] = {{7, 8}, {9, 10}};
    Array<int_> array_c_{c_array_hstack};
    int_ c_array_result[2][5] = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};
    Array<int_> result_sample{c_array_result};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayHSplitTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto result = hsplit(array, 3);
    int_ c_array_2d_1[2][1] = {{1}, {4}};
    Array<int_> result0_sample{c_array_2d_1};
    compare(result[0], result0_sample);
    int_ c_array_2d_2[2][1] = {{2}, {5}};
    Array<int_> result1_sample{c_array_2d_2};
    compare(result[1], result1_sample);
    int_ c_array_2d_3[2][1] = {{3}, {6}};
    Array<int_> result2_sample{c_array_2d_3};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DIntArrayVSplitTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto result = vsplit(array, 2);
    int_ c_array_2d_0[1][3] = {{1, 2, 3}};
    Array<int_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    int_ c_array_2d_1[1][3] = {{4, 5, 6}};
    Array<int_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayTransposeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayRavelTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayReshapeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
    Array<float_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayResizeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
    Array<float_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayAppendTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    Array<float_> array_append{7.7, 8.8, 9.9};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayInsertTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    Array<float_> array_insert{7.7, 8.8, 9.9};
    Array<float_> result_sample{1.1, 7.7, 8.8, 9.9, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayDelTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    Array<float_> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayConcatenateTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_array_concatenate[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_concatenate{c_array_2d_array_concatenate};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayVStackTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_array_vstack[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_vstack{c_array_2d_array_vstack};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayR_Test) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d_array_r_[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_r_{c_array_2d_array_r_};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = r_<float_>(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayHStackTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_> array_hstack{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayColumnStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d1[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_> array1{c_array_2d1};
    float_ c_array_2d_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9},
                                      {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayC_Test) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_hstack[2][2] = {{7.7, 8.8}, {9.9, 10.1}};
    Array<float_> array_c_{c_array_hstack};
    float_ c_array_result[2][5] = {{1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9, 10.1}};
    Array<float_> result_sample{c_array_result};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayHSplitTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto result = hsplit(array, 3);
    float_ c_array_2d_1[2][1] = {{1.1}, {4.4}};
    Array<float_> result0_sample{c_array_2d_1};
    compare(result[0], result0_sample);
    float_ c_array_2d_2[2][1] = {{2.2}, {5.5}};
    Array<float_> result1_sample{c_array_2d_2};
    compare(result[1], result1_sample);
    float_ c_array_2d_3[2][1] = {{3.3}, {6.6}};
    Array<float_> result2_sample{c_array_2d_3};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayVSplitTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto result = vsplit(array, 2);
    float_ c_array_2d_0[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    float_ c_array_2d_1[1][3] = {{4.4, 5.5, 6.6}};
    Array<float_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DFloatArrayExpandDimsTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto result = expand_dims(array, 0);
    float_ c_array_3d_result[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample{c_array_3d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayTransposeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_result_2d[3][2] = {{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample{c_array_result_2d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayRavelTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayReshapeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_2d_result[3][2] = {{"str1", "str2"}, {"str3", "str4"}, {"str5", "str6"}};
    Array<string_> result_sample{c_array_2d_result};
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayResizeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_2d_result[3][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str1", "str2", "str3"}};
    Array<string_> result_sample{c_array_2d_result};
    Shape shape{3, 3};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayAppendTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    Array<string_> array_append{"str7", "str8", "str9"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9"};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayInsertTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    Array<string_> array_insert{"str7", "str8", "str9"};
    Array<string_> result_sample{"str1", "str7", "str8", "str9", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayDelTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayConcatenateTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    string_ c_array_2d_array_concatenate[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2 * 3> array_concatenate{c_array_2d_array_concatenate};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayVStackTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_vstack_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_> array_vstack{c_array_vstack_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayR_Test) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_r_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_> array_r_{c_array_r_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayHStackTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_hstack[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> array_hstack{c_array_hstack};
    string_ c_array_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                    {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_result};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayColumnStackTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_2d1[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> array1{c_array_2d1};
    string_ c_array_2d_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                       {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayC_Test) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_hstack[2][2] = {{"str7", "str8"}, {"str9", "str10"}};
    Array<string_> array_c_{c_array_hstack};
    string_ c_array_result[2][5] = {{"str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9", "str10"}};
    Array<string_> result_sample{c_array_result};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayHSplitTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto result = hsplit(array, 3);
    string_ c_array_2d_0[2][1] = {{"str1"}, {"str4"}};
    Array<string_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    string_ c_array_2d_1[2][1] = {{"str2"}, {"str5"}};
    Array<string_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
    string_ c_array_2d_2[2][1] = {{"str3"}, {"str6"}};
    Array<string_> result2_sample{c_array_2d_2};
    compare(result[2], result2_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayVSplitTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto result = vsplit(array, 2);
    string_ c_array_2d_0[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result0_sample{c_array_2d_0};
    compare(result[0], result0_sample);
    string_ c_array_2d_1[1][3] = {{"str4", "str5", "str6"}};
    Array<string_> result1_sample{c_array_2d_1};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManip2DTest, dynamic2DStringArrayExpandDimsTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto result = expand_dims(array, 0);
    string_ c_array_3d_result[1][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}}};
    Array<string_> result_sample{c_array_3d_result};
    compare(result, result_sample);
}

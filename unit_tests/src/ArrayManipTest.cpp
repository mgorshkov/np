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

class ArrayManipTest : public ArrayTest {
protected:
};

TEST_F(ArrayManipTest, dynamicEmptyIntArrayTransposeTest) {
    // dynamic
    Array<int_> array{};
    auto result = transpose(array);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayRavelTest) {
    // dynamic
    Array<int_> array{};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayReshapeTest) {
    // dynamic
    Array<int_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayResizeTest) {
    // dynamic
    Array<int_> array{};
    Shape shape;
    auto result = array.resize(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayAppendTest) {
    // dynamic
    Array<int_> array{};
    Array<int_> array1{};
    auto result = array.append(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayInsertTest) {
    // dynamic
    Array<int_> array{};
    Array<int_> array1{};
    EXPECT_THROW(array.insert(1, array1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayDelTest) {
    // dynamic
    Array<int_> array{};
    EXPECT_THROW(array.del(1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayConcatenateTest) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = array.concatenate(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayStackTest) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = stack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayVstackTest) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = vstack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayR_Test) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = r_(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayHStackTest) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = hstack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayColumsStackTest) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = column_stack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayC_Test) {
    Array<int_> array1{};
    Array<int_> array2{};
    auto result = c_(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayHSplitTest) {
    Array<int_> array1{};
    auto result = hsplit(array1, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array1);
    compare(result[1], array1);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayVSplitTest) {
    Array<int_> array1{};
    auto result = vsplit(array1, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array1);
    compare(result[1], array1);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayTransposeTest) {
    // dynamic
    Array<float_> array{};
    auto result = transpose(array);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayRavelTest) {
    // dynamic
    Array<float_> array{};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayReshapeTest) {
    // dynamic
    Array<float_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayResizeTest) {
    // dynamic
    Array<float_> array{};
    Shape shape;
    auto result = array.resize(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayAppendTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = array.append(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayInsertTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    EXPECT_THROW(array.insert(1, array1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayDelTest) {
    // dynamic
    Array<float_> array{};
    EXPECT_THROW(array.del(1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayConcatenateTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = array.concatenate(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayVstackTest) {
    // dynamic
    Array<float_> array1{};
    Array<float_> array2{};
    auto result = vstack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayStackTest) {
    // dynamic
    Array<float_> array1{};
    Array<float_> array2{};
    auto result = stack(array1, array2);
    compare(result, array1);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayR_Test) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = r_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayHStackTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = hstack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayColumsStackTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = column_stack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayC_Test) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = c_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayHSplitTest) {
    // dynamic
    Array<float_> array{};
    auto result = hsplit(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayVSplitTest) {
    // dynamic
    Array<float_> array{};
    auto result = vsplit<float_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayTransposeTest) {
    // dynamic
    Array<string_> array{};
    auto result = transpose<string_>(array);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayRavelTest) {
    // dynamic
    Array<string_> array{};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayReshapeTest) {
    // dynamic
    Array<string_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayResizeTest) {
    // dynamic
    Array<string_> array{};
    Shape shape;
    auto result = array.resize(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayAppendTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = array.append(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayInsertTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    EXPECT_THROW(array.insert(1, array1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayDelTest) {
    // dynamic
    Array<string_> array{};
    EXPECT_THROW(array.del(1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayConcatenateTest) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = array.concatenate(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayStackTest) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = stack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayVstackTest) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = vstack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayR_Test) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = r_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayHStackTest) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = hstack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayColumsStackTest) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = column_stack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayC_Test) {
    // dynamic
    auto array = createStringArray();
    auto array1 = createStringArray();
    auto result = c_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayHSplitTest) {
    // dynamic
    auto array = createStringArray();
    auto result = hsplit(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayVSplitTest) {
    // dynamic
    auto array = createStringArray();
    auto result = vsplit(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayTransposeTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto result = transpose(array);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayRavelTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayReshapeTest) {
    // dynamic
    auto array = createUnicodeArray();
    Shape shape;
    auto result = array.reshape(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayResizeTest) {
    // dynamic
    auto array = createUnicodeArray();
    Shape shape;
    auto result = array.resize(shape);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayAppendTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = array.append(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayInsertTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    EXPECT_THROW(array.insert(1, array1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayDelTest) {
    // dynamic
    auto array = createUnicodeArray();
    EXPECT_THROW(array.del(1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayConcatenateTest) {
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = array.concatenate(array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayStackTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = stack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayVstackTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = vstack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayR_Test) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = r_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayHStackTest) {
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = hstack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayColumsStackTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = column_stack(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayC_Test) {
    // dynamic
    auto array = createUnicodeArray();
    auto array1 = createUnicodeArray();
    auto result = c_(array, array1);
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayHSplitTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto result = hsplit(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayVSplitTest) {
    // dynamic
    auto array = createUnicodeArray();
    auto result = vsplit(array, 2);
    EXPECT_EQ(result.size(), 2U);
    compare(result[0], array);
    compare(result[1], array);
}

TEST_F(ArrayManipTest, static1DIntArrayTransposeTest) {
    // static
    std::initializer_list<int_> ar = {1, 2, 3};
    auto array = createIntArray<3>(ar);
    std::initializer_list<int_> res = {1, 2, 3};
    auto result_sample = createIntArray(res);
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayRavelTest) {
    // static
    std::initializer_list<int_> ar = {1, 2, 3};
    auto array = createIntArray(ar);
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, static1DIntArrayReshapeTest) {
    // static
    std::initializer_list<int_> ar = {1, 2, 3};
    auto array = createIntArray<3>(ar);
    int_ c_array_2d[3][1] = {{1}, {2}, {3}};
    auto result_sample = createIntArray(c_array_2d);
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayResizeTest) {
    // static
    std::initializer_list<int_> ar = {1, 2, 3};
    auto array = createIntArray<3>(ar);
    std::initializer_list<int_> res = {1, 2, 3, 1};
    auto result_sample = createIntArray(res);
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayAppendTest) {
    // static
    std::initializer_list<int_> ar = {1, 2, 3};
    auto array = createIntArray<3>(ar);
    std::initializer_list<int_> ar_append = {4, 5, 6};
    auto array_append = createIntArray<3>(ar_append);
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayInsertTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_insert{4, 5, 6};
    Array<int_> result_sample{1, 4, 5, 6, 2, 3};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayDelTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_> result_sample{1, 3};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayConcatenateTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_concatenate{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_stack{4, 5, 6};
    int_ result_sample_c[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayVStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_vstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayR_Test) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_r_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayHStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_hstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayColumnStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_column_stack{4, 5, 6};
    int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayC_Test) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_c_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayHSplitTest) {
    // static
    Array<int_, 4> array{1, 2, 3, 4};
    auto result = hsplit(array, 2);
    Array<int_> result0_sample{1, 2};
    compare(result[0], result0_sample);
    Array<int_> result1_sample{3, 4};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayExpandDimsTest) {
    // static
    Array<int_, 4> array{1, 2, 3, 4};
    auto result = expand_dims(array, 0);
    int_ c_array_2d[1][4] = {{1, 2, 3, 4}};
    Array<int_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayExpandDims1Test) {
    // static
    Array<int_, 4> array{1, 2, 3, 4};
    auto result = expand_dims(array, 1);
    int_ c_array_2d[4][1] = {{1}, {2}, {3}, {4}};
    Array<int_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DIntArrayWhereTest) {
    // static
    Array<int_, 10> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = where<int_>(
            x, [](const auto &element) { return element < 5; }, [](const auto &) { return 1; }, [](const auto &element) { return element * 2; });
    Array<int_> result_sample{1, 1, 1, 1, 1, 10, 12, 14, 16, 18};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayTransposeTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayRavelTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3};
    auto result = array.ravel();
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayReshapeTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
    Array<float_> result_sample{c_array_2d};
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayResizeTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayAppendTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_append{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayInsertTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_insert{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayDelTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 3.3};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayConcatenateTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_concatenate{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayStackTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_stack{4.4, 5.5, 6.6};
    float_ result_sample_c[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayVStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_vstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayR_Test) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_r_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayHStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_hstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayColumnStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_column_stack{4.4, 5.5, 6.6};
    float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayC_Test) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_c_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayHSplitTest) {
    Array<float_, 4> array{1.1, 2.2, 3.3, 4.4};
    auto result = hsplit(array, 2);
    Array<float_> result0_sample{1.1, 2.2};
    compare(result[0], result0_sample);
    Array<float_> result1_sample{3.3, 4.4};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayExpandDimsTest) {
    // static
    Array<float_, 4> array{1.1, 2.2, 3.3, 4.4};
    auto result = expand_dims(array, 0);
    float_ c_array_2d[1][4] = {{1.1, 2.2, 3.3, 4.4}};
    Array<float_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DFloatArrayExpandDims1Test) {
    // static
    Array<float_, 4> array{1.1, 2.2, 3.3, 4.4};
    auto result = expand_dims(array, 1);
    float_ c_array_2d[4][1] = {{1.1}, {2.2}, {3.3}, {4.4}};
    Array<float_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayTransposeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3"};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayRavelTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3"};
    auto result = array.ravel();
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayReshapeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
    Array<string_> result_sample{c_array_2d};
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayResizeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3", "str1"};
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayAppendTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_append{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayInsertTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_insert{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayDelTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str3"};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayConcatenateTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_concatenate{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayStackTest) {
    // static
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_stack{"str4", "str5", "str6"};
    string_ result_sample_c[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayVStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_vstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayR_Test) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_r_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayHStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_hstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayColumnStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_column_stack{"str4", "str5", "str6"};
    string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayC_Test) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_c_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayHSplitTest) {
    Array<string_, 4> array{"str1", "str2", "str3", "str4"};
    auto result = hsplit(array, 2);
    Array<string_> result0_sample{"str1", "str2"};
    compare(result[0], result0_sample);
    Array<string_> result1_sample{"str3", "str4"};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayExpandDimsTest) {
    // static
    Array<string_, 4> array{"str1", "str2", "str3", "str4"};
    auto result = expand_dims(array, 0);
    string_ c_array_2d[1][4] = {{"str1", "str2", "str3", "str4"}};
    Array<string_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, static1DStringArrayExpandDims1Test) {
    // static
    Array<string_, 4> array{"str1", "str2", "str3", "str4"};
    auto result = expand_dims(array, 1);
    string_ c_array_2d[4][1] = {{"str1"}, {"str2"}, {"str3"}, {"str4"}};
    Array<string_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayTransposeTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 2, 3};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayRavelTest) {
    Array<int_> array{1, 2, 3};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayReshapeTest) {
    Array<int_> array{1, 2, 3};
    int_ c_array_2d[3][1] = {{1}, {2}, {3}};
    Array<int_> result_sample{c_array_2d};
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayResizeTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 2, 3, 1};
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayAppendTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_append{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayInsertTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_insert{4, 5, 6};
    Array<int_> result_sample{1, 4, 5, 6, 2, 3};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayDelTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 3};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayConcatenateTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_concatenate{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_stack{4, 5, 6};
    int_ result_sample_c[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayVStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_vstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayR_Test) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_r_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayHStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_hstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayColumnStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_column_stack{4, 5, 6};
    int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayC_Test) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_c_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayHSplitTest) {
    Array<int_> array{1, 2, 3, 4};
    auto result = hsplit(array, 2);
    Array<int_> result0_sample{1, 2};
    compare(result[0], result0_sample);
    Array<int_> result1_sample{3, 4};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayExpandDimsTest) {
    Array<int_> array{1, 2, 3, 4};
    auto result = expand_dims(array, 0);
    int_ c_array_2d[1][4] = {{1, 2, 3, 4}};
    Array<int_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayExpandDims1Test) {
    Array<int_> array{1, 2, 3, 4};
    auto result = expand_dims(array, 1);
    int_ c_array_2d[4][1] = {{1}, {2}, {3}, {4}};
    Array<int_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayWhereTest) {
    Array<int_> x{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = where<int_>(
            x, [](const auto &element) { return element < 5; }, [](const auto &) { return 1; }, [](const auto &element) { return element * 2; });
    Array<int_> result_sample{1, 1, 1, 1, 1, 10, 12, 14, 16, 18};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayTransposeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayRavelTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayReshapeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
    Array<float_> result_sample{c_array_2d};
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayResizeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayAppendTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_append{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayInsertTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_insert{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayDelTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 3.3};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayConcatenateTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_concatenate{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_stack{4.4, 5.5, 6.6};
    float_ result_sample_c[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayVStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_vstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayR_Test) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_r_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayHStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_hstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayColumnStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_column_stack{4.4, 5.5, 6.6};
    float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayC_Test) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_c_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayHSplitTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    auto result = hsplit<float_>(array, 2);
    Array<float_> result0_sample{1.1, 2.2};
    compare(result[0], result0_sample);
    Array<float_> result1_sample{3.3, 4.4};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayExpandDimsTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    auto result = expand_dims(array, 0);
    float_ c_array_2d[1][4] = {{1.1, 2.2, 3.3, 4.4}};
    Array<float_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayExpandDims1Test) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    auto result = expand_dims(array, 1);
    float_ c_array_2d[4][1] = {{1.1}, {2.2}, {3.3}, {4.4}};
    Array<float_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayTransposeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3"};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayRavelTest) {
    Array<string_> array{"str1", "str2", "str3"};
    auto result = array.ravel();
    compare(result, array);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayReshapeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
    Array<string_> result_sample{c_array_2d};
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayResizeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3", "str1"};
    Shape shape{4};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayAppendTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_append{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.append(array_append);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayInsertTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_insert{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
    auto result = array.insert(1, array_insert);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayDelTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str3"};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayConcatenateTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_concatenate{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.concatenate(array_concatenate);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_stack{"str4", "str5", "str6"};
    string_ result_sample_c[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> result_sample{result_sample_c};
    auto result = stack(array, array_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayVStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_vstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = vstack(array, array_vstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayR_Test) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_r_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = r_(array, array_r_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayHStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_hstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = hstack(array, array_hstack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayColumnStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_column_stack{"str4", "str5", "str6"};
    string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample{array_2D};
    auto result = column_stack(array, array_column_stack);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayC_Test) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_c_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = c_(array, array_c_);
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayHSplitTest) {
    Array<string_> array{"str1", "str2", "str3", "str4"};
    auto result = hsplit<string_>(array, 2);
    Array<string_> result0_sample{"str1", "str2"};
    compare(result[0], result0_sample);
    Array<string_> result1_sample{"str3", "str4"};
    compare(result[1], result1_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayExpandDimsTest) {
    Array<string_> array{"str1", "str2", "str3", "str4"};
    auto result = expand_dims(array, 0);
    string_ c_array_2d[1][4] = {{"str1", "str2", "str3", "str4"}};
    Array<string_> result_sample{c_array_2d};
    compare(result, result_sample);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayExpandDims1Test) {
    Array<string_> array{"str1", "str2", "str3", "str4"};
    auto result = expand_dims(array, 1);
    string_ c_array_2d[4][1] = {{"str1"}, {"str2"}, {"str3"}, {"str4"}};
    Array<string_> result_sample{c_array_2d};
    compare(result, result_sample);
}

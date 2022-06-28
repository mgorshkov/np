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

#include <gtest/gtest.h>
#include <iostream>

#include <np/Array.hpp>
#include <np/Comp.hpp>
#include <np/Manip.hpp>

using namespace np;

class ArrayManipTest : public ::testing::Test {
protected:
};

TEST_F(ArrayManipTest, dynamicEmptyIntArrayTransposeTest) {
    // dynamic
    Array<int_> array{};
    auto result = transpose<int_>(array);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayRavelTest) {
    // dynamic
    Array<int_> array{};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayReshapeTest) {
    // dynamic
    Array<int_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayResizeTest) {
    // dynamic
    Array<int_> array{};
    Shape shape;
    auto result = array.resize(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayAppendTest) {
    // dynamic
    Array<int_> array{};
    Array<int_> array1{};
    auto result = array.append(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
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
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayVstackTest) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = vstack<int_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayR_Test) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = r_<int_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayHStackTest) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = hstack<int_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayColumsStackTest) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = column_stack<int_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayC_Test) {
    Array<int_> array{};
    Array<int_> array1{};
    auto result = c_<int_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayHSplitTest) {
    Array<int_> array{};
    auto result = hsplit<int_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyIntArrayVSplitTest) {
    Array<int_> array{};
    auto result = vsplit<int_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayTransposeTest) {
    // dynamic
    Array<float_> array{};
    auto result = transpose<float_>(array);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayRavelTest) {
    // dynamic
    Array<float_> array{};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayReshapeTest) {
    // dynamic
    Array<float_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayResizeTest) {
    // dynamic
    Array<float_> array{};
    Shape shape;
    auto result = array.resize(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayAppendTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = array.append(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
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
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayVstackTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = vstack<float_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayR_Test) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = r_<float_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayHStackTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = hstack<float_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayColumsStackTest) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = column_stack<float_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayC_Test) {
    // dynamic
    Array<float_> array{};
    Array<float_> array1{};
    auto result = c_<float_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayHSplitTest) {
    // dynamic
    Array<float_> array{};
    auto result = hsplit<float_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayVSplitTest) {
    // dynamic
    Array<float_> array{};
    auto result = vsplit<float_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayTransposeTest) {
    // dynamic
    Array<string_> array{};
    auto result = transpose<string_>(array);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayRavelTest) {
    // dynamic
    Array<string_> array{};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayReshapeTest) {
    // dynamic
    Array<string_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayResizeTest) {
    // dynamic
    Array<string_> array{};
    Shape shape;
    auto result = array.resize(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayAppendTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = array.append(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
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
    Array<string_> array{};
    Array<string_> array1{};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayVstackTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = vstack<string_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayR_Test) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = r_<string_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayHStackTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = hstack<string_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayColumsStackTest) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = column_stack<string_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayC_Test) {
    // dynamic
    Array<string_> array{};
    Array<string_> array1{};
    auto result = c_<string_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayHSplitTest) {
    // dynamic
    Array<string_> array{};
    auto result = hsplit<string_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayVSplitTest) {
    // dynamic
    Array<string_> array{};
    auto result = vsplit<string_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayTransposeTest) {
    // dynamic
    Array<unicode_> array{};
    auto result = transpose<unicode_>(array);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayRavelTest) {
    // dynamic
    Array<unicode_> array{};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayReshapeTest) {
    // dynamic
    Array<unicode_> array{};
    Shape shape;
    auto result = array.reshape(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayResizeTest) {
    // dynamic
    Array<unicode_> array{};
    Shape shape;
    auto result = array.resize(shape);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayAppendTest) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = array.append(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayInsertTest) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    EXPECT_THROW(array.insert(1, array1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayDelTest) {
    // dynamic
    Array<unicode_> array{};
    EXPECT_THROW(array.del(1), std::runtime_error);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayConcatenateTest) {
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayVstackTest) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = vstack<unicode_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayR_Test) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = r_<unicode_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayHStackTest) {
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = hstack<unicode_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayColumsStackTest) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = column_stack<unicode_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayC_Test) {
    // dynamic
    Array<unicode_> array{};
    Array<unicode_> array1{};
    auto result = c_<unicode_>(array, array1);
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayHSplitTest) {
    // dynamic
    Array<unicode_> array{};
    auto result = hsplit<unicode_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayVSplitTest) {
    // dynamic
    Array<unicode_> array{};
    auto result = vsplit<unicode_>(array, 2);
    EXPECT_EQ(result.size(), 2U);
    bool equals = array_equal(result[0], array);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayTransposeTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> result_sample{1, 2, 3};
    auto result = transpose<int_, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayRavelTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayReshapeTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    int_ c_array_2d[3][1] = {{1}, {2}, {3}};
    Array<int_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayResizeTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_> result_sample{1, 2, 3, 1};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayAppendTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_append{4, 5, 6};
    Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayInsertTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_insert{4, 5, 6};
    Array<int_, 6> result_sample{1, 4, 5, 6, 2, 3};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayDelTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 2> result_sample{1, 3};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayConcatenateTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_concatenate{4, 5, 6};
    Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayVStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_vstack{4, 5, 6};
    Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
    auto result = vstack<int_, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayR_Test) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_r_{4, 5, 6};
    Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
    auto result = r_<int_, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayHStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_hstack{4, 5, 6};
    Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
    auto result = hstack<int_, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayColumnStackTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_column_stack{4, 5, 6};
    int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample = array_2D;
    auto result = column_stack<int_, 3>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayC_Test) {
    // static
    Array<int_, 3> array{1, 2, 3};
    Array<int_, 3> array_c_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = c_<int_, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DIntArrayHSplitTest) {
    // static
    Array<int_, 4> array{1, 2, 3, 4};
    auto result = hsplit<int_, 4>(array, 2);
    Array<int_> result0_sample{1, 2};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<int_> result1_sample{3, 4};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayTransposeTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> result_sample{1.1, 2.2, 3.3};
    auto result = transpose<float_, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayRavelTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayReshapeTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
    Array<float_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayResizeTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayAppendTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_append{4.4, 5.5, 6.6};
    Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayInsertTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_insert{4.4, 5.5, 6.6};
    Array<float_, 6> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayDelTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 2> result_sample{1.1, 3.3};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayConcatenateTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_concatenate{4.4, 5.5, 6.6};
    Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayVStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_vstack{4.4, 5.5, 6.6};
    Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = vstack<float_, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayR_Test) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_r_{4.4, 5.5, 6.6};
    Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = r_<float_, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayHStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_hstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = hstack<float_, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayCoumnStackTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_column_stack{4.4, 5.5, 6.6};
    float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample = array_2D;
    auto result = column_stack<float_, 3>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayC_Test) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    Array<float_, 3> array_c_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = c_<float_, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DFloatArrayHSplitTest) {
    Array<float_, 4> array{1.1, 2.2, 3.3, 4.4};
    auto result = hsplit<float_, 4>(array, 2);
    Array<float_> result0_sample{1.1, 2.2};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<float_> result1_sample{3.3, 4.4};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayTransposeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> result_sample{"str1", "str2", "str3"};
    auto result = transpose<string_, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayRavelTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayReshapeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
    Array<string_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayResizeTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3", "str1"};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayAppendTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_append{"str4", "str5", "str6"};
    Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayInsertTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_insert{"str4", "str5", "str6"};
    Array<string_, 6> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayDelTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 2> result_sample{"str1", "str3"};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayConcatenateTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_concatenate{"str4", "str5", "str6"};
    Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayVStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_vstack{"str4", "str5", "str6"};
    Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = vstack<string_, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayR_Test) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_r_{"str4", "str5", "str6"};
    Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = r_<string_, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayHStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_hstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = hstack<string_, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayColumnStackTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_column_stack{"str4", "str5", "str6"};
    string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample = array_2D;
    auto result = column_stack<string_, 3>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayC_Test) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    Array<string_, 3> array_c_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = c_<string_, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static1DStringArrayHSplitTest) {
    Array<string_, 4> array{"str1", "str2", "str3", "str4"};
    auto result = hsplit<string_, 4>(array, 2);
    Array<string_> result0_sample{"str1", "str2"};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<string_> result1_sample{"str3", "str4"};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayTransposeTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 2, 3};
    auto result = transpose<int_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayRavelTest) {
    Array<int_> array{1, 2, 3};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayReshapeTest) {
    Array<int_> array{1, 2, 3};
    int_ c_array_2d[3][1] = {{1}, {2}, {3}};
    Array<int_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayResizeTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 2, 3, 1};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayAppendTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_append{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayInsertTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_insert{4, 5, 6};
    Array<int_> result_sample{1, 4, 5, 6, 2, 3};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayDelTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> result_sample{1, 3};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayConcatenateTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_concatenate{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayVStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_vstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = vstack<int_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayR_Test) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_r_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = r_<int_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayHStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_hstack{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = hstack<int_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayColumnStackTest) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_column_stack{4, 5, 6};
    int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample = array_2D;
    auto result = column_stack<int_>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayC_Test) {
    Array<int_> array{1, 2, 3};
    Array<int_> array_c_{4, 5, 6};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    auto result = c_<int_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DIntArrayHSplitTest) {
    Array<int_> array{1, 2, 3, 4};
    auto result = hsplit<int_>(array, 2);
    Array<int_> result0_sample{1, 2};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<int_> result1_sample{3, 4};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayTransposeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3};
    auto result = transpose<float_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayRavelTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayReshapeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
    Array<float_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayResizeTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayAppendTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_append{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayInsertTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_insert{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayDelTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> result_sample{1.1, 3.3};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayConcatenateTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_concatenate{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayVStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_vstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = vstack<float_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayR_Test) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_r_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = r_<float_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayHStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_hstack{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = hstack<float_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayColumnStackTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_column_stack{4.4, 5.5, 6.6};
    float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample = array_2D;
    auto result = column_stack<float_>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayC_Test) {
    Array<float_> array{1.1, 2.2, 3.3};
    Array<float_> array_c_{4.4, 5.5, 6.6};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = c_<float_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayHSplitTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    auto result = hsplit<float_>(array, 2);
    Array<float_> result0_sample{1.1, 2.2};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<float_> result1_sample{3.3, 4.4};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayTransposeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3"};
    auto result = transpose<string_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayRavelTest) {
    Array<string_> array{"str1", "str2", "str3"};
    auto result = array.ravel();
    bool equals = array_equal(result, array);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayReshapeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
    Array<string_> result_sample = c_array_2d;
    Shape shape{3, 1};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayResizeTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str2", "str3", "str1"};
    Shape shape{4};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayAppendTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_append{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayInsertTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_insert{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayDelTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> result_sample{"str1", "str3"};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayConcatenateTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_concatenate{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayVStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_vstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = vstack<string_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayR_Test) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_r_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = r_<string_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayHStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_hstack{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = hstack<string_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayColumnStackTest) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_column_stack{"str4", "str5", "str6"};
    string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample = array_2D;
    auto result = column_stack<string_>(array, array_column_stack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayC_Test) {
    Array<string_> array{"str1", "str2", "str3"};
    Array<string_> array_c_{"str4", "str5", "str6"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    auto result = c_<string_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic1DStringArrayHSplitTest) {
    Array<string_> array{"str1", "str2", "str3", "str4"};
    auto result = hsplit<string_>(array, 2);
    Array<string_> result0_sample{"str1", "str2"};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    Array<string_> result1_sample{"str3", "str4"};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayTransposeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample = c_array_result_2d;
    auto result = transpose<int_, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayRavelTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayReshapeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<int_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayResizeTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
    Array<int_> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayAppendTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_2d_array_append[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_append = c_array_2d_array_append;
    Array<int_, 12> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayInsertTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_2d_array_insert[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_insert = c_array_2d_array_insert;
    Array<int_, 12> result_sample{1, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5, 6};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayDelTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    Array<int_, 5> result_sample{1, 3, 4, 5, 6};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayConcatenateTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_2d_array_concatenate[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_concatenate = c_array_2d_array_concatenate;
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayVStackTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_vstack_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_vstack{c_array_vstack_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = vstack<int_, 2, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayR_Test) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_r_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_r_{c_array_r_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = r_<int_, 2, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayHStackTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_hstack[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_hstack{c_array_hstack};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample = c_array_result;
    auto result = hstack<int_, 2, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayColumnStackTest) {
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array{c_array_2d};
    int_ c_array_2d1[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array1{c_array_2d1};
    int_ c_array_2d_result[2][6] = {{1, 2, 3, 7, 8, 9},
                                    {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = column_stack<int_, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayC_Test) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    int_ c_array_c_[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_, 2, 3> array_c_{c_array_c_};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample = c_array_result;
    auto result = c_<int_, 2, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayHSplitTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    auto result = hsplit<int_, 2, 3>(array, 3);
    int_ c_array_2d_0[2][1] = {{1}, {4}};
    Array<int_> result0_sample = c_array_2d_0;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_1[2][1] = {{2}, {5}};
    Array<int_> result1_sample = c_array_2d_1;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_2[2][1] = {{3}, {6}};
    Array<int_> result2_sample = c_array_2d_2;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DIntArrayVSplitTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array = c_array_2d;
    auto result = vsplit<int_, 2, 3>(array, 2);
    int_ c_array_2d_0[1][3] = {{1, 2, 3}};
    Array<int_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_1[1][3] = {{4, 5, 6}};
    Array<int_> result1_sample{c_array_2d_1};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayTransposeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_, 3, 2> result_sample = c_array_result_2d;
    auto result = transpose<float_, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayRavelTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayReshapeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
    Array<float_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayResizeTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
    Array<float_, 3, 3> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayAppendTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_2d_array_append[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2, 3> array_append = c_array_2d_array_append;
    Array<float_, 12> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayInsertTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_2d_array_insert[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2, 3> array_insert = c_array_2d_array_insert;
    Array<float_, 12> result_sample{1.1, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayDelTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    Array<float_, 5> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayConcatenateTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_2d_array_concatenate[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2, 3> array_concatenate = c_array_2d_array_concatenate;
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayVStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_vstack_2d[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2, 3> array_vstack{c_array_vstack_2d};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = vstack<float_, 2, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayR_Test) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_r_2d[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_, 2, 3> array_r_{c_array_r_2d};
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = r_<float_, 2, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayHStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2, 3> array_hstack{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample = c_array_result;
    auto result = hstack<float_, 2, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayColumnStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array{c_array_2d};
    float_ c_array_2d1[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2, 3> array1{c_array_2d1};
    float_ c_array_2d_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = column_stack<float_, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayC_Test) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_, 2, 3> array_c_{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample = c_array_result;
    auto result = c_<float_, 2, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayHSplitTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    auto result = hsplit<float_, 2, 3>(array, 3);
    float_ c_array_2d_0[2][1] = {{1.1}, {4.4}};
    Array<float_> result0_sample = c_array_2d_0;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_1[2][1] = {{2.2}, {5.5}};
    Array<float_> result1_sample = c_array_2d_1;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_2[2][1] = {{3.3}, {6.6}};
    Array<float_> result2_sample = c_array_2d_2;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DFloatArrayVSplitTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array = c_array_2d;
    auto result = vsplit<float_, 2, 3>(array, 2);
    float_ c_array_2d_0[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_1[1][3] = {{4.4, 5.5, 6.6}};
    Array<float_> result1_sample{c_array_2d_1};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayTransposeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_result_2d[3][2] = {{"str1", "str4"},
                                       {"str2", "str5"},
                                       {"str3", "str6"}};
    Array<string_> result_sample = c_array_result_2d;
    auto result = transpose<string_, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayRavelTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayReshapeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_result[3][2] = {
            {"str1", "str2"},
            {"str3", "str4"},
            {"str5", "str6"}};
    Array<string_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayResizeTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_result[3][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"},
            {"str1", "str2", "str3"}};
    Array<string_> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayAppendTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_append[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_append = c_array_2d_append;
    Array<string_, 12> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9", "str10", "str11", "str12"};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayInsertTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_insert[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_insert = c_array_2d_insert;
    Array<string_, 12> result_sample{"str1", "str7", "str8", "str9", "str10", "str11", "str12", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayDelTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayConcatenateTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_array_concatenate[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_concatenate = c_array_2d_array_concatenate;
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayVStackTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_vstack_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_vstack{c_array_vstack_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = vstack<string_, 2, 3>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayR_Test) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array{c_array_2d};
    string_ c_array_r_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_r_{c_array_r_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = r_<string_, 2, 3>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayHStackTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_hstack[2][3] = {{"str7", "str8", "str9"},
                                    {"str10", "str11", "str12"}};
    ;
    Array<string_, 2, 3> array_hstack{c_array_hstack};
    string_ c_array_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                    {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample = c_array_result;
    auto result = hstack<string_, 2, 3>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayColumnStackTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array{c_array_2d};
    string_ c_array_2d1[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array1{c_array_2d1};
    string_ c_array_2d_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                       {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = column_stack<string_, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayC_Test) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_c_[2][3] = {{"str7", "str8", "str9"},
                                {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_c_{c_array_c_};
    string_ c_array_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                    {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample = c_array_result;
    auto result = c_<string_, 2, 3>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayHSplitTest) {
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    auto result = hsplit<string_, 2, 3>(array, 3);
    string_ c_array_2d_0[2][1] = {{"str1"}, {"str4"}};
    Array<string_> result0_sample = c_array_2d_0;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_1[2][1] = {{"str2"}, {"str5"}};
    Array<string_> result1_sample = c_array_2d_1;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_2[2][1] = {{"str3"}, {"str6"}};
    Array<string_> result2_sample = c_array_2d_2;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static2DStringArrayVSplitTest) {
    // static
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    auto result = vsplit<string_, 2, 3>(array, 2);
    string_ c_array_2d_0[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_2[1][3] = {{"str4", "str5", "str6"}};
    Array<string_> result1_sample{c_array_2d_2};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayTransposeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
    Array<int_> result_sample = c_array_result_2d;
    auto result = transpose<int_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayRavelTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayReshapeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<int_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayResizeTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
    Array<int_> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayAppendTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    Array<int_> array_append{7, 8, 9};
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayInsertTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    Array<int_> array_insert{7, 8, 9};
    Array<int_> result_sample{1, 7, 8, 9, 2, 3, 4, 5, 6};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayDelTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    Array<int_> result_sample{1, 3, 4, 5, 6};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayConcatenateTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_2d_array_concatenate[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_concatenate = c_array_2d_array_concatenate;
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayVStackTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_vstack_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_vstack{c_array_vstack_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = vstack<int_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayR_Test) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_r_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_r_{c_array_r_2d};
    int_ c_array_2d_result[4][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = r_<int_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayHStackTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_hstack[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array_hstack{c_array_hstack};
    int_ c_array_result[2][6] = {{1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample = c_array_result;
    auto result = hstack<int_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayColumnStackTest) {
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ c_array_2d1[2][3] = {{7, 8, 9}, {10, 11, 12}};
    Array<int_> array1{c_array_2d1};
    int_ c_array_2d_result[2][6] = {{1, 2, 3, 7, 8, 9},
                                    {4, 5, 6, 10, 11, 12}};
    Array<int_> result_sample{c_array_2d_result};
    auto result = column_stack<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayC_Test) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    int_ c_array_hstack[2][2] = {{7, 8}, {9, 10}};
    Array<int_> array_c_{c_array_hstack};
    int_ c_array_result[2][5] = {{1, 2, 3, 7, 8}, {4, 5, 6, 9, 10}};
    Array<int_> result_sample = c_array_result;
    auto result = c_<int_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayHSplitTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    auto result = hsplit<int_>(array, 3);
    int_ c_array_2d_1[2][1] = {{1}, {4}};
    Array<int_> result0_sample = c_array_2d_1;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_2[2][1] = {{2}, {5}};
    Array<int_> result1_sample = c_array_2d_2;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_3[2][1] = {{3}, {6}};
    Array<int_> result2_sample = c_array_2d_3;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DIntArrayVSplitTest) {
    /*
    a=np.array([[1, 2, 3], [4, 5, 6]])
     */
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array = c_array_2d;
    auto result = vsplit<int_>(array, 2);
    int_ c_array_2d_0[1][3] = {{1, 2, 3}};
    Array<int_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    int_ c_array_2d_1[1][3] = {{4, 5, 6}};
    Array<int_> result1_sample{c_array_2d_1};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayTransposeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
    Array<float_> result_sample = c_array_result_2d;
    auto result = transpose<float_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayRavelTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayReshapeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
    Array<float_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayResizeTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
    Array<float_> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayAppendTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    Array<float_> array_append{7.7, 8.8, 9.9};
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayInsertTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    Array<float_> array_insert{7.7, 8.8, 9.9};
    Array<float_> result_sample{1.1, 7.7, 8.8, 9.9, 2.2, 3.3, 4.4, 5.5, 6.6};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayDelTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    Array<float_> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayConcatenateTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_array_concatenate[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_concatenate = c_array_2d_array_concatenate;
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayVStackTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_array_vstack[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_vstack = c_array_2d_array_vstack;
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = vstack<float_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayR_Test) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_2d_array_r_[2][3] = {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> array_r_ = c_array_2d_array_r_;
    float_ c_array_2d_result[4][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = r_<float_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayHStackTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_hstack[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_> array_hstack{c_array_hstack};
    float_ c_array_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9}, {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample = c_array_result;
    auto result = hstack<float_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayColumnStackTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ c_array_2d1[2][3] = {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}};
    Array<float_> array1{c_array_2d1};
    float_ c_array_2d_result[2][6] = {{1.1, 2.2, 3.3, 7.7, 8.8, 9.9},
                                      {4.4, 5.5, 6.6, 10.1, 11.11, 12.12}};
    Array<float_> result_sample{c_array_2d_result};
    auto result = column_stack<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayC_Test) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    float_ c_array_hstack[2][2] = {{7.7, 8.8}, {9.9, 10.1}};
    Array<float_> array_c_{c_array_hstack};
    float_ c_array_result[2][5] = {{1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9, 10.1}};
    Array<float_> result_sample = c_array_result;
    auto result = c_<float_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayHSplitTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    auto result = hsplit<float_>(array, 3);
    float_ c_array_2d_1[2][1] = {{1.1}, {4.4}};
    Array<float_> result0_sample = c_array_2d_1;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_2[2][1] = {{2.2}, {5.5}};
    Array<float_> result1_sample = c_array_2d_2;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_3[2][1] = {{3.3}, {6.6}};
    Array<float_> result2_sample = c_array_2d_3;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayVSplitTest) {
    /*
    a=np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
     */
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array = c_array_2d;
    auto result = vsplit<float_>(array, 2);
    float_ c_array_2d_0[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    float_ c_array_2d_1[1][3] = {{4.4, 5.5, 6.6}};
    Array<float_> result1_sample{c_array_2d_1};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayTransposeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_result_2d[3][2] = {{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
    Array<string_> result_sample = c_array_result_2d;
    auto result = transpose<string_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayRavelTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayReshapeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_2d_result[3][2] = {{"str1", "str2"}, {"str3", "str4"}, {"str5", "str6"}};
    Array<string_> result_sample = c_array_2d_result;
    Shape shape{3, 2};
    auto result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayResizeTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_2d_result[3][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str1", "str2", "str3"}};
    Array<string_> result_sample = c_array_2d_result;
    Shape shape{3, 3};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayAppendTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    Array<string_> array_append{"str7", "str8", "str9"};
    Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7", "str8", "str9"};
    auto result = array.append(array_append);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayInsertTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    Array<string_> array_insert{"str7", "str8", "str9"};
    Array<string_> result_sample{"str1", "str7", "str8", "str9", "str2", "str3", "str4", "str5", "str6"};
    auto result = array.insert(1, array_insert);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayDelTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
    auto result = array.del(1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayConcatenateTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {
            {"str1", "str2", "str3"},
            {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array = c_array_2d;
    string_ c_array_2d_array_concatenate[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_, 2, 3> array_concatenate = c_array_2d_array_concatenate;
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = array.concatenate(array_concatenate);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayVStackTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_vstack_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_> array_vstack{c_array_vstack_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = vstack<string_>(array, array_vstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayR_Test) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_r_2d[2][3] = {
            {"str7", "str8", "str9"},
            {"str10", "str11", "str12"}};
    Array<string_> array_r_{c_array_r_2d};
    string_ c_array_2d_result[4][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = r_<string_>(array, array_r_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayHStackTest) {
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
    auto result = hstack<string_>(array, array_hstack);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayColumnStackTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    string_ c_array_2d1[2][3] = {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}};
    Array<string_> array1{c_array_2d1};
    string_ c_array_2d_result[2][6] = {{"str1", "str2", "str3", "str7", "str8", "str9"},
                                       {"str4", "str5", "str6", "str10", "str11", "str12"}};
    Array<string_> result_sample{c_array_2d_result};
    auto result = column_stack<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayC_Test) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    string_ c_array_hstack[2][2] = {{"str7", "str8"}, {"str9", "str10"}};
    Array<string_> array_c_{c_array_hstack};
    string_ c_array_result[2][5] = {{"str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9", "str10"}};
    Array<string_> result_sample = c_array_result;
    auto result = c_<string_>(array, array_c_);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayHSplitTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    auto result = hsplit<string_>(array, 3);
    string_ c_array_2d_0[2][1] = {{"str1"}, {"str4"}};
    Array<string_> result0_sample = c_array_2d_0;
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_1[2][1] = {{"str2"}, {"str5"}};
    Array<string_> result1_sample = c_array_2d_1;
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_2[2][1] = {{"str3"}, {"str6"}};
    Array<string_> result2_sample = c_array_2d_2;
    equals = array_equal(result[2], result2_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic2DStringArrayVSplitTest) {
    /*
    a=np.array([['str1', 'str2', 'str3'], ['str4', 'str5', 'str6']], dtype='string_')
     */
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array = c_array_2d;
    auto result = vsplit<string_>(array, 2);
    string_ c_array_2d_0[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result0_sample{c_array_2d_0};
    bool equals = array_equal(result[0], result0_sample);
    EXPECT_TRUE(equals);
    string_ c_array_2d_1[1][3] = {{"str4", "str5", "str6"}};
    Array<string_> result1_sample{c_array_2d_1};
    equals = array_equal(result[1], result1_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayTransposeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {
            {{1, 7},
             {4, 10}},
            {{2, 8},
             {5, 11}},
            {{3, 9},
             {6, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    auto result = transpose<int_, 2, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayRavelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    auto result = array.ravel();
    Array<int_, 12> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayReshapeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<int_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayResizeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayAppendTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<int_, 24> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayInsertTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_1d[24] = {1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_, 24> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayDelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_1d[11] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_, 11> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayConcatenateTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayVStackTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = vstack<int_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayR_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = r_<int_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayHStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack<int_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayColumnStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = column_stack<int_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayC_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2, 2, 3> array1{c_array_3d1};
    int_ c_array_3d_result[2][2][6] = {{{1, 2, 3, 13, 14, 15},
                                        {4, 5, 6, 16, 17, 18}},
                                       {{7, 8, 9, 19, 20, 21},
                                        {10, 11, 12, 22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = c_<int_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayHSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_result_sample_0[2][1][3] = {{{1, 2, 3}}, {{7, 8, 9}}};
    Array<int_, 2, 1, 3> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<int_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    int_ c_array_result_sample_1[2][1][3] = {{{4, 5, 6}}, {{10, 11, 12}}};
    Array<int_, 2, 1, 3> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DIntArrayVSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ c_array_3d_0[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_> result_sample_0{c_array_3d_0};
    int_ c_array_3d_1[1][2][3] = {{{7, 8, 9}, {10, 11, 12}}};
    Array<int_> result_sample_1{c_array_3d_1};

    auto result = vsplit<int_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayTransposeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 7.7},
             {4.4, 10.10}},
            {{2.2, 8.8},
             {5.5, 11.11}},
            {{3.3, 9.9},
             {6.6, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    auto result = transpose<float_, 2, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayRavelTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    auto result = array.ravel();
    Array<float_, 12> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayReshapeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<float_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayResizeTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayAppendTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<float_, 24> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
                              16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayInsertTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_1d[24] = {1.1, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    Array<float_, 24> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayDelTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_1d[11] = {1.1, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    Array<float_, 11> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayConcatenateTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 4, 2, 3> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayVStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 4, 2, 3> result_sample{c_array_result};
    auto result = vstack<float_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayR_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = r_<float_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayHStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}, {19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack<float_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayColumnStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}, {19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = column_stack<float_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayC_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 3> array1{c_array_3d1};
    float_ c_array_result[2][2][6] = {{{1.1, 2.2, 3.3, 13.13, 14.14, 15.15}, {4.4, 5.5, 6.6, 16.16, 17.17, 18.18}}, {{7.7, 8.8, 9.9, 19.19, 20.2, 21.21}, {10.10, 11.11, 12.12, 22.22, 23.23, 24.24}}};
    Array<float_, 2, 2, 6> result_sample{c_array_result};
    auto result = c_<float_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayHSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_result_sample_0[2][1][3] = {{{1.1, 2.2, 3.3}}, {{7.7, 8.8, 9.9}}};
    Array<float_, 2, 1, 3> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<float_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    float_ c_array_result_sample_1[2][1][3] = {{{4.4, 5.5, 6.6}}, {{10.10, 11.11, 12.12}}};
    Array<float_, 2, 1, 3> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DFloatArrayVSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ c_array_3d_0[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample_0{c_array_3d_0};
    auto result = vsplit<float_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    float_ c_array_3d_1[1][2][3] = {{{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_3d_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayTransposeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str7"},
             {"str4", "str10"}},
            {{"str2", "str8"},
             {"str5", "str11"}},
            {{"str3", "str9"},
             {"str6", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    auto result = transpose<string_, 2, 2, 3>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayRavelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    auto result = array.ravel();
    Array<string_, 12> compare{"str1", "str2", "str3",
                               "str4", "str5", "str6",
                               "str7", "str8", "str9",
                               "str10", "str11", "str12"};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayReshapeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<string_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayResizeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayAppendTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<string_, 24> compare{"str1", "str2", "str3",
                               "str4", "str5", "str6",
                               "str7", "str8", "str9",
                               "str10", "str11", "str12",
                               "str13", "str14", "str15",
                               "str16", "str17", "str18",
                               "str19", "str20", "str21",
                               "str22", "str23", "str24"};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayInsertTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    Array<string_, 24> result_sample{"str1",
                                     "str13", "str14", "str15",
                                     "str16", "str17", "str18",
                                     "str19", "str20", "str21",
                                     "str22", "str23", "str24",
                                     "str2", "str3",
                                     "str4", "str5", "str6",
                                     "str7", "str8", "str9",
                                     "str10", "str11", "str12"};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayDelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_1d[11] = {"str1", "str3",
                              "str4", "str5", "str6",
                              "str7", "str8", "str9",
                              "str10", "str11", "str12"};
    Array<string_, 11> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayConcatenateTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayVStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = vstack<string_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayR_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = r_<string_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayHStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}, {"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = hstack<string_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayColumnStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"},
                                        {"str4", "str5", "str6"},
                                        {"str13", "str14", "str15"},
                                        {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"},
                                        {"str10", "str11", "str12"},
                                        {"str19", "str20", "str21"},
                                        {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = column_stack<string_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayC_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2, 2, 3> array1{c_array_3d1};
    string_ c_array_result[2][2][6] = {{{"str1", "str2", "str3", "str13", "str14", "str15"},
                                        {"str4", "str5", "str6", "str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9", "str19", "str20", "str21"},
                                        {"str10", "str11", "str12", "str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = c_<string_, 2, 2, 3>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayHSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_result_sample_0[2][1][3] = {{{"str1", "str2", "str3"}}, {{"str7", "str8", "str9"}}};
    Array<string_, 2, 1, 3> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<string_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    string_ c_array_result_sample_1[2][1][3] = {{{"str4", "str5", "str6"}}, {{"str10", "str11", "str12"}}};
    Array<string_, 2, 1, 3> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, static3DStringArrayVSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2, 2, 3> array{c_array_3d};
    string_ c_array_3d_0[1][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}}};
    Array<string_> result_sample_0{c_array_3d_0};
    string_ c_array_3d_1[1][2][3] = {
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_3d_1};

    auto result = vsplit<string_, 2, 2, 3>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayTransposeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {
            {{1, 7},
             {4, 10}},
            {{2, 8},
             {5, 11}},
            {{3, 9},
             {6, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    auto result = transpose<int_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayRavelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto result = array.ravel();
    Array<int_> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayReshapeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<int_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayResizeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayAppendTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<int_> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayInsertTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_1d[24] = {1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayDelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_1d[11] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayConcatenateTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayVStackTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = vstack<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayR_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = r_<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayHStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayColumnStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = column_stack<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayC_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_3d_result[2][2][6] = {{{1, 2, 3, 13, 14, 15}, {4, 5, 6, 16, 17, 18}}, {{7, 8, 9, 19, 20, 21}, {10, 11, 12, 22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = c_<int_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayHSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_sample_0[2][1][3] = {{{1, 2, 3}}, {{7, 8, 9}}};
    Array<int_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<int_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    int_ c_array_result_sample_1[2][1][3] = {{{4, 5, 6}}, {{10, 11, 12}}};
    Array<int_> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DIntArrayVSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d_0[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_> result_sample_0{c_array_3d_0};
    int_ c_array_3d_1[1][2][3] = {{{7, 8, 9}, {10, 11, 12}}};
    Array<int_> result_sample_1{c_array_3d_1};

    auto result = vsplit<int_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayTransposeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 7.7},
             {4.4, 10.10}},
            {{2.2, 8.8},
             {5.5, 11.11}},
            {{3.3, 9.9},
             {6.6, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    auto result = transpose<float_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayRavelTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto result = array.ravel();
    Array<float_> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayReshapeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<float_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayResizeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayAppendTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    double c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<float_> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
                          16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayInsertTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_1d[24] = {1.1, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    Array<float_> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayDelTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_1d[11] = {1.1, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    Array<float_> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayConcatenateTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayVStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = vstack<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayR_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                   {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = r_<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayHStackTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}, {19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayColumnStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}, {19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = column_stack<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayC_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][2][6] = {{{1.1, 2.2, 3.3, 13.13, 14.14, 15.15}, {4.4, 5.5, 6.6, 16.16, 17.17, 18.18}}, {{7.7, 8.8, 9.9, 19.19, 20.2, 21.21}, {10.10, 11.11, 12.12, 22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = c_<float_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayHSplitTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_sample_0[2][1][3] = {{{1.1, 2.2, 3.3}}, {{7.7, 8.8, 9.9}}};
    Array<float_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<float_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    float_ c_array_result_sample_1[2][1][3] = {{{4.4, 5.5, 6.6}}, {{10.1, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayVSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d_0[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample_0{c_array_3d_0};
    auto result = vsplit<float_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    float_ c_array_3d_1[1][2][3] = {{{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_3d_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayTransposeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str7"},
             {"str4", "str10"}},
            {{"str2", "str8"},
             {"str5", "str11"}},
            {{"str3", "str9"},
             {"str6", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    auto result = transpose<string_>(array);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayRavelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    auto result = array.ravel();
    Array<string_> compare{"str1", "str2", "str3",
                           "str4", "str5", "str6",
                           "str7", "str8", "str9",
                           "str10", "str11", "str12"};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayReshapeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    Array<string_> result = array.reshape(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayResizeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample = c_array_result_3d;
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayAppendTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<string_> compare{"str1", "str2", "str3",
                           "str4", "str5", "str6",
                           "str7", "str8", "str9",
                           "str10", "str11", "str12",
                           "str13", "str14", "str15",
                           "str16", "str17", "str18",
                           "str19", "str20", "str21",
                           "str22", "str23", "str24"};
    bool equals = array_equal(result, compare);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayInsertTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    Array<string_, 24> result_sample{"str1",
                                     "str13", "str14", "str15",
                                     "str16", "str17", "str18",
                                     "str19", "str20", "str21",
                                     "str22", "str23", "str24",
                                     "str2", "str3",
                                     "str4", "str5", "str6",
                                     "str7", "str8", "str9",
                                     "str10", "str11", "str12"};
    auto result = array.insert(1, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayDelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_1d[11] = {"str1", "str3",
                              "str4", "str5", "str6",
                              "str7", "str8", "str9",
                              "str10", "str11", "str12"};
    Array<string_> array1{c_array_1d};
    auto result = array.del(1);
    bool equals = array_equal(result, array1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayConcatenateTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayVStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = vstack<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayR_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = r_<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayHStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}, {"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = hstack<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayColumnStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"},
                                        {"str4", "str5", "str6"},
                                        {"str13", "str14", "str15"},
                                        {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"},
                                        {"str10", "str11", "str12"},
                                        {"str19", "str20", "str21"},
                                        {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = column_stack<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayC_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_> array1{c_array_3d1};
    string_ c_array_result[2][2][6] = {{{"str1", "str2", "str3", "str13", "str14", "str15"},
                                        {"str4", "str5", "str6", "str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9", "str19", "str20", "str21"},
                                        {"str10", "str11", "str12", "str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = c_<string_>(array, array1);
    bool equals = array_equal(result, result_sample);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayHSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_result_sample_0[2][1][3] = {{{"str1", "str2", "str3"}}, {{"str7", "str8", "str9"}}};
    Array<string_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit<string_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    string_ c_array_result_sample_1[2][1][3] = {{{"str4", "str5", "str6"}}, {{"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_result_sample_1};
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

TEST_F(ArrayManipTest, dynamic3DStringArrayVSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_3d_0[1][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}}};
    Array<string_> result_sample_0{c_array_3d_0};
    string_ c_array_3d_1[1][2][3] = {
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_3d_1};

    auto result = vsplit<string_>(array, 2);
    bool equals = array_equal(result[0], result_sample_0);
    EXPECT_TRUE(equals);
    equals = array_equal(result[1], result_sample_1);
    EXPECT_TRUE(equals);
}

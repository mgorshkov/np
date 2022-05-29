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
#include <np/Manip.hpp>
#include <np/Comp.hpp>

using namespace np;

// dynamic arrays
class ArrayManipTest : public ::testing::Test {
protected:

};

TEST_F(ArrayManipTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    {
        auto result = transpose<int_>(array);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        bool equals = array_equal<int_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.reshape(shape);
        bool equals = array_equal<int_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.resize(shape);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        auto result = array.append(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        EXPECT_THROW(array.insert(1, array1), std::runtime_error);
    }
    {
        EXPECT_THROW(array.del(1), std::runtime_error);
    }
    {
        Array<int_> array1;
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        auto result = vstack<int_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        auto result = r_<int_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        auto result = hstack<int_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array1;
        EXPECT_THROW(column_stack<int_>(array, array1), std::runtime_error);
    }
    {
        Array<int_> array1;
        auto result = c_<int_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        EXPECT_THROW(hsplit<int_>(array, 1), std::runtime_error);
    }
    {
        EXPECT_THROW(vsplit<int_>(array, 1), std::runtime_error);
    }
}

TEST_F(ArrayManipTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    {
        auto result = transpose<float_>(array);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        bool equals = array_equal<float_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.reshape(shape);
        bool equals = array_equal<float_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.resize(shape);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        auto result = array.append(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        EXPECT_THROW(array.insert(1, array1), std::runtime_error);
    }
    {
        EXPECT_THROW(array.del(1), std::runtime_error);
    }
    {
        Array<float_> array1;
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        auto result = vstack<float_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        auto result = r_<float_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        auto result = hstack<float_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array1;
        EXPECT_THROW(column_stack<float_>(array, array1), std::runtime_error);
    }
    {
        Array<float_> array1;
        auto result = c_<float_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        EXPECT_THROW(hsplit<float_>(array, 1), std::runtime_error);
    }
    {
        EXPECT_THROW(vsplit<float_>(array, 1), std::runtime_error);
    }
}

TEST_F(ArrayManipTest, dynamicEmptyStringArrayTest) {
    Array<string_> array{};
    {
        auto result = transpose<string_>(array);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        bool equals = array_equal<string_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.reshape(shape);
        bool equals = array_equal<string_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.resize(shape);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        auto result = array.append(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        EXPECT_THROW(array.insert(1, array1), std::runtime_error);
    }
    {
        EXPECT_THROW(array.del(1), std::runtime_error);
    }
    {
        Array<string_> array1;
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        auto result = vstack<string_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        auto result = r_<string_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        auto result = hstack<string_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array1;
        EXPECT_THROW(column_stack<string_>(array, array1), std::runtime_error);
    }
    {
        Array<string_> array1;
        auto result = c_<string_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        EXPECT_THROW(hsplit<string_>(array, 1), std::runtime_error);
    }
    {
        EXPECT_THROW(vsplit<string_>(array, 1), std::runtime_error);
    }
}

TEST_F(ArrayManipTest, dynamicEmptyUnicodeArrayTest) {
    Array<unicode_> array{};
    {
        auto result = transpose<unicode_>(array);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        bool equals = array_equal<unicode_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.reshape(shape);
        bool equals = array_equal<unicode_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Shape shape;
        auto result = array.resize(shape);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        auto result = array.append(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        EXPECT_THROW(array.insert(1, array1), std::runtime_error);
    }
    {
        EXPECT_THROW(array.del(1), std::runtime_error);
    }
    {
        Array<unicode_> array1;
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        auto result = vstack<unicode_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        auto result = r_<unicode_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        auto result = hstack<unicode_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<unicode_> array1;
        EXPECT_THROW(column_stack<unicode_>(array, array1), std::runtime_error);
    }
    {
        Array<unicode_> array1;
        auto result = c_<unicode_>(array, array1);
        bool equals = array_equal(result, array);
        EXPECT_TRUE(equals);
    }
    {
        EXPECT_THROW(hsplit<unicode_>(array, 1), std::runtime_error);
    }
    {
        EXPECT_THROW(vsplit<unicode_>(array, 1), std::runtime_error);
    }
}

TEST_F(ArrayManipTest, static1DIntArrayTest) {
    // static
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> result_sample{1, 2, 3};
        auto result = transpose<int_, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        auto result = array.ravel();
        bool equals = array_equal<int_, 3>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        int_ c_array_2d[3][1] = {{1}, {2}, {3}};
        Array<int_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<int_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_> result_sample{1, 2, 3, 1};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_append{4, 5, 6};
        Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_insert{4, 5, 6};
        Array<int_, 6> result_sample{1, 4, 5, 6, 2, 3};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_> result_sample{1, 3};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_concatenate{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_vstack{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = vstack<int_, 3>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_r_{4, 5, 6};
        Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
        auto result = r_<int_, 3>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_hstack{4, 5, 6};
        Array<int_, 6> result_sample{1, 2, 3, 4, 5, 6};
        auto result = hstack<int_, 3>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_column_stack{4, 5, 6};
        int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
        Array<int_> result_sample = array_2D;
        auto result = column_stack<int_, 3>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        Array<int_, 3> array_c_{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = c_<int_, 3>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 3> array{1, 2, 3};
        auto result = hsplit<int_, 3>(array, 1);
        Array<int_> result0_sample{1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<int_> result1_sample{2, 3};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static1DFloatArrayTest) {
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> result_sample{1.1, 2.2, 3.3};
        auto result = transpose<float_, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        auto result = array.ravel();
        bool equals = array_equal<float_, 3>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
        Array<float_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_append{4.4, 5.5, 6.6};
        Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_insert{4.4, 5.5, 6.6};
        Array<float_, 6> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 2> result_sample{1.1, 3.3};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_concatenate{4.4, 5.5, 6.6};
        Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_vstack{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = vstack<float_, 3>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_r_{4.4, 5.5, 6.6};
        Array<float_, 6> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = r_<float_, 3>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_hstack{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = hstack<float_, 3>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_column_stack{4.4, 5.5, 6.6};
        float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
        Array<float_> result_sample = array_2D;
        auto result = column_stack<float_, 3>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        Array<float_, 3> array_c_{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = c_<float_, 3>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_, 3> array{1.1, 2.2, 3.3};
        auto result = hsplit<float_, 3>(array, 1);
        Array<float_> result0_sample{1.1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<float_> result1_sample{2.2, 3.3};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static1DStringArrayTest) {
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> result_sample{"str1", "str2", "str3"};
        auto result = transpose<string_, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        auto result = array.ravel();
        bool equals = array_equal<string_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
        Array<string_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_> result_sample{"str1", "str2", "str3", "str1"};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_append{"str4", "str5", "str6"};
        Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_insert{"str4", "str5", "str6"};
        Array<string_, 6> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 2> result_sample{"str1", "str3"};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_concatenate{"str4", "str5", "str6"};
        Array<string_, 6> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_vstack{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = vstack<string_, 3>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_r_{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = r_<string_, 3>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_hstack{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = hstack<string_, 3>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_column_stack{"str4", "str5", "str6"};
        string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
        Array<string_> result_sample = array_2D;
        auto result = column_stack<string_, 3>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        Array<string_, 3> array_c_{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = c_<string_, 3>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_, 3> array{"str1", "str2", "str3"};
        auto result = hsplit<string_, 3>(array, 1);
        Array<string_> result0_sample{"str1"};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<string_> result1_sample{"str2", "str3"};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic1DIntArrayTest) {
    // dynamic
    {
        Array<int_> array{1, 2, 3};
        Array<int_> result_sample{1, 2, 3};
        auto result = transpose<int_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        auto result = array.ravel();
        bool equals = array_equal<int_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        int_ c_array_2d[3][1] = {{1}, {2}, {3}};
        Array<int_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<int_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> result_sample{1, 2, 3, 1};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_append{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_insert{4, 5, 6};
        Array<int_> result_sample{1, 4, 5, 6, 2, 3};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> result_sample{1, 3};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_concatenate{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_vstack{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = vstack<int_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_r_{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = r_<int_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_hstack{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = hstack<int_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_column_stack{4, 5, 6};
        int_ array_2D[3][2]{{1, 4}, {2, 5}, {3, 6}};
        Array<int_> result_sample = array_2D;
        auto result = column_stack<int_>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        Array<int_> array_c_{4, 5, 6};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        auto result = c_<int_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array{1, 2, 3};
        auto result = hsplit<int_>(array, 1);
        Array<int_> result0_sample{1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<int_> result1_sample{2, 3};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic1DFloatArrayTest) {
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> result_sample{1.1, 2.2, 3.3};
        auto result = transpose<float_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        auto result = array.ravel();
        bool equals = array_equal<float_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        float_ c_array_2d[3][1] = {{1.1}, {2.2}, {3.3}};
        Array<float_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> result_sample{1.1, 2.2, 3.3, 1.1};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_append{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_insert{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 4.4, 5.5, 6.6, 2.2, 3.3};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> result_sample{1.1, 3.3};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_concatenate{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_vstack{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = vstack<float_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_r_{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = r_<float_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_hstack{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = hstack<float_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_column_stack{4.4, 5.5, 6.6};
        float_ array_2D[3][2]{{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
        Array<float_> result_sample = array_2D;
        auto result = column_stack<float_>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        Array<float_> array_c_{4.4, 5.5, 6.6};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = c_<float_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array{1.1, 2.2, 3.3};
        auto result = hsplit<float_>(array, 1);
        Array<float_> result0_sample{1.1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<float_> result1_sample{2.2, 3.3};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic1DStringArrayTest) {
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> result_sample{"str1", "str2", "str3"};
        auto result = transpose<string_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        auto result = array.ravel();
        bool equals = array_equal<string_>(result, array);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        string_ c_array_2d[3][1] = {{"str1"}, {"str2"}, {"str3"}};
        Array<string_> result_sample = c_array_2d;
        Shape shape{3, 1};
        auto result = array.reshape(shape);
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> result_sample{"str1", "str2", "str3", "str1"};
        Shape shape{4};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_append{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_insert{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str4", "str5", "str6", "str2", "str3"};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> result_sample{"str1", "str3"};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_concatenate{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_vstack{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = vstack<string_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_r_{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = r_<string_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_hstack{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = hstack<string_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_column_stack{"str4", "str5", "str6"};
        string_ array_2D[3][2]{{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
        Array<string_> result_sample = array_2D;
        auto result = column_stack<string_>(array, array_column_stack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        Array<string_> array_c_{"str4", "str5", "str6"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        auto result = c_<string_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array{"str1", "str2", "str3"};
        auto result = hsplit<string_>(array, 1);
        Array<string_> result0_sample{"str1"};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        Array<string_> result1_sample{"str2", "str3"};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static2DIntArrayTest) {
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
        Array<int_> result_sample = c_array_result_2d;
        auto result = transpose<int_, 2, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        auto result = array.ravel();
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        bool equals = array_equal<int_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        Array<int_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<int_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
        Array<int_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_2d_array_append[2][3] = {{7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_append = c_array_2d_array_append;
        Array<int_, 12> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_2d_array_insert[2][3] = {{7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_insert = c_array_2d_array_insert;
        Array<int_, 12> result_sample{1, 7, 8, 9, 10, 11, 12, 2, 3, 4, 5, 6};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        Array<int_> result_sample{1, 3, 4, 5, 6};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_2d_array_concatenate[2][3] = {{7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_concatenate = c_array_2d_array_concatenate;
        Array<int_, 12> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_vstack_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_vstack{c_array_vstack_2d};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto result = vstack<int_, 2, 3>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_r_2d[2][3] = {{7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_r_{c_array_r_2d};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        auto result = r_<int_, 2, 3>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_hstack[2][3] = {{ 7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_hstack{c_array_hstack};
        int_ c_array_result[2][6] = {{ 1, 2, 3, 7, 8, 9}, {4, 5, 6, 10, 11, 12}};
        Array<int_> result_sample = c_array_result;
        auto result = hstack<int_, 2, 3>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        int_ c_array_hstack[2][3] = {{ 7, 8, 9}, {10, 11, 12}};
        Array<int_, 2, 3> array_c_{c_array_hstack};
        int_ c_array_result[2][6] = {{ 1, 2, 3, 7, 8, 9}, {4, 5, 6, 10,11, 12}};
        Array<int_> result_sample = c_array_result;
        auto result = c_<int_, 2, 3>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        auto result = hsplit<int_, 2, 3>(array, 1);
        int_ c_array_2d_1[2][1] = {{1}, {4}};
        Array<int_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        int_ c_array_2d_2[2][2] = {{2, 3}, {5, 6}};
        Array<int_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_, 2, 3> array = c_array_2d;
        auto result = vsplit<int_, 2, 3>(array, 1);
        int_ c_array_2d_1[1][3] = {{1, 2, 3}};
        Array<int_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        int_ c_array_2d_2[1][3] = {{4, 5, 6}};
        Array<int_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static2DFloatArrayTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    {
        Array<float_> array = c_array_2d;
        float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
        Array<float_> result_sample = c_array_result_2d;
        auto result = transpose<float_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = array.ravel();
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
        Array<float_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
        Array<float_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_append{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_insert{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 7.7, 8.8, 9.9, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_concatenate{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_vstack_2d[1][3] = {{7.7, 8.8, 9.9}};
        Array<float_> array_vstack{c_array_vstack_2d};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = vstack<float_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_r_2d[1][3] = {{7.7, 8.8, 9.9}};
        Array<float_> array_r_{c_array_r_2d};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = r_<float_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_hstack[2][2] = {{ 7.7, 8.8}, {9.9, 10.1}};
        Array<float_> array_hstack{c_array_hstack};
        float_ c_array_result[2][5] = {{ 1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9,10.1}};
        Array<float_> result_sample = c_array_result;
        auto result = hstack<float_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_hstack[2][2] = {{ 7.7, 8.8}, {9.9, 10.1}};
        Array<float_> array_c_{c_array_hstack};
        float_ c_array_result[2][5] = {{ 1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9,10.1}};
        Array<float_> result_sample = c_array_result;
        auto result = c_<float_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = hsplit<float_>(array, 1);
        float_ c_array_2d_1[2][1] = {{1.1}, {4.4}};
        Array<float_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        float_ c_array_2d_2[2][2] = {{2.2, 3.3}, {5.5, 6.6}};
        Array<float_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = vsplit<float_>(array, 1);
        float_ c_array_2d_1[1][3] = {{1.1, 2.2, 3.3}};
        Array<float_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        float_ c_array_2d_2[1][3] = {{4.4, 5.5, 6.6}};
        Array<float_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static2DStringArrayTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    {
        Array<string_> array = c_array_2d;
        string_ c_array_result_2d[3][2] = {{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
        Array<string_> result_sample = c_array_result_2d;
        auto result = transpose<string_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = array.ravel();
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_2d_result[3][2] = {{"str1", "str2"}, {"str3", "str4"}, {"str5", "str6"}};
        Array<string_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_2d_result[3][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str1", "str2", "str3"}};
        Array<string_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_append{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_insert{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str7", "str8", "str9", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_concatenate{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_vstack_2d[1][3] = {{"str7", "str8", "str9"}};
        Array<string_> array_vstack{c_array_vstack_2d};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = vstack<string_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_r_2d[1][3] = {{"str7", "str8", "str9"}};
        Array<string_> array_r_{c_array_r_2d};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = r_<string_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_hstack[2][2] = {{ "str7", "str8"}, {"str9", "str10"}};
        Array<string_> array_hstack{c_array_hstack};
        string_ c_array_result[2][5] = {{ "str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9","str10"}};
        Array<string_> result_sample = c_array_result;
        auto result = hstack<string_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_hstack[2][2] = {{ "str7", "str8"}, {"str9", "str10"}};
        Array<string_> array_c_{c_array_hstack};
        string_ c_array_result[2][5] = {{ "str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9","str10"}};
        Array<string_> result_sample = c_array_result;
        auto result = c_<string_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = hsplit<string_>(array, 1);
        string_ c_array_2d_1[2][1] = {{"str1"}, {"str4"}};
        Array<string_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        string_ c_array_2d_2[2][2] = {{"str2", "str3"}, {"str5", "str6"}};
        Array<string_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = vsplit<string_>(array, 1);
        string_ c_array_2d_1[1][3] = {{"str1", "str2", "str3"}};
        Array<string_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        string_ c_array_2d_2[1][3] = {{"str4", "str5", "str6"}};
        Array<string_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic2DIntArrayTest) {
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    {
        Array<int_> array = c_array_2d;
        int_ c_array_result_2d[3][2] = {{1, 4}, {2, 5}, {3, 6}};
        Array<int_> result_sample = c_array_result_2d;
        auto result = transpose<int_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        auto result = array.ravel();
        Array<int_> result_sample{1, 2, 3, 4, 5, 6};
        bool equals = array_equal<int_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_2d_result[3][2] = {{1, 2}, {3, 4}, {5, 6}};
        Array<int_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<int_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_2d_result[3][3] = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};
        Array<int_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        Array<int_> array_append{7, 8, 9};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        Array<int_> array_insert{7, 8, 9};
        Array<int_> result_sample{1, 7, 8, 9, 2, 3, 4, 5, 6};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        Array<int_> result_sample{1, 3, 4, 5, 6};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        Array<int_> array_concatenate{7, 8, 9};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_vstack_2d[1][3] = {{7, 8, 9}};
        Array<int_> array_vstack{c_array_vstack_2d};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto result = vstack<int_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_r_2d[1][3] = {{7, 8, 9}};
        Array<int_> array_r_{c_array_r_2d};
        Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto result = r_<int_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_hstack[2][2] = {{ 7, 8}, {9, 10}};
        Array<int_> array_hstack{c_array_hstack};
        int_ c_array_result[2][5] = {{ 1, 2, 3, 7, 8}, {4, 5, 6, 9,10}};
        Array<int_> result_sample = c_array_result;
        auto result = hstack<int_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        int_ c_array_hstack[2][2] = {{ 7, 8}, {9, 10}};
        Array<int_> array_c_{c_array_hstack};
        int_ c_array_result[2][5] = {{ 1, 2, 3, 7, 8}, {4, 5, 6, 9,10}};
        Array<int_> result_sample = c_array_result;
        auto result = c_<int_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        auto result = hsplit<int_>(array, 1);
        int_ c_array_2d_1[2][1] = {{1}, {4}};
        Array<int_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        int_ c_array_2d_2[2][2] = {{2, 3}, {5, 6}};
        Array<int_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<int_> array = c_array_2d;
        auto result = vsplit<int_>(array, 1);
        int_ c_array_2d_1[1][3] = {{1, 2, 3}};
        Array<int_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        int_ c_array_2d_2[1][3] = {{4, 5, 6}};
        Array<int_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic2DFloatArrayTest) {
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    {
        Array<float_> array = c_array_2d;
        float_ c_array_result_2d[3][2] = {{1.1, 4.4}, {2.2, 5.5}, {3.3, 6.6}};
        Array<float_> result_sample = c_array_result_2d;
        auto result = transpose<float_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = array.ravel();
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_2d_result[3][2] = {{1.1, 2.2}, {3.3, 4.4}, {5.5, 6.6}};
        Array<float_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<float_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_2d_result[3][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {1.1, 2.2, 3.3}};
        Array<float_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_append{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_insert{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 7.7, 8.8, 9.9, 2.2, 3.3, 4.4, 5.5, 6.6};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> result_sample{1.1, 3.3, 4.4, 5.5, 6.6};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        Array<float_> array_concatenate{7.7, 8.8, 9.9};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_vstack_2d[1][3] = {{7.7, 8.8, 9.9}};
        Array<float_> array_vstack{c_array_vstack_2d};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = vstack<float_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_r_2d[1][3] = {{7.7, 8.8, 9.9}};
        Array<float_> array_r_{c_array_r_2d};
        Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9};
        auto result = r_<float_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_hstack[2][2] = {{ 7.7, 8.8}, {9.9, 10.1}};
        Array<float_> array_hstack{c_array_hstack};
        float_ c_array_result[2][5] = {{ 1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9,10.1}};
        Array<float_> result_sample = c_array_result;
        auto result = hstack<float_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        float_ c_array_hstack[2][2] = {{ 7.7, 8.8}, {9.9, 10.1}};
        Array<float_> array_c_{c_array_hstack};
        float_ c_array_result[2][5] = {{ 1.1, 2.2, 3.3, 7.7, 8.8}, {4.4, 5.5, 6.6, 9.9,10.1}};
        Array<float_> result_sample = c_array_result;
        auto result = c_<float_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = hsplit<float_>(array, 1);
        float_ c_array_2d_1[2][1] = {{1.1}, {4.4}};
        Array<float_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        float_ c_array_2d_2[2][2] = {{2.2, 3.3}, {5.5, 6.6}};
        Array<float_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<float_> array = c_array_2d;
        auto result = vsplit<float_>(array, 1);
        float_ c_array_2d_1[1][3] = {{1.1, 2.2, 3.3}};
        Array<float_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        float_ c_array_2d_2[1][3] = {{4.4, 5.5, 6.6}};
        Array<float_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic2DStringArrayTest) {
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    {
        Array<string_> array = c_array_2d;
        string_ c_array_result_2d[3][2] = {{"str1", "str4"}, {"str2", "str5"}, {"str3", "str6"}};
        Array<string_> result_sample = c_array_result_2d;
        auto result = transpose<string_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = array.ravel();
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6"};
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_2d_result[3][2] = {{"str1", "str2"}, {"str3", "str4"}, {"str5", "str6"}};
        Array<string_> result_sample = c_array_2d_result;
        Shape shape{3, 2};
        auto result = array.reshape(shape);
        bool equals = array_equal<string_>(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_2d_result[3][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str1", "str2", "str3"}};
        Array<string_> result_sample = c_array_2d_result;
        Shape shape{3, 3};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_append{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = array.append(array_append);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_insert{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str7", "str8", "str9", "str2", "str3", "str4", "str5", "str6"};
        auto result = array.insert(1, array_insert);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> result_sample{"str1", "str3", "str4", "str5", "str6"};
        auto result = array.del(1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        Array<string_> array_concatenate{"str7", "str8", "str9"};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = array.concatenate(array_concatenate);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_vstack_2d[1][3] = {{"str7", "str8", "str9"}};
        Array<string_> array_vstack{c_array_vstack_2d};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = vstack<string_>(array, array_vstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_r_2d[1][3] = {{"str7", "str8", "str9"}};
        Array<string_> array_r_{c_array_r_2d};
        Array<string_> result_sample{"str1", "str2", "str3", "str4", "str5", "str6", "str7","str8", "str9"};
        auto result = r_<string_>(array, array_r_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_hstack[2][2] = {{ "str7", "str8"}, {"str9", "str10"}};
        Array<string_> array_hstack{c_array_hstack};
        string_ c_array_result[2][5] = {{ "str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9","str10"}};
        Array<string_> result_sample = c_array_result;
        auto result = hstack<string_>(array, array_hstack);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        string_ c_array_hstack[2][2] = {{ "str7", "str8"}, {"str9", "str10"}};
        Array<string_> array_c_{c_array_hstack};
        string_ c_array_result[2][5] = {{ "str1", "str2", "str3", "str7", "str8"}, {"str4", "str5", "str6", "str9","str10"}};
        Array<string_> result_sample = c_array_result;
        auto result = c_<string_>(array, array_c_);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = hsplit<string_>(array, 1);
        string_ c_array_2d_1[2][1] = {{"str1"}, {"str4"}};
        Array<string_> result0_sample = c_array_2d_1;
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        string_ c_array_2d_2[2][2] = {{"str2", "str3"}, {"str5", "str6"}};
        Array<string_> result1_sample = c_array_2d_2;
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
    {
        Array<string_> array = c_array_2d;
        auto result = vsplit<string_>(array, 1);
        string_ c_array_2d_1[1][3] = {{"str1", "str2", "str3"}};
        Array<string_> result0_sample{c_array_2d_1};
        bool equals = array_equal(result[0], result0_sample);
        EXPECT_TRUE(equals);
        string_ c_array_2d_2[1][3] = {{"str4", "str5", "str6"}};
        Array<string_> result1_sample{c_array_2d_2};
        equals = array_equal(result[1], result1_sample);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    {
        int_ c_array_result_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        auto result = transpose<int_, 2, 2, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<int_, 12> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<int_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        auto result = array.append(array1);
        Array<int_, 24> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_1d[11] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11};
        Array<int_, 11> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = vstack<int_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = r_<int_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = hstack<int_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = column_stack<int_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample{c_array_1d};
        auto result = c_<int_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample_1{c_array_1d_1};
        auto result = hsplit<int_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        long c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_, 2, 2, 3> array1{c_array_3d1};
        long c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample_1{c_array_1d_1};
        auto result = vsplit<int_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        long c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{
        {1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
    {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}
    };
    Array<float_, 2, 2, 3> array{c_array_3d};
    {
        float_ c_array_result_3d[2][2][3] = {
            {
                {1, 2, 3},
                {4, 5, 6}
            },
            {
                {7, 8, 9},
                {10, 11, 12}
            }
        };
        Array<float_, 2, 2, 3> result_sample = c_array_result_3d;
        auto result = transpose<float_, 2, 2, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<float_, 12> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_result_3d[3][2][2] = {
            {
                {1, 2},
                {3, 4}
            },
            {
                {5, 6},
                {7, 8}
            },
            {
                {9, 10},
                {11, 12}
            }
        };
        Array<float_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<float_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_result_3d[3][2][2] = {
            {
                {1, 2},
                {3, 4}
            },
            {
                {5, 6},
                {7, 8}
            },
            {
                {9, 10},
                {11, 12}
            }
        };
        Array<float_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        double c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        auto result = array.append(array1);
        Array<float_, 24> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
            16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_1d[11] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11};
        Array<float_, 11> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = vstack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = r_<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = hstack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = column_stack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = c_<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_1{c_array_1d_1};
        auto result = hsplit<float_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        float_ c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_1{c_array_1d_1};
        auto result = vsplit<float_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        float_ c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, static3DStringArrayTest) {
    string_ c_array_3d[2][2][3] = {
        {
            {"str1_1", "str1_2", "str1_3"},
            {"str2_1", "str2_2", "str2_3"}
        },
        {
            { "str5_1", "str5_2", "str5_3" },
            { "str6_1", "str6_2", "str6_3" }
        }
    };
    Array<string_, 2, 2, 3> array{c_array_3d};
    {
        string_ c_array_result_3d[2][2][3] = {
            {
                {"str1", "str2", "str3"},
                {"str4", "str5", "str6"}
            },
            {
                { "str7", "str8", "str9" },
                { "str10", "str11", "str12" }
            }
        };
        Array<string_, 2, 2, 3> result_sample = c_array_result_3d;
        auto result = transpose<string_, 2, 2, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<string_, 12> compare{"str1", "str2", "str3",
            "str4", "str5", "str6",
            "str7", "str8", "str9",
            "str10", "str11", "str12"};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_result_3d[3][2][2] = {
            {
            {"str1", "str2"},
            {"str3", "str4"}
            },
            {
                {"str5", "str6"},
                {"str7", "str8"}
            },
            {
                { "str9", "str10"},
                { "str11", "str12"}
            }
        };
        Array<string_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<string_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_result_3d[3][2][2] = {
            {
                {"str1", "str2"},
                {"str3", "str4"}
            },
            {
                {"str5", "str6"},
                {"str7", "str8"}
            },
            {
                { "str9", "str10"},
                { "str11", "str12"}
            }
        };
        Array<string_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
            {
            {"str9_1", "str9_2", "str9_3"},
            {"str10_1", "str10_2", "str10_3"}
            },
            {
            { "str13_1", "str13_2", "str13_3" },
            { "str14_1", "str14_2", "str14_3" }
            }
        };
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
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> c_array_1d{"str1", "str2", "str3",
                                   "str4", "str5", "str6",
                                   "str7", "str8", "str9",
                                   "str10", "str11", "str12",
                                   "str13", "str14", "str15",
                                   "str16", "str17", "str18",
                                   "str19", "str20", "str21",
                                   "str22", "str23", "str24"};
        Array<string_, 24> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_1d[11] = {"str1", "str2", "str3",
                                  "str4", "str5", "str6",
                                  "str7", "str8", "str9",
                                  "str10", "str11"};
        Array<string_, 11> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                      "str4", "str5", "str6",
                                      "str7", "str8", "str9",
                                      "str10", "str11", "str12",
                                      "str13", "str14", "str15",
                                      "str16", "str17", "str18",
                                      "str19", "str20", "str21",
                                      "str22", "str23", "str24"};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = vstack<string_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = r_<string_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = hstack<string_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = column_stack<string_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = c_<string_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample_1{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = hsplit<string_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        Array<string_, 24> result_sample_2{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_, 2, 2, 3> array1{c_array_3d1};
        Array<string_, 24> result_sample_1{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = vsplit<string_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        Array<string_, 24> result_sample_2{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    {
        int_ c_array_result_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        auto result = transpose<int_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<int_> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<int_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
        Array<int_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        auto result = array.append(array1);
        Array<int_> compare{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_1d[11] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11};
        Array<int_> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = vstack<int_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = r_<int_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = hstack<int_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = column_stack<int_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample{c_array_1d};
        auto result = c_<int_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample_1{c_array_1d_1};
        auto result = hsplit<int_>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        long c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<int_> array1{c_array_3d1};
        long c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample_1{c_array_1d_1};
        auto result = vsplit<int_>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        long c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<int_> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{
                                          {1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}
    };
    Array<float_, 2, 2, 3> array{c_array_3d};
    {
        float_ c_array_result_3d[2][2][3] = {
                {
                        {1, 2, 3},
                        {4, 5, 6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        Array<float_, 2, 2, 3> result_sample = c_array_result_3d;
        auto result = transpose<float_, 2, 2, 3>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<float_, 12> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_result_3d[3][2][2] = {
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {5, 6},
                        {7, 8}
                },
                {
                        {9, 10},
                        {11, 12}
                }
        };
        Array<float_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<float_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_result_3d[3][2][2] = {
                {
                        {1, 2},
                        {3, 4}
                },
                {
                        {5, 6},
                        {7, 8}
                },
                {
                        {9, 10},
                        {11, 12}
                }
        };
        Array<float_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        double c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        auto result = array.append(array1);
        Array<float_, 24> compare{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
                                  16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_1d[11] = {1, 2, 4, 5, 6, 7, 8, 9, 10, 11};
        Array<float_, 11> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = vstack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = r_<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = hstack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = column_stack<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample{c_array_1d};
        auto result = c_<float_, 2, 2, 3>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_1{c_array_1d_1};
        auto result = hsplit<float_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        float_ c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        float_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
        Array<float_, 2, 2, 3> array1{c_array_3d1};
        float_ c_array_1d_1[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_1{c_array_1d_1};
        auto result = vsplit<float_, 2, 2, 3>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        float_ c_array_1d_2[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        Array<float_, 24> result_sample_2{c_array_1d_2};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}

TEST_F(ArrayManipTest, dynamic3DStringArrayTest) {
    string_ c_array_3d[2][2][3] = {
            {
                    {"str1_1", "str1_2", "str1_3"},
                    {"str2_1", "str2_2", "str2_3"}
            },
            {
                    { "str5_1", "str5_2", "str5_3" },
                    { "str6_1", "str6_2", "str6_3" }
            }
    };
    Array<string_> array{c_array_3d};
    {
        string_ c_array_result_3d[2][2][3] = {
                {
                        {"str1", "str2", "str3"},
                        {"str4", "str5", "str6"}
                },
                {
                        { "str7", "str8", "str9" },
                        { "str10", "str11", "str12" }
                }
        };
        Array<string_> result_sample = c_array_result_3d;
        auto result = transpose<string_>(array);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        auto result = array.ravel();
        Array<string_> compare{"str1", "str2", "str3",
                                   "str4", "str5", "str6",
                                   "str7", "str8", "str9",
                                   "str10", "str11", "str12"};
        bool equals = array_equal(result, compare);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_result_3d[3][2][2] = {
                {
                        {"str1", "str2"},
                        {"str3", "str4"}
                },
                {
                        {"str5", "str6"},
                        {"str7", "str8"}
                },
                {
                        { "str9", "str10"},
                        { "str11", "str12"}
                }
        };
        Array<string_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        Array<string_> result = array.reshape(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_result_3d[3][2][2] = {
                {
                        {"str1", "str2"},
                        {"str3", "str4"}
                },
                {
                        {"str5", "str6"},
                        {"str7", "str8"}
                },
                {
                        { "str9", "str10"},
                        { "str11", "str12"}
                }
        };
        Array<string_> result_sample = c_array_result_3d;
        Shape shape{3, 2, 2};
        auto result = array.resize(shape);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
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
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> c_array_1d{"str1", "str2", "str3",
                                      "str4", "str5", "str6",
                                      "str7", "str8", "str9",
                                      "str10", "str11", "str12",
                                      "str13", "str14", "str15",
                                      "str16", "str17", "str18",
                                      "str19", "str20", "str21",
                                      "str22", "str23", "str24"};
        Array<string_> result_sample{c_array_1d};
        auto result = array.insert(1, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_1d[11] = {"str1", "str2", "str3",
                                  "str4", "str5", "str6",
                                  "str7", "str8", "str9",
                                  "str10", "str11"};
        Array<string_> array1{c_array_1d};
        auto result = array.del(1);
        bool equals = array_equal(result, array1);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = array.concatenate(array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = vstack<string_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = r_<string_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = hstack<string_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = column_stack<string_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample{"str1", "str2", "str3",
                                         "str4", "str5", "str6",
                                         "str7", "str8", "str9",
                                         "str10", "str11", "str12",
                                         "str13", "str14", "str15",
                                         "str16", "str17", "str18",
                                         "str19", "str20", "str21",
                                         "str22", "str23", "str24"};
        auto result = c_<string_>(array, array1);
        bool equals = array_equal(result, result_sample);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample_1{"str1", "str2", "str3",
                                           "str4", "str5", "str6",
                                           "str7", "str8", "str9",
                                           "str10", "str11", "str12",
                                           "str13", "str14", "str15",
                                           "str16", "str17", "str18",
                                           "str19", "str20", "str21",
                                           "str22", "str23", "str24"};
        auto result = hsplit<string_>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        Array<string_> result_sample_2{"str1", "str2", "str3",
                                           "str4", "str5", "str6",
                                           "str7", "str8", "str9",
                                           "str10", "str11", "str12",
                                           "str13", "str14", "str15",
                                           "str16", "str17", "str18",
                                           "str19", "str20", "str21",
                                           "str22", "str23", "str24"};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
    {
        string_ c_array_3d1[2][2][3] = {
                {
                        {"str9_1", "str9_2", "str9_3"},
                        {"str10_1", "str10_2", "str10_3"}
                },
                {
                        { "str13_1", "str13_2", "str13_3" },
                        { "str14_1", "str14_2", "str14_3" }
                }
        };
        Array<string_> array1{c_array_3d1};
        Array<string_> result_sample_1{"str1", "str2", "str3",
                                           "str4", "str5", "str6",
                                           "str7", "str8", "str9",
                                           "str10", "str11", "str12",
                                           "str13", "str14", "str15",
                                           "str16", "str17", "str18",
                                           "str19", "str20", "str21",
                                           "str22", "str23", "str24"};
        auto result = vsplit<string_>(array, 1);
        bool equals = array_equal(result[0], result_sample_1);
        EXPECT_TRUE(equals);
        Array<string_> result_sample_2{"str1", "str2", "str3",
                                           "str4", "str5", "str6",
                                           "str7", "str8", "str9",
                                           "str10", "str11", "str12",
                                           "str13", "str14", "str15",
                                           "str16", "str17", "str18",
                                           "str19", "str20", "str21",
                                           "str22", "str23", "str24"};
        equals = array_equal(result[1], result_sample_2);
        EXPECT_TRUE(equals);
    }
}
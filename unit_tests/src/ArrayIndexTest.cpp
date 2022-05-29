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
#include <np/Comp.hpp>

using namespace np;

class ArrayIndexTest : public ::testing::Test {
protected:

};

TEST_F(ArrayIndexTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyStringArrayTest) {
    Array<string_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyUnicodeArrayTest) {
    Array<unicode_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, static1DIntArrayTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    auto slice = array[0];
    auto equal = array_equal<int_, 3>(slice, 1);
    EXPECT_TRUE(equal);
}

TEST_F(ArrayIndexTest, static1DFloatArrayTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    {
        auto slice = array[0];
        auto equal = array_equal<float_, 3>(slice, 1.1);
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, static1DStringArrayTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    auto slice = array[0];
    auto equal = array_equal<string_, 3>(slice, string_{"str1"});
    EXPECT_TRUE(equal);
}

TEST_F(ArrayIndexTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    {
        auto slice = array[0];
        auto equal = array_equal(slice, 1L);
        EXPECT_TRUE(equal);
    }
    {
        auto filtered = array["array <= 2"];
        auto equal = array_equal(filtered, Array<int_>{1, 2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, dynamic1DFloatArrayTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    {
        auto slice = array[0];
        auto equal = array_equal(slice, 1.1);
        EXPECT_TRUE(equal);
    }
    {
        auto filtered = array["array <= 2.2"];
        auto equal = array_equal(filtered, Array<float_>{1.1, 2.2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, dynamic1DStringArrayTest) {
    Array<string_> array{"str1", "str2", "str3"};
    auto slice = array[0];
    EXPECT_TRUE(array_equal(slice, string_{"str1"}));
}

TEST_F(ArrayIndexTest, static2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array{c_array_2d};
    Array<int_, 3> slice{1, 2, 3};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, static2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array{c_array_2d};
    Array<float_, 3> slice{1.1, 2.2, 3.3};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, static2DStringArrayTest) {
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array{c_array_2d};
    Array<string_, 3> slice{"str1", "str2", "str3"};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, dynamic2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    {
        Array<int_> slice{1, 2, 3};
        auto array0 = array[0];
        EXPECT_TRUE(array_equal(array0, slice));
    }
    {
        auto filtered = array["array <= 2"];
        auto equal = array_equal(filtered, Array<int_>{1, 2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, dynamic2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    {
        Array<float_> slice{1.1, 2.2, 3.3};
        EXPECT_TRUE(array_equal(array[0], slice));
    }
    {
        auto filtered = array["array <= 2.2"];
        auto equal = array_equal(filtered, Array<float_>{1.1, 2.2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, dynamic2DStringArrayTest) {
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    Array<string_> slice{"str1", "str2", "str3"};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> slice{c_array_2d};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, dynamic3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    {
        long c_array_2d[2][3] = {{1, 2, 3},
                                 {4, 5, 6}};
        Array<int_> slice{c_array_2d};
        EXPECT_TRUE(array_equal(array[0], slice));
    }
    {
        auto filtered = array["array <= 2"];
        auto equal = array_equal(filtered, Array<int_>{1, 2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> slice{c_array_2d};
    EXPECT_TRUE(array_equal(array[0], slice));
}

TEST_F(ArrayIndexTest, dynamic3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    {
        Array<float_> slice{c_array_2d};
        EXPECT_TRUE(array_equal(array[0], slice));
    }
    {
        auto filtered = array["array <= 3"];
        auto equal = array_equal(filtered, Array<float_>{1.1, 2.2});
        EXPECT_TRUE(equal);
    }
}

TEST_F(ArrayIndexTest, static3DStringArrayTest) {
    string_ c_array_3d[2][4][3] = {
        {
            {"str1_1", "str1_2", "str1_3"},
            {"str2_1", "str2_2", "str2_3"},
            {"str3_1", "str3_2", "str3_3"},
            {"str4_1", "str4_2", "str4_3"}
        },
        {
            { "str5_1", "str5_2", "str5_3" },
            { "str6_1", "str6_2", "str6_3" },
            { "str7_1", "str7_2", "str7_3" },
            { "str8_1", "str8_2", "str8_3" }
        }
    };
    Array<string_, 2, 4, 3> array{c_array_3d};
    string_ c_array_2d[4][3] = {
        {"str1_1", "str1_2", "str1_3"},
        {"str2_1", "str2_2", "str2_3"},
        {"str3_1", "str3_2", "str3_3"},
        {"str4_1", "str4_2", "str4_3"}
    };
    Array<string_, 4, 3> slice{c_array_2d};
    EXPECT_TRUE(array_equal(array[0], slice));
}

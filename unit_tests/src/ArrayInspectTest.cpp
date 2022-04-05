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
#include <np/Inspect.hpp>

using namespace np;

class ArrayInspectTest : public ::testing::Test {
protected:

};

TEST_F(ArrayInspectTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    EXPECT_EQ(std::vector<Size>{}, array.shape());
    auto l = len<int_>(array);
    EXPECT_EQ(0, l);
    auto n = array.ndim();
    EXPECT_EQ(0, n);
    auto s = array.size();
    EXPECT_EQ(0, s);
    auto d = array.dtype();
    auto is_int = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_int);
    auto converted = array.astype<float_>();

    EXPECT_EQ(std::vector<Size>{}, converted.shape());
    l = len<float_>(converted);
    EXPECT_EQ(0, l);
    n = converted.ndim();
    EXPECT_EQ(0, n);
    s = converted.size();
    EXPECT_EQ(0, s);
}

TEST_F(ArrayInspectTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    EXPECT_EQ(std::vector<Size>{}, array.shape());
    auto l = len<float_>(array);
    EXPECT_EQ(0, l);
    auto n = array.ndim();
    EXPECT_EQ(0, n);
    auto s = array.size();
    EXPECT_EQ(0, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);

    auto converted = array.astype<int_>();

    EXPECT_EQ(std::vector<Size>{}, converted.shape());
    l = len<int_>(converted);
    EXPECT_EQ(0, l);
    n = converted.ndim();
    EXPECT_EQ(0, n);
    s = converted.size();
    EXPECT_EQ(0, s);
}

TEST_F(ArrayInspectTest, dynamicEmptyStringArrayTest) {
    Array<string_> array{};
    EXPECT_EQ(std::vector<Size>{}, array.shape());
    auto l = len<string_>(array);
    EXPECT_EQ(0, l);
    auto n = array.ndim();
    EXPECT_EQ(0, n);
    auto s = array.size();
    EXPECT_EQ(0, s);
    auto d = array.dtype();
    auto is_string = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_string);
}

TEST_F(ArrayInspectTest, dynamicEmptyUnicodeArrayTest) {
    Array<unicode_> array{};
    EXPECT_EQ(std::vector<Size>{}, array.shape());
    auto l = len<unicode_>(array);
    EXPECT_EQ(0, l);
    auto n = array.ndim();
    EXPECT_EQ(0, n);
    auto s = array.size();
    EXPECT_EQ(0, s);
    auto d = array.dtype();
    auto is_unicode = std::is_same<decltype(d), unicode_>::value;
    EXPECT_TRUE(is_unicode);
}

TEST_F(ArrayInspectTest, static1DIntArrayTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<int_, 3>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_int = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_int);
}

TEST_F(ArrayInspectTest, static1DFloatArrayTest) {
    Array<float_, 3> array{1.1, 2.2, 3.3};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<float_, 3>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, static1DStringArrayTest) {
    Array<string_, 3> array{"str1", "str2", "str3"};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<string_, 3>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_unicode = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_unicode);
}

TEST_F(ArrayInspectTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<int_>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_int = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_int);
}

TEST_F(ArrayInspectTest, dynamic1DFloatArrayTest) {
    Array<float_> array{1.1, 2.2, 3.3};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<float_>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, dynamic1DStringArrayTest) {
    Array<string_> array{"str1", "str2", "str3"};
    EXPECT_EQ(std::vector<Size>{3}, array.shape());
    auto l = len<string_>(array);
    EXPECT_EQ(3, l);
    auto n = array.ndim();
    EXPECT_EQ(1, n);
    auto s = array.size();
    EXPECT_EQ(3, s);
    auto d = array.dtype();
    auto is_string = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_string);
}

TEST_F(ArrayInspectTest, static2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<int_, 2, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_int = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_int);
}

TEST_F(ArrayInspectTest, static2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<float_, 2, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, static2DStringArrayTest) {
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2, 3> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<string_, 2, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, dynamic2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<int_>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_int = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_int);
}

TEST_F(ArrayInspectTest, dynamic2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<float_>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, dynamic2DStringArrayTest) {
    std::string c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    std::vector<Size> sh{2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<string_>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(2, n);
    auto s = array.size();
    EXPECT_EQ(6, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    std::vector<Size> sh{2, 2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<int_, 2, 2, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(3, n);
    auto s = array.size();
    EXPECT_EQ(12, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), int_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    std::vector<Size> sh{2, 2, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<float_, 2, 2, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(3, n);
    auto s = array.size();
    EXPECT_EQ(12, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), float_>::value;
    EXPECT_TRUE(is_float);
}

TEST_F(ArrayInspectTest, static3DStringArrayTest) {
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
    std::vector<Size> sh{2, 4, 3};
    EXPECT_EQ(sh, array.shape());
    auto l = len<string_, 2, 4, 3>(array);
    EXPECT_EQ(2, l);
    auto n = array.ndim();
    EXPECT_EQ(3, n);
    auto s = array.size();
    EXPECT_EQ(24, s);
    auto d = array.dtype();
    auto is_float = std::is_same<decltype(d), string_>::value;
    EXPECT_TRUE(is_float);
}
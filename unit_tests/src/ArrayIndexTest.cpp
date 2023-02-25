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

class ArrayIndexTest : public ArrayTest {
protected:
};

TEST_F(ArrayIndexTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyFloatArrayTest) {
    // dynamic
    Array<float_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyStringArrayTest) {
    // dynamic
    Array<string_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamicEmptyUnicodeArrayTest) {
    // dynamic
    Array<unicode_> array{};
    EXPECT_THROW(auto slice = array[0], std::runtime_error);
}

TEST_F(ArrayIndexTest, static1DIntArraySubsettingTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    auto subset = array[0];
    Array<int_> result{1};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static1DIntArrayBooleanIndexingTest) {
    // static
    const Array<int_, 3> array{1, 2, 3};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static1DIntArraySlicingTest) {
    // static
    Array<int_, 3> array{1, 2, 3};
    auto slice = array["0:1"];
    Array<int_> result{1};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static1DFloatArraySubsettingTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    auto subset = array[0];
    Array<float_> result{1.1};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static1DFloatArrayBooleanIndexingTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static1DFloatArraySlicingTest) {
    // static
    Array<float_, 3> array{1.1, 2.2, 3.3};
    auto slice = array["0:1"];
    Array<float_> result{1.1};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static1DStringArraySubsettingTest) {
    // static
    Array<string_, 3> array{"str1", "str2", "str3"};
    auto subset = array[0];
    Array<string_> result{"str1"};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static1DStringArrayBooleanIndexingTest) {
    // static
    const Array<string_, 3> array{"str1", "str2", "str3"};
    auto booleanIndex = array["array != str2"];
    Array<string_> result{"str1", "str3"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static1DStringArraySlicingTest) {
    // static
    Array<string_, 3> array{"str1", "str2", "str3"};
    auto slice = array["0:1"];
    Array<string_> result{"str1"};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySubsettingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto subset = array[0];
    Array<int_> result{1};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArrayBooleanIndexingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySlicingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto slice = array["0:1"];
    Array<int_> result{1};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic1DFloatArraySubsettingTest) {
    // dynamic
    Array<float_> array{1.1, 2.2, 3.3};
    auto subset = array[0];
    Array<float_> result{1.1};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic1DFloatArrayBooleanIndexingTest) {
    // dynamic
    Array<float_> array{1.1, 2.2, 3.3};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic1DFloatArraySlicingTest) {
    // dynamic
    Array<float_> array{1.1, 2.2, 3.3};
    auto slice = array["0:1"];
    Array<float_> result{1.1};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic1DStringArraySubsettingTest) {
    // dynamic
    Array<string_> array{"str1", "str2", "str3"};
    auto subset = array[0];
    Array<string_> result{"str1"};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic1DStringArrayBooleanIndexingTest) {
    // dynamic
    Array<string_> array{"str1", "str2", "str3"};
    auto booleanIndex = array["array <= str2"];
    Array<string_> result{"str1", "str2"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic1DStringArraySlicingTest) {
    // dynamic
    Array<string_> array{"str1", "str2", "str3"};
    auto slice = array["0:1"];
    Array<string_> result{"str1"};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DIntArraySubsettingTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto subset = array[0];
    Array<int_> result{1, 2, 3};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static2DIntArrayBooleanIndexingTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static2DIntArraySlicingTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,"];
    int_ c_array_result[1][3] = {{1, 2, 3}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DFloatArraySubsettingTest) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto subset = array[0];
    Array<float_> result{1.1, 2.2, 3.3};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static2DFloatArrayBooleanIndexingTest) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static2DFloatArraySlicingTest) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,"];
    float_ c_array_result[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DFloatArraySlicingRangeTest) {
    static float_ kData[5][4] =
            {{5.1, 3.5, 1.4, 0.2},
             {4.9, 3., 1.4, 0.2},
             {4.7, 3.2, 1.3, 0.2},
             {4.6, 3.1, 1.5, 0.2},
             {5., 3.6, 1.4, 0.2}};
    Array<float_, 5 * 4> data{kData};
    auto slice = data["2:"];
    float_ c_array_result[3][4] = {{4.7, 3.2, 1.3, 0.2},
                                   {4.6, 3.1, 1.5, 0.2},
                                   {5., 3.6, 1.4, 0.2}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DStringArraySubsettingTest) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto subset = array[0];
    Array<string_> result{"str1", "str2", "str3"};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static2DStringArrayBooleanIndexingTest) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto booleanIndex = array["array <= str2"];
    Array<string_> result{"str1", "str2"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static2DStringArraySlicingTest) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,"];
    string_ c_array_result[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array[0];
    Array<int_> result{1, 2, 3};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArrayBooleanIndexingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySlicingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto slice = array["0:1,"];
    int_ c_array_result[1][3] = {{1, 2, 3}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySubsettingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto subset = array[0];
    Array<float_> result{1.1, 2.2, 3.3};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArrayBooleanIndexingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySlicingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto slice = array["0:1,"];
    float_ c_array_result[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySubsettingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto subset = array[0];
    Array<string_> result{"str1", "str2", "str3"};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArrayBooleanIndexingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto booleanIndex = array["array <= str2"];
    Array<string_> result{"str1", "str2"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySlicingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto slice = array["0:1,"];
    string_ c_array_result[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DIntArraySubsettingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto subset = array[0];
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static3DIntArrayBooleanIndexingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto booleanIndex = array["array <= 2"];
    int_ c_array_2d[2] = {1, 2};
    Array<int_> result{c_array_2d};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static3DIntArraySlicingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    int_ c_array_result[1][1][3] = {{{4, 5, 6}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[0];
    int_ c_array_2d[2][3] = {{1, 2, 3},
                             {4, 5, 6}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArrayBooleanIndexingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySlicingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    int_ c_array_result[1][1][3] = {{{4, 5, 6}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DFloatArraySubsettingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto subset = array[0];
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static3DFloatArrayBooleanIndexingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto booleanIndex = array["array <= 2.2"];
    float_ c_array_2d[2] = {1.1, 2.2};
    Array<float_> result{c_array_2d};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static3DFloatArraySlicingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    float_ c_array_result[1][1][3] = {{{4.4, 5.5, 6.6}}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DFloatArraySubsettingTest) {
    // dynamic
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto subset = array[0];
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DFloatArrayBooleanIndexingTest) {
    // dynamic
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto booleanIndex = array["array <= 3.3"];
    float_ c_array_2d[3] = {1.1, 2.2, 3.3};
    Array<float_> result{c_array_2d};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic3DFloatArraySlicingTest) {
    // dynamic
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    float_ c_array_2d[1][1][3] = {{{4.4, 5.5, 6.6}}};
    Array<float_> result{c_array_2d};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DStringArraySubsettingTest) {
    // static
    string_ c_array_3d[2][4][3] = {{{"str1", "str2", "str3"},
                                    {"str4", "str5", "str6"},
                                    {"str13", "str14", "str15"},
                                    {"str16", "str17", "str18"}},
                                   {{"str7", "str8", "str9"},
                                    {"str10", "str11", "str12"},
                                    {"str19", "str20", "str21"},
                                    {"str22", "str23", "str24"}}};
    Array<string_, 2 * 4 * 3> array{c_array_3d};
    auto subset = array[0];
    string_ c_array_2d[4][3] = {{"str1", "str2", "str3"},
                                {"str4", "str5", "str6"},
                                {"str13", "str14", "str15"},
                                {"str16", "str17", "str18"}};
    Array<string_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static3DStringArrayBooleanIndexingTest) {
    // static
    string_ c_array_3d[2][4][3] = {{{"str1", "str2", "str3"},
                                    {"str4", "str5", "str6"},
                                    {"str13", "str14", "str15"},
                                    {"str16", "str17", "str18"}},
                                   {{"str7", "str8", "str9"},
                                    {"str10", "str11", "str12"},
                                    {"str19", "str20", "str21"},
                                    {"str22", "str23", "str24"}}};
    Array<string_, 2 * 4 * 3> array{c_array_3d};
    auto booleanIndex = array["array <= str2"];
    string_ c_array_2d[12] = {"str1", "str2", "str13", "str14", "str15", "str16", "str17", "str18", "str10", "str11", "str12", "str19"};
    Array<string_> result{c_array_2d};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, static3DStringArraySlicingTest) {
    // static
    string_ c_array_3d[2][4][3] = {{{"str1", "str2", "str3"},
                                    {"str4", "str5", "str6"},
                                    {"str13", "str14", "str15"},
                                    {"str16", "str17", "str18"}},
                                   {{"str7", "str8", "str9"},
                                    {"str10", "str11", "str12"},
                                    {"str19", "str20", "str21"},
                                    {"str22", "str23", "str24"}}};
    Array<string_, 2 * 4 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    string_ c_array_2d[1][1][3] = {{{"str4", "str5", "str6"}}};
    Array<string_> result{c_array_2d};
    compare(slice, result);
}

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

TEST_F(ArrayIndexTest, dynamic1DIntArraySubsettingOutOfBoundTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    EXPECT_THROW(auto slice = array[3], std::runtime_error);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySlicingOutOfBoundTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    EXPECT_THROW(auto slice = array["1:,:1"], std::runtime_error);
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

TEST_F(ArrayIndexTest, static1DStringArraySlicingIterTest) {
    // static
    Array<string_, 3> array{"str1", "str2", "str3"};
    auto slice = array["0:1"];
    std::vector<string_> iterResult;
    for (const auto &i: slice)
        iterResult.emplace_back(i);
    EXPECT_EQ(iterResult, std::vector<string_>{"str1"});
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySubsettingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto subset = array[0];
    Array<int_> result{1};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySubsettingIterTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto subset = array[0];
    std::vector<int_> iterResult;
    for (auto i: subset)
        iterResult.emplace_back(i);
    EXPECT_EQ(iterResult, std::vector<int_>{1});
}

TEST_F(ArrayIndexTest, dynamic1DIntArrayBooleanIndexingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArrayBooleanIndexingIterTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto booleanIndex = array["array <= 2"];
    std::vector<int_> iterResult;
    for (auto i: booleanIndex)
        iterResult.emplace_back(i);
    std::vector<int_> result{1, 2};
    EXPECT_EQ(iterResult, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySlicingTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto slice = array["0:1"];
    Array<int_> result{1};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic1DIntArraySlicingIterTest) {
    // dynamic
    Array<int_> array{1, 2, 3};
    auto slice = array["0:1"];
    std::vector<int_> iterResult;
    for (auto i: slice)
        iterResult.emplace_back(i);
    std::vector<int_> result{1};
    EXPECT_EQ(iterResult, result);
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

TEST_F(ArrayIndexTest, dynamic1DFloatArrayHashingTest) {
    // dynamic
    {
        Array<float_> array{1.1, 2.2, 3.3, 1.1, 3.3};
        ndarray::array_dynamic::NDArrayDynamicIndexMap<float_, std::size_t> counts;
        for (np::Size i = 0; i < array.shape()[0]; ++i) {
            auto value = array[i];
            ++counts[value];
        }

        EXPECT_EQ(counts.size(), 3);
        EXPECT_EQ(counts[array[0]], 2);
        EXPECT_EQ(counts[array[1]], 1);
        EXPECT_EQ(counts[array[2]], 2);
    }
    {
        const Array<float_> array{1.1, 2.2, 3.3, 1.1, 3.3};
        ndarray::array_dynamic::NDArrayDynamicIndexConstMap<float_, std::size_t> counts;
        for (np::Size i = 0; i < array.shape()[0]; ++i) {
            auto value = array[i];
            ++counts[value];
        }

        EXPECT_EQ(counts.size(), 3);
        EXPECT_EQ(counts[array[0]], 2);
        EXPECT_EQ(counts[array[1]], 1);
        EXPECT_EQ(counts[array[2]], 2);
    }
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

    auto subset2 = subset[1];
    Array<int_> result2{2};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, static2DIntArraySubsettingTest2) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};

    auto subset = array["0,1"];
    Array<int_> result{2};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static2DIntArrayBooleanIndexingTest) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2"];
    Array<int_> result2{2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, static2DIntArraySlicingTest1) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};

    auto slice = array["1:,0:2"];
    int_ c_array_result[1][2] = {{4, 5}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DIntArraySlicingTest2) {
    // static
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};

    auto slice = array["0:2,1:"];
    int_ c_array_result[2][2] = {{2, 3}, {5, 6}};
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

    auto subset2 = subset[1];
    Array<float_> result2{2.2};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, static2DFloatArraySubsettingTest2) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto subset = array["0,1"];
    Array<float_> result{2.2};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, static2DFloatArrayBooleanIndexingTest) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2.2"];
    Array<float_> result2{2.2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, static2DFloatArraySlicingTest1) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,"];
    float_ c_array_result[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DFloatArraySlicingTest2) {
    // static
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,1:2"];
    float_ c_array_result[1][1] = {{2.2}};
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

    auto subset2 = subset[1];
    Array<string_> result2{"str2"};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, static2DStringArraySubsettingTest2) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};

    auto subset = array["0,1"];
    Array<string_> result{"str2"};
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

TEST_F(ArrayIndexTest, static2DStringArraySlicingTest1) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,"];
    string_ c_array_result[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static2DStringArraySlicingTest2) {
    // static
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_, 2 * 3> array{c_array_2d};
    auto slice = array["0:1,1:2"];
    string_ c_array_result[1][1] = {{"str2"}};
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

    auto subset2 = subset[1];
    Array<int_> result2{2};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest2) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array["0,1"];
    Array<int_> result{2};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArrayBooleanIndexingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2"];
    Array<int_> result2{2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingAndBooleanIndexingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto booleanIndex = array["0,array >= 2"];
    Array<int_> result{2, 3};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySlicingTest1) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto slice = array["0:1,"];
    int_ c_array_result[1][3] = {{1, 2, 3}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySlicingTest2) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto slice = array[":,0"];
    int_ c_array_result[2] = {1, 4};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySlicingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto slice = array["0:1,1:2"];
    int_ c_array_result[1][1] = {{2}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySlicingAndBooleanIndexingTest) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto booleanIndex = array["0:1,array >= 2"];
    Array<int_> result{2, 3};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySubsettingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto subset = array[0];
    Array<float_> result{1.1, 2.2, 3.3};
    compare(subset, result);

    auto subset2 = subset[1];
    Array<float_> result2{2.2};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySubsettingTest2) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto subset = array["0,1"];
    Array<float_> result{2.2};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArrayBooleanIndexingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto booleanIndex = array["array <= 2.2"];
    Array<float_> result{1.1, 2.2};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2.2"];
    Array<float_> result2{2.2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySubsettingAndBooleanIndexingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto booleanIndex = array["0,array >= 2"];
    Array<float_> result{2.2, 3.3};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySlicingTest1) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto slice = array["0:1,"];
    float_ c_array_result[1][3] = {{1.1, 2.2, 3.3}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySlicingTest2) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto slice = array["0:1,1:2"];
    float_ c_array_result[1][1] = {{2.2}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySlicingTest3) {
    // dynamic
    float_ c_array[2][2] = {{1.0, 2.0}, {0.0, 1.0}};
    Array<float_> array{c_array};
    auto slice = array["1:2,0:1"];
    float_ c_array_result[1][1] = {{0.0}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DFloatArraySlicingAndBooleanIndexingTest) {
    // dynamic
    float_ c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    auto booleanIndex = array["0:1,array >= 2"];
    Array<float_> result{2.2, 3.3};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySubsettingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto subset = array[0];
    Array<string_> result{"str1", "str2", "str3"};
    compare(subset, result);

    auto subset2 = subset[1];
    Array<string_> result2{"str2"};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySubsettingTest2) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto subset = array["0,1"];
    Array<string_> result{"str2"};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArrayBooleanIndexingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto booleanIndex = array["array <= str2"];
    Array<string_> result{"str1", "str2"};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= str2"];
    Array<string_> result2{"str2"};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySubsettingAndBooleanIndexingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto booleanIndex = array["0,array >= str2"];
    Array<string_> result{"str2", "str3"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySlicingTest1) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto slice = array["0:1,"];
    string_ c_array_result[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySlicingAndBooleanIndexingTest) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto booleanIndex = array["0:1,array >= str2"];
    Array<string_> result{"str2", "str3"};
    compare(booleanIndex, result);
}

TEST_F(ArrayIndexTest, dynamic2DStringArraySlicingTest2) {
    // dynamic
    string_ c_array_2d[2][3] = {{"str1", "str2", "str3"}, {"str4", "str5", "str6"}};
    Array<string_> array{c_array_2d};
    auto slice = array["0:1,1:2"];
    string_ c_array_result[1][1] = {{"str2"}};
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

    auto subset2 = subset[1];
    Array<int_> result2{4, 5, 6};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, static3DIntArrayBooleanIndexingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto booleanIndex = array["array <= 2"];
    int_ c_array_2d[2] = {1, 2};
    Array<int_> result{c_array_2d};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2"];
    Array<int_> result2{2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, static3DIntArraySubsettingAndBooleanIndexingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0,1:3,array <= 5"];
    Array<int_> result{4, 5};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DIntArraySlicingTest1) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    int_ c_array_result[1][1][3] = {{{4, 5, 6}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DIntArraySlicingTest2) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,1:2"];
    int_ c_array_result[1][1][1] = {{{5}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DIntArraySlicingAndBooleanIndexingTest) {
    // static
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:3,array <= 5"];
    Array<int_> result{4, 5};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest1) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[0];
    int_ c_array_2d[2][3] = {{1, 2, 3},
                             {4, 5, 6}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest2) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[0];
    auto subset2 = subset[1];
    auto subset3 = subset2[2];
    int_ c_array[1] = {6};
    Array<int_> result{c_array};
    compare(subset3, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArrayBooleanIndexingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto booleanIndex = array["array <= 2"];
    Array<int_> result{1, 2};
    compare(booleanIndex, result);

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2"];
    Array<int_> result2{2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingAndBooleanIndexingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto slice = array["0,1:3,array <= 5"];
    Array<int_> result{4, 5};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySlicingTest1) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto slice = array["0:1,1:2,"];
    int_ c_array_result[1][1][3] = {{{4, 5, 6}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySlicingTest2) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto slice = array["0:1,1:2,1:2"];
    int_ c_array_result[1][1][1] = {{{5}}};
    Array<int_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySlicingAndBooleanIndexingTest) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto slice = array["0:1,1:3,array <= 5"];
    Array<int_> result{4, 5};
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

    auto subset2 = subset[1];
    Array<float_> result2{4.4, 5.5, 6.6};
    compare(subset2, result2);
}

TEST_F(ArrayIndexTest, static3DFloatArraySubsettingAndBooleanIndexingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0,1:3,array <= 5.5"];
    Array<float_> result{4.4, 5.5};
    compare(slice, result);
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

    auto booleanIndex2 = booleanIndex["booleanIndex >= 2.2"];
    Array<float_> result2{2.2};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, static3DFloatArraySlicingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:2,2:"];
    float_ c_array_result[1][1][1] = {{{6.6}}};
    Array<float_> result{c_array_result};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DFloatArraySlicingAndBooleanIndexingTest) {
    // static
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto slice = array["0:1,1:3,array <= 5.5"];
    Array<float_> result{4.4, 5.5};
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

    auto subset2 = subset[1];
    Array<float_> result2{4.4, 5.5, 6.6};
    compare(subset2, result2);
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

    auto booleanIndex2 = booleanIndex["booleanIndex >= 3.3"];
    Array<float_> result2{3.3};
    compare(booleanIndex2, result2);
}

TEST_F(ArrayIndexTest, dynamic3DFloatArraySubsettingAndBooleanIndexingTest) {
    // dynamic
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto slice = array["0,1:3,array <= 5.5"];
    Array<float_> result{4.4, 5.5};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, dynamic3DFloatArraySlicingTest) {
    // dynamic
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto slice = array["0:1,1:2,2:"];
    float_ c_array_result[1][1][1] = {{{6.6}}};
    Array<float_> result{c_array_result};
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

    auto slice = subset["0:1,"];
    string_ c_array_result[1][3] = {{"str1", "str2", "str3"}};
    Array<string_> result2{c_array_result};
    compare(slice, result2);
}

TEST_F(ArrayIndexTest, static3DStringArraySubsettingAndBooleanIndexingTest) {
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
    auto slice = array["0,1:3,array <= str5"];
    Array<string_> result{"str4", "str5", "str13", "str14", "str15"};
    compare(slice, result);
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

    auto booleanIndex2 = booleanIndex["array <= str2"];
    Array<string_> result2{"str1", "str2", "str13", "str14", "str15", "str16", "str17", "str18", "str10", "str11", "str12", "str19"};
    compare(booleanIndex2, result2);

    auto booleanIndex3 = booleanIndex2["booleanIndex2 >= str2"];
    Array<string_> result3{"str2"};
    compare(booleanIndex3, result3);
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
    auto slice = array["0:1,1:2,2:"];
    string_ c_array_2d[1][1][1] = {{{"str6"}}};
    Array<string_> result{c_array_2d};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, static3DStringArraySlicingAndBooleanIndexingTest) {
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
    auto slice = array["0:1,1:3,array <= str5"];
    Array<string_> result{"str4", "str5", "str13", "str14", "str15"};
    compare(slice, result);
}

TEST_F(ArrayIndexTest, ravelMultiIndexesTest) {
    EXPECT_EQ(4, ravel_multi_index(Shape{0, 1, 0}, Shape{2, 3, 4}));
    EXPECT_EQ(7, ravel_multi_index(Shape{0, 1, 3}, Shape{2, 3, 4}));
    EXPECT_EQ(16, ravel_multi_index(Shape{1, 1, 0}, Shape{2, 3, 4}));
    EXPECT_EQ(19, ravel_multi_index(Shape{1, 1, 3}, Shape{2, 3, 4}));
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest0_) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array["0,:"];
    int_ c_array_1d[3] = {1, 2, 3};
    Array<int_> result{c_array_1d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest1_) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array["1,:"];
    int_ c_array_1d[3] = {4, 5, 6};
    Array<int_> result{c_array_1d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest_0) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array[":,0"];
    int_ c_array_1d[2] = {1, 4};
    Array<int_> result{c_array_1d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest_1) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array[":,1"];
    int_ c_array_1d[2] = {2, 5};
    Array<int_> result{c_array_1d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic2DIntArraySubsettingTest_2) {
    // dynamic
    int_ c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    auto subset = array[":,2"];
    int_ c_array_1d[2] = {3, 6};
    Array<int_> result{c_array_1d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest0__) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array["0,:,:"];
    int_ c_array_2d[2][3] = {{1, 2, 3},
                             {4, 5, 6}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest1__) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array["1,:,:"];
    int_ c_array_2d[2][3] = {{7, 8, 9},
                             {10, 11, 12}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest_0_) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[":,0,:"];
    int_ c_array_2d[2][3] = {{1, 2, 3},
                             {7, 8, 9}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest_1_) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[":,1,:"];
    int_ c_array_2d[2][3] = {{4, 5, 6},
                             {10, 11, 12}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest__0) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[":,:,0"];
    int_ c_array_2d[2][2] = {{1, 4},
                             {7, 10}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest__1) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[":,:,1"];
    int_ c_array_2d[2][2] = {{2, 5},
                             {8, 11}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic3DIntArraySubsettingTest__2) {
    // dynamic
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto subset = array[":,:,2"];
    int_ c_array_2d[2][2] = {{3, 6},
                             {9, 12}};
    Array<int_> result{c_array_2d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest0___) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array["0,:,:,:"];
    int_ c_array_3d[2][2][3] = {{{1, 2, 3},
                                 {4, 5, 6}},
                                {{7, 8, 9},
                                 {10, 11, 12}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest1___) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array["1,:,:,:"];
    int_ c_array_3d[2][2][3] = {{{13, 14, 15},
                                 {16, 17, 18}},
                                {{19, 20, 21},
                                 {22, 23, 24}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest_0__) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,0,:,:"];
    int_ c_array_3d[2][2][3] = {{{1, 2, 3},
                                 {4, 5, 6}},
                                {{13, 14, 15},
                                 {16, 17, 18}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest_1__) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,1,:,:"];
    int_ c_array_3d[2][2][3] = {{{7, 8, 9},
                                 {10, 11, 12}},
                                {{19, 20, 21},
                                 {22, 23, 24}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest__0_) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,:,0,:"];
    int_ c_array_3d[2][2][3] = {{{1, 2, 3},
                                 {7, 8, 9}},
                                {{13, 14, 15},
                                 {19, 20, 21}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest__1_) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,:,1,:"];
    int_ c_array_3d[2][2][3] = {{{4, 5, 6},
                                 {10, 11, 12}},
                                {{16, 17, 18},
                                 {22, 23, 24}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest___0) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,:,:,0"];
    int_ c_array_3d[2][2][2] = {{{1, 4},
                                 {7, 10}},
                                {{13, 16},
                                 {19, 22}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest___1) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,:,:,1"];
    int_ c_array_3d[2][2][2] = {{{2, 5},
                                 {8, 11}},
                                {{14, 17},
                                 {20, 23}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, dynamic4DIntArraySubsettingTest___2) {
    // dynamic
    int_ c_array_4d[2][2][2][3] = {{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}, {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}}};
    Array<int_> array{c_array_4d};
    auto subset = array[":,:,:,2"];
    int_ c_array_3d[2][2][2] = {{{3, 6},
                                 {9, 12}},
                                {{15, 18},
                                 {21, 24}}};
    Array<int_> result{c_array_3d};
    compare(subset, result);
}

TEST_F(ArrayIndexTest, extendedGCDTest) {
    using ndarray::internal::extendedGCD;
    {
        auto result = extendedGCD(1, 1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, 1);
    }
    {
        auto result = extendedGCD(-1, 1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, 1);
    }
    {
        auto result = extendedGCD(1, -1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, -1);
    }
    {
        auto result = extendedGCD(-1, -1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, -1);
    }
    {
        auto result = extendedGCD(2, 1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, 1);
    }
    {
        auto result = extendedGCD(-2, 1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, 1);
    }
    {
        auto result = extendedGCD(2, -1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, -1);
    }
    {
        auto result = extendedGCD(-2, -1);
        EXPECT_EQ(result.gcd, 1);
        EXPECT_EQ(result.bezout_coeff.first, 0);
        EXPECT_EQ(result.bezout_coeff.second, -1);
    }
}

TEST_F(ArrayIndexTest, submatrix2x2Test) {
    float_ c_array[2][2] = {{1.0, 2.0}, {0.0, 1.0}};
    Array<float_> array{c_array};
    float_ c_array_submatrix[1][1] = {{1.0}};
    Array<float_> array_submatrix{c_array_submatrix};
    auto result = array["0:1,0:1"];
    compare(array_submatrix, result);
}

TEST_F(ArrayIndexTest, submatrix3x3Test) {
    float_ c_array[3][3] = {{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 1.0}};
    Array<float_> array{c_array};
    float_ c_array_submatrix[2][2] = {{1.0, 2.0}, {0.0, 1.0}};
    Array<float_> array_submatrix{c_array_submatrix};
    auto result = array["0:2,0:2"];
    compare(array_submatrix, result);
}

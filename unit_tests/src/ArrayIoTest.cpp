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
#include <np/Io.hpp>
#include <np/Comp.hpp>

using namespace np;

class ArrayIoTest : public ::testing::Test {
protected:
    static void compareFiles(const std::string& file1, const std::string& file2) {
        std::ifstream input1(file1, std::ios::binary);
        ASSERT_TRUE(input1.is_open());

        input1.seekg(0, std::ios::end);
        std::streampos fileSize1 = input1.tellg();
        input1.seekg(0, std::ios::beg);

        std::ifstream input2(file2, std::ios::binary);
        ASSERT_TRUE(input2.is_open());

        input2.seekg(0, std::ios::end);
        std::streampos fileSize2 = input2.tellg();
        input2.seekg(0, std::ios::beg);

        ASSERT_EQ(fileSize1, fileSize2);

        std::vector<unsigned char> buffer1;
        buffer1.reserve(static_cast<std::size_t>(fileSize1));
        buffer1.insert(buffer1.begin(),
                   std::istream_iterator<unsigned char>(input1),
                   std::istream_iterator<unsigned char>());

        std::vector<unsigned char> buffer2;
        buffer2.reserve(static_cast<std::size_t>(fileSize2));
        buffer2.insert(buffer2.begin(),
                       std::istream_iterator<unsigned char>(input2),
                       std::istream_iterator<unsigned char>());

        ASSERT_EQ(buffer1, buffer2);
    }

    static void compareFileWithTestData(const std::string &filename) {
        std::string full_filename{filename};
        full_filename += ".npy";
        compareFiles(full_filename,"../test_data/" + full_filename);
    }
};

TEST_F(ArrayIoTest, dynamicEmptyIntArraySaveLoadTest) {
    /*
     >> np.array([])
     */
    // dynamic
    Array<intc> array{};
    const char* filename = "empty_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<intc>(filename);
    ASSERT_TRUE(array_equal<intc>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamicEmptyFloatArraySaveLoadTest) {
    Array<float_> array{};
    const char* filename = "empty_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal<float_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamicEmptyStringArraySaveLoadTest) {
    Array<string_> array{};
    const char* filename = "empty_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal<string_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamicEmptyUnicodeArraySaveLoadTest) {
    Array<unicode_> array{};
    const char* filename = "empty_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal<unicode_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static1DIntArraySaveLoadTest) {
    /*
    >>> np.array([1,2,3,4])
     */
    // static
    Array<int_, 4> array{1, 2, 3, 4};
    const char* filename = "1D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic1DIntArraySaveLoadTest) {
    /*
    >>> np.array([1,2,3,4])
     */
    // dynamic
    Array<int_> array{1, 2, 3, 4};
    const char* filename = "1D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal<int_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static1DFloatArraySaveLoadTest) {
    /*
    >>> np.array([1.1, 2.2, 3.3, 4.4])
     */
    Array<float_, 4> array{1.1, 2.2, 3.3, 4.4};
    const char* filename = "1D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic1DFloatArraySaveLoadTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    const char* filename = "1D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal<float_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static1DStringArraySaveLoadTest) {
    Array<string_, 4> array{"str1", "str2", "str3", "str4"};
    static const char* filename = "1D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic1DStringArraySaveLoadTest) {
    Array<string_> array{"str1", "str2", "str3", "str4"};
    static const char* filename = "1D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal<string_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static1DUnicodeArraySaveLoadTest) {
    Array<unicode_, 4> array{L"str1", L"str2", L"str3", L"str4"};
    static const char* filename = "1D_string";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic1DUnicodeArraySaveLoadTest) {
    Array<unicode_> array{L"str1", L"str2", L"str3", L"str4"};
    static const char* filename = "1D_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal<unicode_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static2DIntArraySaveLoadTest) {
    /*
    >>> np.array([[1,2,3,4],[5,6,7,8]])
     */
    // static
    long arr[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    Array<int_, 2, 4> array{arr};
    static const char* filename = "2D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic2DIntArraySaveLoadTest) {
    /*
    >>> np.array([[1,2,3,4],[5,6,7,8]])
     */
    // dynamic
    long arr[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
    Array<int_> array{arr};
    static const char* filename = "2D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal<int_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static2DFloatArraySaveLoadTest) {
    /*
    >>> np.array([[1.1,2.2,3.3,4.4],[5.5,6.6,7.7,8.8]])
     */
    double arr[2][4] = {{1.1, 2.2, 3.3, 4.4}, {5.5, 6.6, 7.7, 8.8}};
    Array<float_, 2, 4> array{arr};
    static const char* filename = "2D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic2DFloatArraySaveLoadTest) {
    double arr[2][4] = {{1.1, 2.2, 3.3, 4.4}, {5.5, 6.6, 7.7, 8.8}};
    Array<float_> array{arr};
    static const char* filename = "2D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal<float_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static2DStringArraySaveLoadTest) {
    string_ arr[2][4] = {{"str1", "str2", "str3", "str4"},
                             {"str5", "str6", "str7", "str8"}};
    Array<string_, 2, 4> array{arr};
    static const char* filename = "2D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic2DStringArraySaveLoadTest) {
    string_ arr[2][4] = {{"str1", "str2", "str3", "str4"},
                             {"str5", "str6", "str7", "str8"}};
    Array<string_> array{arr};
    static const char* filename = "2D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal<string_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static2DUnicodeArraySaveLoadTest) {
    unicode_ arr[2][4] = {{L"str1", L"str2", L"str3", L"str4"},
                              {L"str5", L"str6", L"str7", L"str8"}};
    Array<unicode_, 2, 4> array{arr};
    static const char* filename = "2D_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic2DUnicodeArraySaveLoadTest) {
    unicode_ arr[2][4] = {{L"str1", L"str2", L"str3", L"str4"},
                              {L"str5", L"str6", L"str7", L"str8"}};
    Array<unicode_> array{arr};
    static const char* filename = "2D_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal<unicode_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static3DIntArraySaveLoadTest) {
    /*
    >>> np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
     */
    // static
    long arr[2][4][3] = {
        {
        {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11,12}
        },
        {
            {13, 14, 15},
            {16, 17, 18},
            {19, 20, 21},
            {22, 23, 24}
        }
    };
    Array<int_, 2, 4, 3> array{arr};
    static const char* filename = "3D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic3DIntArraySaveLoadTest) {
    /*
    >>> np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
     */
    // dynamic
    long arr[2][4][3] = {
    {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {10, 11,12}
        },
    {
        {13, 14, 15},
        {16, 17, 18},
        {19, 20, 21},
        {22, 23, 24}
        }
    };
    Array<int_> array{arr};
    static const char* filename = "3D_int";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<int_>(filename);
    ASSERT_TRUE(array_equal<int_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static3DFloatArraySaveLoadTest) {
    /*
    >>> np.array([[[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9], [10.1, 11.11, 12.12]],
        [[13.13, 14.14, 15.15], [16.16, 17.17, 18.18], [19.19, 20.2, 21.21], [22.22, 23.23, 24.24]]])
     */
    double arr[2][4][3] = {
    {
        {1.1, 2.2, 3.3},
        {4.4, 5.5, 6.6},
        {7.7, 8.8, 9.9},
        {10.1, 11.11,12.12}
        },
        {
        {13.13, 14.14, 15.15},
        {16.16, 17.17, 18.18},
        {19.19, 20.2, 21.21},
        {22.22, 23.23, 24.24}
        }
    };
    Array<float_, 2, 4, 3> array{arr};
    static const char* filename = "3D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic3DFloatArraySaveLoadTest) {
    double arr[2][4][3] = {
        {
            {1.1, 2.2, 3.3},
            {4.4, 5.5, 6.6},
            {7.7, 8.8, 9.9},
            {10.1, 11.11,12.12}
        },
        {
            {13.13, 14.14, 15.15},
            {16.16, 17.17, 18.18},
            {19.19, 20.2, 21.21},
            {22.22, 23.23, 24.24}
        }
    };
    Array<float_> array{arr};
    static const char* filename = "3D_float";
    array.save(filename);
    compareFileWithTestData(filename);
    auto arrayLoaded = load<float_>(filename);
    ASSERT_TRUE(array_equal<float_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static3DStringArraySaveLoadTest) {
    string_ arr[2][4][3] = {
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
    Array<string_, 2, 4, 3> array{arr};
    static const char* filename = "3D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic3DStringArraySaveLoadTest) {
    string_ arr[2][4][3] = {
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
    Array<string_> array{arr};
    static const char* filename = "3D_string";
    array.save(filename);
    auto arrayLoaded = load<string_>(filename);
    ASSERT_TRUE(array_equal<string_>(array, arrayLoaded));
}

TEST_F(ArrayIoTest, static3DUnicodeArraySaveLoadTest) {
    unicode_ arr[2][4][3] = {{
        {L"str1_1", L"str1_2", L"str1_3"},
        {L"str2_1", L"str2_2", L"str2_3"},
        {L"str3_1", L"str3_2", L"str3_3"},
        {L"str4_1", L"str4_2", L"str4_3"}
        },{
        { L"str5_1", L"str5_2", L"str5_3" },
        { L"str6_1", L"str6_2", L"str6_3" },
        { L"str7_1", L"str7_2", L"str7_3" },
        { L"str8_1", L"str8_2", L"str8_3" }
        }
    };
    Array<unicode_, 2, 4, 3> array{arr};
    static const char* filename = "3D_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal(array, arrayLoaded));
}

TEST_F(ArrayIoTest, dynamic3DUnicodeArraySaveLoadTest) {
    unicode_ arr[2][4][3] = {{
     {L"str1_1", L"str1_2", L"str1_3"},
     {L"str2_1", L"str2_2", L"str2_3"},
     {L"str3_1", L"str3_2", L"str3_3"},
     {L"str4_1", L"str4_2", L"str4_3"}
     },{
     { L"str5_1", L"str5_2", L"str5_3" },
     { L"str6_1", L"str6_2", L"str6_3" },
     { L"str7_1", L"str7_2", L"str7_3" },
     { L"str8_1", L"str8_2", L"str8_3" }
        }
    };
    Array<unicode_> array{arr};
    static const char* filename = "3D_unicode";
    array.save(filename);
    auto arrayLoaded = load<unicode_>(filename);
    ASSERT_TRUE(array_equal<unicode_>(array, arrayLoaded));
}
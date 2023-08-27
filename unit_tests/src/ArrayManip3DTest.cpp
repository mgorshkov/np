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

class ArrayManip3DTest : public ArrayTest {
protected:
};

TEST_F(ArrayManip3DTest, static3DIntArrayTransposeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {
            {{1, 7},
             {4, 10}},
            {{2, 8},
             {5, 11}},
            {{3, 9},
             {6, 12}}};
    Array<int_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayRavelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayReshapeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<int_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayResizeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayAppendTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayInsertTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_1d[24] = {1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayDelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_1d[11] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_> array1{c_array_1d};
    auto result = array.del(1);
    compare(result, array1);
}

TEST_F(ArrayManip3DTest, static3DIntArrayConcatenateTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = array.concatenate(array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayVStackTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayR_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_3d_result[4][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                       {{7, 8, 9}, {10, 11, 12}},
                                       {{13, 14, 15}, {16, 17, 18}},
                                       {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayHStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayColumnStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayC_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_, 2 * 2 * 3> array1{c_array_3d1};
    int_ c_array_3d_result[2][2][6] = {{{1, 2, 3, 13, 14, 15},
                                        {4, 5, 6, 16, 17, 18}},
                                       {{7, 8, 9, 19, 20, 21},
                                        {10, 11, 12, 22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DIntArrayHSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_result_sample_0[2][1][3] = {{{1, 2, 3}}, {{7, 8, 9}}};
    Array<int_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    int_ c_array_result_sample_1[2][1][3] = {{{4, 5, 6}}, {{10, 11, 12}}};
    Array<int_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DIntArrayVSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ c_array_3d_0[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
    Array<int_> result_sample_0{c_array_3d_0};
    int_ c_array_3d_1[1][2][3] = {{{7, 8, 9}, {10, 11, 12}}};
    Array<int_> result_sample_1{c_array_3d_1};

    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DIntArrayExpandDimsTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto result = expand_dims(array, 2);
    int_ result_c_array_4d[2][2][1][3] = {{{{1, 2, 3}}, {{4, 5, 6}}},
                                          {{{7, 8, 9}}, {{10, 11, 12}}}};
    Array<int_> result_sample{result_c_array_4d};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayTransposeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 7.7},
             {4.4, 10.10}},
            {{2.2, 8.8},
             {5.5, 11.11}},
            {{3.3, 9.9},
             {6.6, 12.12}}};
    Array<float_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayRavelTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayReshapeTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<float_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayResizeTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayAppendTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
                                16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayInsertTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_1d[24] = {1.1, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    Array<float_> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayDelTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_1d[11] = {1.1, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    Array<float_> result_sample{c_array_1d};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayConcatenateTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayVStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayR_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[4][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}},
                                      {{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayHStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}, {19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayColumnStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}, {19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayC_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_, 2 * 2 * 3> array1{c_array_3d1};
    float_ c_array_result[2][2][6] = {{{1.1, 2.2, 3.3, 13.13, 14.14, 15.15}, {4.4, 5.5, 6.6, 16.16, 17.17, 18.18}}, {{7.7, 8.8, 9.9, 19.19, 20.2, 21.21}, {10.10, 11.11, 12.12, 22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayHSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_result_sample_0[2][1][3] = {{{1.1, 2.2, 3.3}}, {{7.7, 8.8, 9.9}}};
    Array<float_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    float_ c_array_result_sample_1[2][1][3] = {{{4.4, 5.5, 6.6}}, {{10.10, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayVSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ c_array_3d_0[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample_0{c_array_3d_0};
    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    float_ c_array_3d_1[1][2][3] = {{{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_3d_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DFloatArrayExpandDimsTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    auto result = expand_dims(array, 2);
    float_ c_array_4d_result[2][2][1][3] = {{{{1.1, 2.2, 3.3}}, {{4.4, 5.5, 6.6}}},
                                            {{{7.7, 8.8, 9.9}}, {{10.1, 11.11, 12.12}}}};

    Array<float_> result_sample{c_array_4d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayTransposeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str7"},
             {"str4", "str10"}},
            {{"str2", "str8"},
             {"str5", "str11"}},
            {{"str3", "str9"},
             {"str6", "str12"}}};
    Array<string_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayRavelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12"};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayReshapeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<string_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayResizeTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_result_3d[3][2][2] = {
            {{"str1", "str2"},
             {"str3", "str4"}},
            {{"str5", "str6"},
             {"str7", "str8"}},
            {{"str9", "str10"},
             {"str11", "str12"}}};
    Array<string_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayAppendTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<string_> result_sample{"str1", "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12",
                                 "str13", "str14", "str15",
                                 "str16", "str17", "str18",
                                 "str19", "str20", "str21",
                                 "str22", "str23", "str24"};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayInsertTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    Array<string_> result_sample{"str1",
                                 "str13", "str14", "str15",
                                 "str16", "str17", "str18",
                                 "str19", "str20", "str21",
                                 "str22", "str23", "str24",
                                 "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12"};
    auto result = array.insert(1, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayDelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_1d[11] = {"str1", "str3",
                              "str4", "str5", "str6",
                              "str7", "str8", "str9",
                              "str10", "str11", "str12"};
    Array<string_> result_sample{c_array_1d};
    auto result = array.del(1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayConcatenateTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = array.concatenate(array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayVStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayR_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[4][2][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}},
                                       {{"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayHStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"}, {"str4", "str5", "str6"}, {"str13", "str14", "str15"}, {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"}, {"str10", "str11", "str12"}, {"str19", "str20", "str21"}, {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayColumnStackTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[2][4][3] = {{{"str1", "str2", "str3"},
                                        {"str4", "str5", "str6"},
                                        {"str13", "str14", "str15"},
                                        {"str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9"},
                                        {"str10", "str11", "str12"},
                                        {"str19", "str20", "str21"},
                                        {"str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayC_Test) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d1[2][2][3] = {
            {{"str13", "str14", "str15"},
             {"str16", "str17", "str18"}},
            {{"str19", "str20", "str21"},
             {"str22", "str23", "str24"}}};
    Array<string_, 2 * 2 * 3> array1{c_array_3d1};
    string_ c_array_result[2][2][6] = {{{"str1", "str2", "str3", "str13", "str14", "str15"},
                                        {"str4", "str5", "str6", "str16", "str17", "str18"}},
                                       {{"str7", "str8", "str9", "str19", "str20", "str21"},
                                        {"str10", "str11", "str12", "str22", "str23", "str24"}}};
    Array<string_> result_sample{c_array_result};
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, static3DStringArrayHSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_result_sample_0[2][1][3] = {{{"str1", "str2", "str3"}}, {{"str7", "str8", "str9"}}};
    Array<string_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    string_ c_array_result_sample_1[2][1][3] = {{{"str4", "str5", "str6"}}, {{"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DStringArrayVSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    string_ c_array_3d_0[1][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}}};
    Array<string_> result_sample_0{c_array_3d_0};
    string_ c_array_3d_1[1][2][3] = {
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_3d_1};

    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, static3DStringArrayExpandDimsTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_, 2 * 2 * 3> array{c_array_3d};
    auto result = expand_dims(array, 2);
    string_ c_array_4d_result[2][2][1][3] = {
            {{{"str1", "str2", "str3"}},
             {{"str4", "str5", "str6"}}},
            {{{"str7", "str8", "str9"}},
             {{"str10", "str11", "str12"}}}};
    Array<string_> result_sample{c_array_4d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayTransposeTest) {
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
    Array<int_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayRavelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    auto result = array.ravel();
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayReshapeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<int_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayResizeTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_3d[3][2][2] = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{9, 10}, {11, 12}}};
    Array<int_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayAppendTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<int_> result_sample{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayInsertTest) {
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
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayDelTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_1d[11] = {1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Array<int_> array1{c_array_1d};
    auto result = array.del(1);
    compare(result, array1);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayConcatenateTest) {
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
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayVStackTest) {
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
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayR_Test) {
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
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayHStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}},
                                 {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayColumnStackTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_result[2][4][3] = {{{1, 2, 3}, {4, 5, 6}, {13, 14, 15}, {16, 17, 18}},
                                    {{7, 8, 9}, {10, 11, 12}, {19, 20, 21}, {22, 23, 24}}};
    Array<int_> result_sample{c_array_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayC_Test) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    long c_array_3d1[2][2][3] = {{{13, 14, 15}, {16, 17, 18}}, {{19, 20, 21}, {22, 23, 24}}};
    Array<int_> array1{c_array_3d1};
    int_ c_array_3d_result[2][2][6] = {{{1, 2, 3, 13, 14, 15}, {4, 5, 6, 16, 17, 18}}, {{7, 8, 9, 19, 20, 21}, {10, 11, 12, 22, 23, 24}}};
    Array<int_> result_sample{c_array_3d_result};
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayHSplitTest) {
    /*
    a=np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype='int')
     */
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_> array{c_array_3d};
    int_ c_array_result_sample_0[2][1][3] = {{{1, 2, 3}}, {{7, 8, 9}}};
    Array<int_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    int_ c_array_result_sample_1[2][1][3] = {{{4, 5, 6}}, {{10, 11, 12}}};
    Array<int_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayVSplitTest) {
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

    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DIntArrayExpandDimsTest) {
    int_ c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}},
                                {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    auto result = expand_dims(array, 2);
    int_ c_array_4d_result[2][2][1][3] = {
            {{{1, 2, 3}}, {{4, 5, 6}}},
            {{{7, 8, 9}}, {{10, 11, 12}}}};
    Array<int_> result_sample{c_array_4d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayTransposeTest) {
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
    Array<float_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayRavelTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto result = array.ravel();
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayReshapeTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<float_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayResizeTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_3d[3][2][2] = {
            {{1.1, 2.2},
             {3.3, 4.4}},
            {{5.5, 6.6},
             {7.7, 8.8}},
            {{9.9, 10.10},
             {11.11, 12.12}}};
    Array<float_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayAppendTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    auto result = array.append(array1);
    Array<float_> result_sample{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12, 13.13, 14.14, 15.15,
                                16.16, 17.17, 18.18, 19.19, 20.2, 21.21, 22.22, 23.23, 24.24};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayInsertTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_1d[24] = {1.1, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.11, 12.12};
    Array<float_> result_sample{c_array_1d};
    auto result = array.insert(1, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayDelTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_1d[11] = {1.1, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12};
    Array<float_> array1{c_array_1d};
    auto result = array.del(1);
    compare(result, array1);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayConcatenateTest) {
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
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayVStackTest) {
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
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayR_Test) {
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
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayHStackTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}, {19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayColumnStackTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][4][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}},
                                      {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}, {19.19, 20.2, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayC_Test) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d1[2][2][3] = {{{13.13, 14.14, 15.15}, {16.16, 17.17, 18.18}}, {{19.19, 20.20, 21.21}, {22.22, 23.23, 24.24}}};
    Array<float_> array1{c_array_3d1};
    float_ c_array_result[2][2][6] = {{{1.1, 2.2, 3.3, 13.13, 14.14, 15.15}, {4.4, 5.5, 6.6, 16.16, 17.17, 18.18}}, {{7.7, 8.8, 9.9, 19.19, 20.2, 21.21}, {10.10, 11.11, 12.12, 22.22, 23.23, 24.24}}};
    Array<float_> result_sample{c_array_result};
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayHSplitTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_result_sample_0[2][1][3] = {{{1.1, 2.2, 3.3}}, {{7.7, 8.8, 9.9}}};
    Array<float_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    float_ c_array_result_sample_1[2][1][3] = {{{4.4, 5.5, 6.6}}, {{10.1, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayVSplitTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    float_ c_array_3d_0[1][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}}};
    Array<float_> result_sample_0{c_array_3d_0};
    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    float_ c_array_3d_1[1][2][3] = {{{7.7, 8.8, 9.9}, {10.10, 11.11, 12.12}}};
    Array<float_> result_sample_1{c_array_3d_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DFloatArrayExpandDimsTest) {
    float_ c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_> array{c_array_3d};
    auto result = expand_dims(array, 0);
    float_ c_array_3d_result[1][2][2][3] = {{{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                             {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}}};
    Array<float_> result_sample{c_array_3d_result};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayTransposeTest) {
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
    Array<string_> result_sample{c_array_result_3d};
    auto result = transpose(array);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayRavelTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    auto result = array.ravel();
    Array<string_> result_sample{"str1", "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12"};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayReshapeTest) {
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
    Array<string_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    Array<string_> result = array.reshape(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayResizeTest) {
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
    Array<string_> result_sample{c_array_result_3d};
    Shape shape{3, 2, 2};
    auto result = array.resize(shape);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayAppendTest) {
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
    Array<string_> result_sample{"str1", "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12",
                                 "str13", "str14", "str15",
                                 "str16", "str17", "str18",
                                 "str19", "str20", "str21",
                                 "str22", "str23", "str24"};
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayInsertTest) {
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
    Array<string_> result_sample{"str1",
                                 "str13", "str14", "str15",
                                 "str16", "str17", "str18",
                                 "str19", "str20", "str21",
                                 "str22", "str23", "str24",
                                 "str2", "str3",
                                 "str4", "str5", "str6",
                                 "str7", "str8", "str9",
                                 "str10", "str11", "str12"};
    auto result = array.insert(1, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayDelTest) {
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
    compare(result, array1);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayConcatenateTest) {
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
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayVStackTest) {
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
    auto result = vstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayR_Test) {
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
    auto result = r_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayHStackTest) {
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
    auto result = hstack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayColumnStackTest) {
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
    auto result = column_stack(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayC_Test) {
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
    auto result = c_(array, array1);
    compare(result, result_sample);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayHSplitTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    string_ c_array_result_sample_0[2][1][3] = {{{"str1", "str2", "str3"}}, {{"str7", "str8", "str9"}}};
    Array<string_> result_sample_0{c_array_result_sample_0};
    auto result = hsplit(array, 2);
    compare(result[0], result_sample_0);
    string_ c_array_result_sample_1[2][1][3] = {{{"str4", "str5", "str6"}}, {{"str10", "str11", "str12"}}};
    Array<string_> result_sample_1{c_array_result_sample_1};
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayVSplitTest) {
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

    auto result = vsplit(array, 2);
    compare(result[0], result_sample_0);
    compare(result[1], result_sample_1);
}

TEST_F(ArrayManip3DTest, dynamic3DStringArrayExpandDimsTest) {
    string_ c_array_3d[2][2][3] = {
            {{"str1", "str2", "str3"},
             {"str4", "str5", "str6"}},
            {{"str7", "str8", "str9"},
             {"str10", "str11", "str12"}}};
    Array<string_> array{c_array_3d};
    auto result = expand_dims(array, 2);
    string_ c_array_4d_result[2][2][1][3] = {
            {{{"str1", "str2", "str3"}},
             {{"str4", "str5", "str6"}}},
            {{{"str7", "str8", "str9"}},
             {{"str10", "str11", "str12"}}}};
    Array<string_> result_sample{c_array_4d_result};
    compare(result, result_sample);
}
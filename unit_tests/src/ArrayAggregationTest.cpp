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

#include <np/Agg.hpp>
#include <np/Array.hpp>
#include <np/Comp.hpp>

using namespace np;

using np::ndarray::array_dynamic::NDArrayDynamic;
using np::ndarray::array_static::NDArrayStatic;

class ArrayAggregationTest : public ::testing::Test {
protected:
    // static arrays
    template<typename DType, Size SizeCumSum, Size SizeT, Size... SizeTs>
    inline void checkArrayAggregation(const NDArrayStatic<DType, SizeT, SizeTs...> &array,
                                      DType sum_, DType min_, DType max_, NDArrayStatic<DType, SizeCumSum> cumsum_, DType mean_, DType median_, const NDArrayDynamic<DType> &corrcoef_, DType std__, bool corrException = false) {
        auto sum__ = sum<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(sum_, sum__);
        auto min__ = min<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(min_, min__);
        auto max__ = max<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(max_, max__);
        auto cumsum__ = cumsum<DType, SizeT, SizeTs...>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        EXPECT_TRUE(equal);
        auto mean__ = mean<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(mean_, mean__);
        auto median__ = median<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(median_, median__);
        try {
            auto corrcoef__ = corrcoef<DType, SizeT, SizeTs...>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<DType, SizeT, SizeTs...>(array);
        EXPECT_EQ(std__, std___);
    }

    template<Size SizeCumSum, Size SizeT, Size... SizeTs>
    inline void checkArrayAggregation(const NDArrayStatic<float_, SizeT, SizeTs...> &array,
                                      float_ sum_, float_ min_, float_ max_, NDArrayStatic<float_, SizeCumSum> cumsum_, float_ mean_, float_ median_,
                                      const NDArrayDynamic<float_> &corrcoef_, float_ std__, bool corrException = false) {
        auto sum__ = sum<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(sum_, sum__);
        auto min__ = min<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(min_, min__);
        auto max__ = max<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(max_, max__);
        auto cumsum__ = cumsum<float_, SizeT, SizeTs...>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        EXPECT_TRUE(equal);
        auto mean__ = mean<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(mean_, mean__);
        auto median__ = median<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(median_, median__);
        try {
            auto corrcoef__ = corrcoef<float_, SizeT, SizeTs...>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<float_, SizeT, SizeTs...>(array);
        EXPECT_DOUBLE_EQ(std__, std___);
    }

    // dynamic arrays
    template<typename DType>
    inline void checkArrayAggregation(const NDArrayDynamic<DType> &array,
                                      DType sum_, DType min_, DType max_, NDArrayDynamic<DType> cumsum_, DType mean_, DType median_, const NDArrayDynamic<DType> &corrcoef_, DType std__, bool corrException = false) {
        auto sum__ = sum<DType>(array);
        EXPECT_EQ(sum_, sum__);
        auto min__ = min<DType>(array);
        EXPECT_EQ(min_, min__);
        auto max__ = max<DType>(array);
        EXPECT_EQ(max_, max__);
        auto cumsum__ = cumsum<DType>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        EXPECT_TRUE(equal);
        auto mean__ = mean<DType>(array);
        EXPECT_EQ(mean_, mean__);
        auto median__ = median<DType>(array);
        EXPECT_EQ(median_, median__);
        try {
            auto corrcoef__ = corrcoef<DType>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<DType>(array);
        EXPECT_EQ(std__, std___);
    }

    inline void checkArrayAggregation(const NDArrayDynamic<float_> &array,
                                      float_ sum_, float_ min_, float_ max_, NDArrayDynamic<float_> cumsum_,
                                      float_ mean_, float_ median_, const NDArrayDynamic<float_> &corrcoef_, float_ std__, bool corrException) {
        auto sum__ = sum<float_>(array);
        EXPECT_DOUBLE_EQ(sum_, sum__);
        auto min__ = min<float_>(array);
        EXPECT_DOUBLE_EQ(min_, min__);
        auto max__ = max<float_>(array);
        EXPECT_DOUBLE_EQ(max_, max__);
        auto cumsum__ = cumsum<float_>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        EXPECT_TRUE(equal);
        auto mean__ = mean<float_>(array);
        EXPECT_DOUBLE_EQ(mean_, mean__);
        auto median__ = median<float_>(array);
        EXPECT_DOUBLE_EQ(median_, median__);
        try {
            auto corrcoef__ = corrcoef<float_>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<float_>(array);
        EXPECT_DOUBLE_EQ(std__, std___);
    }
};

TEST_F(ArrayAggregationTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    int_ sum = 0;
    int_ min = 0;
    int_ max = 0;
    Array<int_> cumsum{};
    int_ mean = 0;
    int_ median = 0;
    Array<int_> corrcoef{};
    int_ std_ = 0;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    float_ sum = 0.0;
    float_ min = 0.0;
    float_ max = 0.0;
    Array<float_> cumsum{};
    float_ mean = 0.0;
    float_ median = 0.0;
    Array<float_> corrcoef{};
    float_ std_ = 0.0;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static1DIntArrayTest) {
    // static
    Array<int_, 5> array{17, 28, 3, 46, 72};
    int_ sum = 166;
    int_ min = 3;
    int_ max = 72;
    Array<int_, 5> cumsum{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    int_ mean = 33;
    int_ median = 28;
    Array<int_> corrcoef{1};
    int_ std_ = 23;
    checkArrayAggregation<int_, 5>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static1DFloatArrayTest) {
    Array<float_, 5> array{17.1, 42.2, 83.3, 24.4, 16.6};
    float_ sum = 183.6;
    float_ min = 16.6;
    float_ max = 83.3;
    Array<float_, 5> cumsum{17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    float_ mean = 36.72;
    float_ median = 24.4;
    Array<float_> corrcoef{1.0};
    float_ std_ = 25.064987532412616;
    checkArrayAggregation<5>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array{17, 28, 3, 46, 72};
    int_ sum = 166;
    int_ min = 3;
    int_ max = 72;
    Array<int_> cumsum{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    int_ mean = 33;
    int_ median = 28;
    Array<int_> corrcoef{1};
    int_ std_ = 23;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic1DFloatArrayTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    float_ sum = 11;
    float_ min = 1.1;
    float_ max = 4.4;
    Array<float_> cumsum{1.1, 3.3, 6.6, 11};
    float_ mean = 2.75;
    float_ median = 2.75;
    Array<float_> corrcoef{1};
    float_ std_ = 1.2298373876248845;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2, 3> array{c_array_2d};
    int_ sum = 21;
    int_ min = 1;
    int_ max = 6;
    Array<int_, 6> cumsum{1, 3, 6, 10, 15, 21};
    int_ mean = 3;
    int_ median = 3;
    long c_array_corrcoef[2][2] = {{0, 0}, {0, 0}};
    Array<int_> corrcoef{c_array_corrcoef};
    int_ std_ = 1;
    checkArrayAggregation<int_, 6, 2, 3>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2, 3> array{c_array_2d};
    float_ sum = 23.1;
    float_ min = 1.1;
    float_ max = 6.6;
    Array<float_, 6> cumsum{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    float_ mean = 3.85;
    float_ median = 3.85;
    float_ c_array2d_corrcoef[2][2] = {{0, 0}, {0, 0}};
    Array<float_> corrcoef{c_array2d_corrcoef};
    float_ std_ = 1.8786076404259262;
    checkArrayAggregation<6, 2, 3>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ sum = 21;
    int_ min = 1;
    int_ max = 6;
    Array<int_> cumsum{1, 3, 6, 10, 15, 21};
    int_ mean = 3;
    int_ median = 3;
    long c_array_2d_corrcoef[2][2] = {{0, 0}, {0, 0}};
    Array<int_> corrcoef{c_array_2d_corrcoef};
    int_ std_ = 1;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    float_ sum = 23.1;
    float_ min = 1.1;
    float_ max = 6.6;
    Array<float_> cumsum{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    float_ mean = 3.85;
    float_ median = 3.85;
    double c_array_2d_corrcoef[2][2] = {{0, 0}, {0, 0}};
    Array<float_> corrcoef{c_array_2d_corrcoef};
    float_ std_ = 1.8786076404259262;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2, 2, 3> array{c_array_3d};
    int_ sum = 78;
    int_ min = 1;
    int_ max = 12;

    Array<int_, 12> cumsum{1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78};
    int_ mean = 6;
    int_ median = 6;
    Array<int_> corrcoef{c_array_3d};
    int_ std_ = 3;
    checkArrayAggregation<int_, 12, 2, 2, 3>(array, sum, min, max, cumsum, mean, median, corrcoef, std_, true);
}

TEST_F(ArrayAggregationTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2, 2, 3> array{c_array_3d};
    float_ sum = 82.83;
    float_ min = 1.1;
    float_ max = 12.12;
    Array<float_, 12> cumsum{1.1, 3.3, 6.6, 11, 16.5, 23.1, 30.8, 39.6, 49.5, 59.6, 70.71, 82.83};
    float_ mean = 6.9025;
    float_ median = 7.15;
    Array<float_> corrcoef{c_array_3d};
    float_ std_ = 3.4815277417631854;
    checkArrayAggregation<12, 2, 2, 3>(array, sum, min, max, cumsum, mean, median, corrcoef, std_, true);
}

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

using np::ndarray::array_dynamic::NDArrayDynamic;
using np::ndarray::array_static::NDArrayStatic;

class ArrayAggregationTest : public ArrayTest {
protected:
    template<typename DType>
    struct AggNaNResults {
        DType nansum{0};
        Array<DType> nancumsum{};
        float_ nanmean{0.0};
        float_ nanmedian{0.0};
        float_ nanstd{0.0};
        float_ nanvar{0.0};
    };

    template<typename DType>
    struct AggResults : AggNaNResults<DType> {
        DType sum{0};
        DType min{0};
        DType max{0};
        Array<DType> cumsum{};
        float_ mean{0.0};
        float_ median{0.0};
        Array<float_> cov{};
        Array<float_> corrcoef{};
        float_ std_{0.0};
        float_ var{0.0};
        bool covException{false};
        bool corrException{false};
    };

    // static arrays
    template<typename DType, Size SizeT>
    static void checkArrayAggregation(const NDArrayStatic<DType, SizeT> &array,
                                      const AggResults<DType> &aggResults) {
        auto sum_ = sum<DType, SizeT>(array);
        compareValue(aggResults.sum, sum_);
        auto nansum_ = nansum<DType, SizeT>(array);
        compareValue(aggResults.nansum, nansum_);
        auto min_ = min<DType, SizeT>(array);
        compareValue(aggResults.min, min_);
        auto max_ = max<DType, SizeT>(array);
        compareValue(aggResults.max, max_);
        auto cumsum_ = cumsum<DType, SizeT>(array);
        compare(aggResults.cumsum, cumsum_);
        auto nancumsum_ = nancumsum<DType, SizeT>(array);
        compare(aggResults.nancumsum, nancumsum_);
        auto mean_ = mean<DType, SizeT>(array);
        compareValue(aggResults.mean, mean_);
        auto nanmean_ = nanmean<DType, SizeT>(array);
        compareValue(aggResults.nanmean, nanmean_);
        auto median_ = median<DType, SizeT>(array);
        compareValue(aggResults.median, median_);
        auto nanmedian_ = nanmedian<DType, SizeT>(array);
        compareValue(aggResults.nanmedian, nanmedian_);
        try {
            auto cov_ = cov<DType, SizeT>(array);
            compare(aggResults.cov, cov_);
            EXPECT_FALSE(aggResults.covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(aggResults.covException);
        }
        try {
            auto corrcoef_ = corrcoef<DType, SizeT>(array);
            compare(aggResults.corrcoef, corrcoef_);
            EXPECT_FALSE(aggResults.corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(aggResults.corrException);
        }
        auto std__ = std_<DType, SizeT>(array);
        compareValue(aggResults.std_, std__);
        auto nanstd_ = nanstd<DType, SizeT>(array);
        compareValue(aggResults.nanstd, nanstd_);
        auto var_ = var<DType, SizeT>(array);
        compareValue(aggResults.var, var_);
        auto nanvar_ = nanvar<DType, SizeT>(array);
        compareValue(aggResults.nanvar, nanvar_);
    }

    template<typename DType, Size SizeT>
    static void checkArrayAggregation(const NDArrayStatic<DType, SizeT> &array,
                                      const AggNaNResults<DType> &aggResults) {
        auto nansum_ = nansum<DType, SizeT>(array);
        compareValue(aggResults.nansum, nansum_);
        auto nancumsum_ = nancumsum<DType, SizeT>(array);
        compare(aggResults.nancumsum, nancumsum_);
        auto nanmean_ = nanmean<DType, SizeT>(array);
        compareValue(aggResults.nanmean, nanmean_);
        auto nanmedian_ = nanmedian<DType, SizeT>(array);
        compareValue(aggResults.nanmedian, nanmedian_);
        auto nanstd_ = nanstd<DType, SizeT>(array);
        compareValue(aggResults.nanstd, nanstd_);
        auto nanvar_ = nanvar<DType, SizeT>(array);
        compareValue(aggResults.nanvar, nanvar_);
    }

    // dynamic arrays
    template<typename DType>
    static void checkArrayAggregation(const NDArrayDynamic<DType> &array,
                                      const AggResults<DType> &aggResults) {
        auto sum_ = sum<DType>(array);
        compareValue(aggResults.sum, sum_);
        auto nansum_ = nansum<DType>(array);
        compareValue(aggResults.nansum, nansum_);
        auto min_ = min<DType>(array);
        compareValue(aggResults.min, min_);
        auto max_ = max<DType>(array);
        compareValue(aggResults.max, max_);
        auto cumsum_ = cumsum<DType>(array);
        compare(aggResults.cumsum, cumsum_);
        auto nancumsum_ = nancumsum<DType>(array);
        compare(aggResults.nancumsum, nancumsum_);
        auto mean_ = mean<DType>(array);
        compareValue(aggResults.mean, mean_);
        auto nanmean_ = nanmean<DType>(array);
        compareValue(aggResults.nanmean, nanmean_);
        auto median_ = median<DType>(array);
        compareValue(aggResults.median, median_);
        auto nanmedian_ = nanmedian<DType>(array);
        compareValue(aggResults.nanmedian, nanmedian_);
        try {
            auto cov_ = cov<DType>(array);
            compare(aggResults.cov, cov_);
            EXPECT_FALSE(aggResults.covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(aggResults.covException);
        }
        try {
            auto corrcoef_ = corrcoef<DType>(array);
            compare(aggResults.corrcoef, corrcoef_);
            EXPECT_FALSE(aggResults.corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(aggResults.corrException);
        }
        auto std__ = std_<DType>(array);
        compareValue(aggResults.std_, std__);
        auto nanstd_ = nanstd<DType>(array);
        compareValue(aggResults.nanstd, nanstd_);
        auto var_ = var<DType>(array);
        compareValue(aggResults.var, var_);
        auto nanvar_ = nanvar<DType>(array);
        compareValue(aggResults.nanvar, nanvar_);
    }

    template<typename DType>
    static void checkArrayAggregation(const NDArrayDynamic<DType> &array,
                                      const AggNaNResults<DType> &aggResults) {
        auto nansum_ = nansum<DType>(array);
        compareValue(aggResults.nansum, nansum_);
        auto nancumsum_ = nancumsum<DType>(array);
        compare(aggResults.nancumsum, nancumsum_);
        auto nanmean_ = nanmean<DType>(array);
        compareValue(aggResults.nanmean, nanmean_);
        auto nanmedian_ = nanmedian<DType>(array);
        compareValue(aggResults.nanmedian, nanmedian_);
        auto nanstd_ = nanstd<DType>(array);
        compareValue(aggResults.nanstd, nanstd_);
        auto nanvar_ = nanvar<DType>(array);
        compareValue(aggResults.nanvar, nanvar_);
    }
};

TEST_F(ArrayAggregationTest, dynamicEmptyIntArrayTest) {
    // dynamic
    Array<int_> array{};
    AggResults<int_> aggResults{};
    aggResults.covException = true;
    aggResults.corrException = true;
    checkArrayAggregation<int_>(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    AggResults<float_> aggResults{};
    aggResults.covException = true;
    aggResults.corrException = true;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static1DIntArrayTest) {
    // static
    Array<int_, 5> array{17, 28, 3, 46, 72};
    AggResults<int_> aggResults{};
    aggResults.sum = 166;
    aggResults.nansum = 166;
    aggResults.min = 3;
    aggResults.max = 72;
    aggResults.cumsum = Array<int_>{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    aggResults.nancumsum = Array<int_>{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    aggResults.mean = 33.2;
    aggResults.nanmean = 33.2;
    aggResults.median = 28;
    aggResults.nanmedian = 28;
    aggResults.cov = Array<float_>{1.0};
    aggResults.corrcoef = Array<float_>{1.0};
    aggResults.std_ = 23.961636004246458;
    aggResults.nanstd = 23.961636004246458;
    aggResults.var = 574.16000000000008;
    aggResults.nanvar = 574.16000000000008;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static1DFloatArrayTest) {
    Array<float_, 5> array{17.1, 42.2, 83.3, 24.4, 16.6};
    AggResults<float_> aggResults{};
    aggResults.sum = 183.6;
    aggResults.nansum = 183.6;
    aggResults.min = 16.6;
    aggResults.max = 83.3;
    aggResults.cumsum = Array<float_>{17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    aggResults.nancumsum = Array<float_>{17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    aggResults.mean = 36.72;
    aggResults.nanmean = 36.72;
    aggResults.median = 24.4;
    aggResults.nanmedian = 24.4;
    aggResults.cov = Array<float_>{1.0};
    aggResults.corrcoef = Array<float_>{1.0};
    aggResults.std_ = 25.064987532412616;
    aggResults.nanstd = 25.064987532412616;
    aggResults.var = 628.25359999999989;
    aggResults.nanvar = 628.25359999999989;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static1DFloatArrayNaNTest) {
    Array<float_, 7> array{17.1, NaN, 42.2, 83.3, 24.4, NaN, 16.6};
    AggNaNResults<float_> aggResults{};
    aggResults.nansum = 183.6;
    aggResults.nancumsum = Array<float_>{17.1, 17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    aggResults.nanmean = 36.72;
    aggResults.nanmedian = 24.4;
    aggResults.nanstd = 25.064987532412616;
    aggResults.nanvar = 628.25359999999989;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array{17, 28, 3, 46, 72};
    AggResults<int_> aggResults{};
    aggResults.sum = 166;
    aggResults.nansum = 166;
    aggResults.min = 3;
    aggResults.max = 72;
    aggResults.cumsum = Array<int_>{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    aggResults.nancumsum = Array<int_>{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    aggResults.mean = 33.2;
    aggResults.nanmean = 33.2;
    aggResults.median = 28;
    aggResults.nanmedian = 28;
    aggResults.cov = Array<float_>{1.0};
    aggResults.corrcoef = Array<float_>{1.0};
    aggResults.std_ = 23.961636004246458;
    aggResults.nanstd = 23.961636004246458;
    aggResults.var = 574.16000000000008;
    aggResults.nanvar = 574.16000000000008;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamic1DFloatArrayTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    AggResults<float_> aggResults{};
    aggResults.sum = 11;
    aggResults.nansum = 11;
    aggResults.min = 1.1;
    aggResults.max = 4.4;
    aggResults.cumsum = Array<float_>{1.1, 3.3, 6.6, 11};
    aggResults.nancumsum = Array<float_>{1.1, 3.3, 6.6, 11};
    aggResults.mean = 2.75;
    aggResults.nanmean = 2.75;
    aggResults.median = 2.75;
    aggResults.nanmedian = 2.75;
    aggResults.cov = Array<float_>{1};
    aggResults.corrcoef = Array<float_>{1};
    aggResults.std_ = 1.2298373876248845;
    aggResults.nanstd = 1.2298373876248845;
    aggResults.var = 1.5125000000000002;
    aggResults.nanvar = 1.5125000000000002;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamic1DFloatArrayNaNTest) {
    Array<float_> array{17.1, NaN, 42.2, 83.3, 24.4, NaN, 16.6};
    AggNaNResults<float_> aggResults{};
    aggResults.nansum = 183.6;
    aggResults.nancumsum = Array<float_>{17.1, 17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    aggResults.nanmean = 36.72;
    aggResults.nanmedian = 24.4;
    aggResults.nanstd = 25.064987532412616;
    aggResults.nanvar = 628.25359999999989;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    AggResults<int_> aggResults{};
    aggResults.sum = 21;
    aggResults.nansum = 21;
    aggResults.min = 1;
    aggResults.max = 6;
    aggResults.cumsum = Array<int_>{1, 3, 6, 10, 15, 21};
    aggResults.nancumsum = Array<int_>{1, 3, 6, 10, 15, 21};
    aggResults.mean = 3.5;
    aggResults.nanmean = 3.5;
    aggResults.median = 3.5;
    aggResults.nanmedian = 3.5;
    float_ c_array_cov[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.cov = Array<float_>{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.corrcoef = Array<float_>{c_array_corrcoef};
    aggResults.std_ = 1.707825127659933;
    aggResults.nanstd = 1.707825127659933;
    aggResults.var = 2.9166666666666665;
    aggResults.nanvar = 2.9166666666666665;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    AggResults<float_> aggResults{};
    aggResults.sum = 23.1;
    aggResults.nansum = 23.1;
    aggResults.min = 1.1;
    aggResults.max = 6.6;
    aggResults.cumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    aggResults.nancumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    aggResults.mean = 3.85;
    aggResults.nanmean = 3.85;
    aggResults.median = 3.85;
    aggResults.nanmedian = 3.85;
    float_ c_array_cov[2][2] = {{1.21, 1.21}, {1.21, 1.21}};
    aggResults.cov = Array<float_>{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.corrcoef = Array<float_>{c_array_corrcoef};
    aggResults.std_ = 1.8786076404259262;
    aggResults.nanstd = 1.8786076404259262;
    aggResults.var = 3.5291666666666663;
    aggResults.nanvar = 3.5291666666666663;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamic2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    AggResults<int_> aggResults{};
    aggResults.sum = 21;
    aggResults.nansum = 21;
    aggResults.min = 1;
    aggResults.max = 6;
    aggResults.cumsum = Array<int_>{1, 3, 6, 10, 15, 21};
    aggResults.nancumsum = Array<int_>{1, 3, 6, 10, 15, 21};
    aggResults.mean = 3.5;
    aggResults.nanmean = 3.5;
    aggResults.median = 3.5;
    aggResults.nanmedian = 3.5;
    float_ c_array_cov[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.cov = Array<float_>{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.corrcoef = Array<float_>{c_array_corrcoef};
    aggResults.std_ = 1.707825127659933;
    aggResults.nanstd = 1.707825127659933;
    aggResults.var = 2.9166666666666665;
    aggResults.nanvar = 2.9166666666666665;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, dynamic2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_> array{c_array_2d};
    AggResults<float_> aggResults{};
    aggResults.sum = 23.1;
    aggResults.nansum = 23.1;
    aggResults.min = 1.1;
    aggResults.max = 6.6;
    aggResults.cumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    aggResults.nancumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    aggResults.mean = 3.85;
    aggResults.nanmean = 3.85;
    aggResults.median = 3.85;
    aggResults.nanmedian = 3.85;
    float_ c_array_cov[2][2] = {{1.21, 1.21}, {1.21, 1.21}};
    aggResults.cov = Array<float_>{c_array_cov};
    float_ c_array_2d_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    aggResults.corrcoef = Array<float_>{c_array_2d_corrcoef};
    aggResults.std_ = 1.8786076404259262;
    aggResults.nanstd = 1.8786076404259262;
    aggResults.var = 3.5291666666666663;
    aggResults.nanvar = 3.5291666666666663;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    AggResults<int_> aggResults{};
    aggResults.sum = 78;
    aggResults.nansum = 78;
    aggResults.min = 1;
    aggResults.max = 12;
    aggResults.cumsum = Array<int_>{1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78};
    aggResults.nancumsum = Array<int_>{1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78};
    aggResults.mean = 6.5;
    aggResults.nanmean = 6.5;
    aggResults.median = 6.5;
    aggResults.nanmedian = 6.5;
    float_ c_array_cov[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
    aggResults.cov = Array<float_>{c_array_cov};
    aggResults.corrcoef = Array<float_>{c_array_cov};
    aggResults.std_ = 3.4520525295346629;
    aggResults.nanstd = 3.4520525295346629;
    aggResults.var = 11.916666666666666;
    aggResults.nanvar = 11.916666666666666;
    aggResults.covException = true;
    aggResults.corrException = true;
    checkArrayAggregation(array, aggResults);
}

TEST_F(ArrayAggregationTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    AggResults<float_> aggResults{};
    aggResults.sum = 82.83;
    aggResults.nansum = 82.83;
    aggResults.min = 1.1;
    aggResults.max = 12.12;
    aggResults.cumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1, 30.8, 39.6, 49.5, 59.6, 70.71, 82.83};
    aggResults.nancumsum = Array<float_>{1.1, 3.3, 6.6, 11, 16.5, 23.1, 30.8, 39.6, 49.5, 59.6, 70.71, 82.83};
    aggResults.mean = 6.9025;
    aggResults.nanmean = 6.9025;
    aggResults.median = 7.15;
    aggResults.nanmedian = 7.15;
    float_ c_array_cov[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
    aggResults.cov = Array<float_>{c_array_cov};
    aggResults.corrcoef = Array<float_>{c_array_cov};
    aggResults.std_ = 3.4815277417631854;
    aggResults.nanstd = 3.4815277417631854;
    aggResults.var = 12.121035416666663;
    aggResults.nanvar = 12.121035416666663;
    aggResults.covException = true;
    aggResults.corrException = true;
    checkArrayAggregation(array, aggResults);
}

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
    // static arrays
    template<typename DType, Size SizeT>
    inline void checkArrayAggregation(const NDArrayStatic<DType, SizeT> &array,
                                      DType sum_, DType min_, DType max_,
                                      const NDArrayStatic<DType, SizeT> &cumsum_,
                                      float_ mean_, float_ median_,
                                      const NDArrayDynamic<float_> &cov_,
                                      const NDArrayDynamic<float_> &corrcoef_,
                                      float_ std__,
                                      bool covException = false,
                                      bool corrException = false) {
        auto sum__ = sum<DType, SizeT>(array);
        EXPECT_EQ(sum_, sum__);
        auto min__ = min<DType, SizeT>(array);
        EXPECT_EQ(min_, min__);
        auto max__ = max<DType, SizeT>(array);
        EXPECT_EQ(max_, max__);
        auto cumsum__ = cumsum<DType, SizeT>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        if (!equal) {
            std::cerr << "Array " << cumsum_ << " is not equal to " << cumsum__ << std::endl;
        }
        EXPECT_TRUE(equal);
        auto mean__ = mean<DType, SizeT>(array);
        EXPECT_EQ(mean_, mean__);
        auto median__ = median<DType, SizeT>(array);
        EXPECT_EQ(median_, median__);
        try {
            auto cov__ = cov<DType, SizeT>(array);
            equal = array_equal(cov_, cov__);
            if (!equal) {
                std::cerr << "Array " << cov_ << " is not equal to " << cov__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(covException);
        }
        try {
            auto corrcoef__ = corrcoef<DType, SizeT>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            if (!equal) {
                std::cerr << "Array " << corrcoef_ << " is not equal to " << corrcoef__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<DType, SizeT>(array);
        EXPECT_EQ(std__, std___);
    }

    template<Size SizeT>
    inline void checkArrayAggregation(const NDArrayStatic<float_, SizeT> &array,
                                      float_ sum_, float_ min_, float_ max_,
                                      const NDArrayStatic<float_, SizeT> &cumsum_,
                                      float_ mean_, float_ median_,
                                      const NDArrayDynamic<float_> &cov_,
                                      const NDArrayDynamic<float_> &corrcoef_,
                                      float_ std__,
                                      bool covException = false,
                                      bool corrException = false) {
        auto sum__ = sum<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(sum_, sum__);
        auto min__ = min<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(min_, min__);
        auto max__ = max<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(max_, max__);
        auto cumsum__ = cumsum<float_, SizeT>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        if (!equal) {
            std::cerr << "Array " << cumsum_ << " is not equal to " << cumsum__ << std::endl;
        }
        EXPECT_TRUE(equal);
        auto mean__ = mean<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(mean_, mean__);
        auto median__ = median<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(median_, median__);
        try {
            auto cov__ = cov<float_, SizeT>(array);
            equal = array_equal(cov_, cov__);
            if (!equal) {
                std::cerr << "Array " << cov_ << " is not equal to " << cov__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(covException);
        }
        try {
            auto corrcoef__ = corrcoef<float_, SizeT>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            if (!equal) {
                std::cerr << "Array " << corrcoef_ << " is not equal to " << corrcoef__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<float_, SizeT>(array);
        EXPECT_DOUBLE_EQ(std__, std___);
    }

    // dynamic arrays
    template<typename DType>
    inline void checkArrayAggregation(const NDArrayDynamic<DType> &array,
                                      DType sum_, DType min_, DType max_,
                                      const NDArrayDynamic<DType> &cumsum_,
                                      float_ mean_, float_ median_,
                                      const NDArrayDynamic<float_> &cov_,
                                      const NDArrayDynamic<float_> &corrcoef_,
                                      float_ std__,
                                      bool covException = false,
                                      bool corrException = false) {
        auto sum__ = sum<DType>(array);
        EXPECT_EQ(sum_, sum__);
        auto min__ = min<DType>(array);
        EXPECT_EQ(min_, min__);
        auto max__ = max<DType>(array);
        EXPECT_EQ(max_, max__);
        auto cumsum__ = cumsum<DType>(array);
        bool equal = array_equal(cumsum_, cumsum__);
        if (!equal) {
            std::cerr << "Array " << cumsum_ << " is not equal to " << cumsum__ << std::endl;
        }
        EXPECT_TRUE(equal);
        auto mean__ = mean<DType>(array);
        EXPECT_EQ(mean_, mean__);
        auto median__ = median<DType>(array);
        EXPECT_EQ(median_, median__);
        try {
            auto cov__ = cov<DType>(array);
            equal = array_equal(cov_, cov__);
            if (!equal) {
                std::cerr << "Array " << cov_ << " is not equal to " << cov__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(covException);
        }
        try {
            auto corrcoef__ = corrcoef<DType>(array);
            equal = array_equal(corrcoef_, corrcoef__);
            if (!equal) {
                std::cerr << "Array " << corrcoef_ << " is not equal to " << corrcoef__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(corrException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(corrException);
        }
        auto std___ = std_<DType>(array);
        EXPECT_EQ(std__, std___);
    }

    inline void checkArrayAggregation(const NDArrayDynamic<float_> &array,
                                      float_ sum_, float_ min_, float_ max_,
                                      const NDArrayDynamic<float_> &cumsum_,
                                      float_ mean_, float_ median_,
                                      const NDArrayDynamic<float_> &cov_,
                                      const NDArrayDynamic<float_> &corrcoef_,
                                      float_ std__, bool covException, bool corrException) {
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
            auto cov__ = cov(array);
            equal = array_equal(cov_, cov__);
            if (!equal) {
                std::cerr << "Array " << cov_ << " is not equal to " << cov__ << std::endl;
            }
            EXPECT_TRUE(equal);
            EXPECT_FALSE(covException);
        } catch (const std::runtime_error &) {
            EXPECT_TRUE(covException);
        }
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
    float_ mean = 0;
    float_ median = 0;
    Array<float_> cov{};
    Array<float_> corrcoef{};
    float_ std_ = 0;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_, true, true);
}

TEST_F(ArrayAggregationTest, dynamicEmptyFloatArrayTest) {
    Array<float_> array{};
    float_ sum = 0.0;
    float_ min = 0.0;
    float_ max = 0.0;
    Array<float_> cumsum{};
    float_ mean = 0.0;
    float_ median = 0.0;
    Array<float_> cov{};
    Array<float_> corrcoef{};
    float_ std_ = 0.0;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_, true, true);
}

TEST_F(ArrayAggregationTest, static1DIntArrayTest) {
    // static
    Array<int_, 5> array{17, 28, 3, 46, 72};
    int_ sum = 166;
    int_ min = 3;
    int_ max = 72;
    Array<int_, 5> cumsum{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    float_ mean = 33.2;
    float_ median = 28;
    Array<float_> cov{1.0};
    Array<float_> corrcoef{1.0};
    float_ std_ = 23.961636004246458;
    checkArrayAggregation<int_, 5>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static1DFloatArrayTest) {
    Array<float_, 5> array{17.1, 42.2, 83.3, 24.4, 16.6};
    float_ sum = 183.6;
    float_ min = 16.6;
    float_ max = 83.3;
    Array<float_, 5> cumsum{17.1, 17.1 + 42.2, 17.1 + 42.2 + 83.3, 17.1 + 42.2 + 83.3 + 24.4, 17.1 + 42.2 + 83.3 + 24.4 + 16.6};
    float_ mean = 36.72;
    float_ median = 24.4;
    Array<float_> cov{1.0};
    Array<float_> corrcoef{1.0};
    float_ std_ = 25.064987532412616;
    checkArrayAggregation<5>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic1DIntArrayTest) {
    // dynamic
    Array<int_> array{17, 28, 3, 46, 72};
    int_ sum = 166;
    int_ min = 3;
    int_ max = 72;
    Array<int_> cumsum{17, 17 + 28, 17 + 28 + 3, 17 + 28 + 3 + 46, 17 + 28 + 3 + 46 + 72};
    float_ mean = 33.2;
    float_ median = 28;
    Array<float_> cov{1.0};
    Array<float_> corrcoef{1.0};
    float_ std_ = 23.961636004246458;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic1DFloatArrayTest) {
    Array<float_> array{1.1, 2.2, 3.3, 4.4};
    float_ sum = 11;
    float_ min = 1.1;
    float_ max = 4.4;
    Array<float_> cumsum{1.1, 3.3, 6.6, 11};
    float_ mean = 2.75;
    float_ median = 2.75;
    Array<float_> cov{1};
    Array<float_> corrcoef{1};
    float_ std_ = 1.2298373876248845;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_, 2 * 3> array{c_array_2d};
    int_ sum = 21;
    int_ min = 1;
    int_ max = 6;
    Array<int_, 2 * 3> cumsum{1, 3, 6, 10, 15, 21};
    float_ mean = 3.5;
    float_ median = 3.5;
    float_ c_array_cov[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> cov{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> corrcoef{c_array_corrcoef};
    float_ std_ = 1.707825127659933;
    checkArrayAggregation<int_, 2 * 3>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static2DFloatArrayTest) {
    double c_array_2d[2][3] = {{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}};
    Array<float_, 2 * 3> array{c_array_2d};
    float_ sum = 23.1;
    float_ min = 1.1;
    float_ max = 6.6;
    Array<float_, 2 * 3> cumsum{1.1, 3.3, 6.6, 11, 16.5, 23.1};
    float_ mean = 3.85;
    float_ median = 3.85;
    float_ c_array_cov[2][2] = {{1.21, 1.21}, {1.21, 1.21}};
    Array<float_> cov{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> corrcoef{c_array_corrcoef};
    float_ std_ = 1.8786076404259262;
    checkArrayAggregation<2 * 3>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, dynamic2DIntArrayTest) {
    long c_array_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};
    Array<int_> array{c_array_2d};
    int_ sum = 21;
    int_ min = 1;
    int_ max = 6;
    Array<int_> cumsum{1, 3, 6, 10, 15, 21};
    float_ mean = 3.5;
    float_ median = 3.5;
    float_ c_array_cov[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> cov{c_array_cov};
    float_ c_array_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> corrcoef{c_array_corrcoef};
    float_ std_ = 1.707825127659933;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
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
    float_ c_array_cov[2][2] = {{1.21, 1.21}, {1.21, 1.21}};
    Array<float_> cov{c_array_cov};
    float_ c_array_2d_corrcoef[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    Array<float_> corrcoef{c_array_2d_corrcoef};
    float_ std_ = 1.8786076404259262;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_);
}

TEST_F(ArrayAggregationTest, static3DIntArrayTest) {
    long c_array_3d[2][2][3] = {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}};
    Array<int_, 2 * 2 * 3> array{c_array_3d};
    int_ sum = 78;
    int_ min = 1;
    int_ max = 12;
    Array<int_, 2 * 2 * 3> cumsum{1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78};
    float_ mean = 6.5;
    float_ median = 6.5;
    float_ c_array_cov[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
    Array<float_> cov{c_array_cov};
    Array<float_> corrcoef{c_array_cov};
    float_ std_ = 3.4520525295346629;
    checkArrayAggregation<int_>(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_, true, true);
}

TEST_F(ArrayAggregationTest, static3DFloatArrayTest) {
    double c_array_3d[2][2][3] = {{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}},
                                  {{7.7, 8.8, 9.9}, {10.1, 11.11, 12.12}}};
    Array<float_, 2 * 2 * 3> array{c_array_3d};
    float_ sum = 82.83;
    float_ min = 1.1;
    float_ max = 12.12;
    Array<float_, 2 * 2 * 3> cumsum{1.1, 3.3, 6.6, 11, 16.5, 23.1, 30.8, 39.6, 49.5, 59.6, 70.71, 82.83};
    float_ mean = 6.9025;
    float_ median = 7.15;
    float_ c_array_cov[2][2][3] = {{{0, 0, 0}, {0, 0, 0}}, {{0, 0, 0}, {0, 0, 0}}};
    Array<float_> cov{c_array_cov};
    Array<float_> corrcoef{c_array_cov};
    float_ std_ = 3.4815277417631854;
    checkArrayAggregation(array, sum, min, max, cumsum, mean, median, cov, corrcoef, std_, true, true);
}

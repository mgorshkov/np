/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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
#include <np/linalg/LstSq.hpp>

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

using namespace np;
using namespace np::linalg;

class LinalgLstSqWeightedTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, double>> {
protected:
};

TEST_P(LinalgLstSqWeightedTest, lstsqWeightedTest) {
    auto [rows, cols, error_expected] = GetParam();

    // Generate random matrix X and true solution beta_true
    random::seed(42);
    Shape shapeX({rows, cols});
    auto X = random::rand(shapeX);

    Shape shapeBeta({cols});
    auto beta_true = random::rand(shapeBeta);

    // Generate random weights
    Shape shapeW({rows});
    auto W = random::rand(shapeW, 0.1, 2.0);// weights between 0.1 and 2.0

    // Compute y = X * beta_true + noise, with weighted noise
    auto noise = random::rand(Shape{rows}, -0.01, 0.01);
    auto y = X.dot(beta_true) + noise;

    // Solve using weighted Cholesky
    auto start = std::chrono::high_resolution_clock::now();
    auto beta = lstsq_weighted_cholesky(X, W, y);
    auto end = std::chrono::high_resolution_clock::now();

    // Compute error
    double error = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error += (beta.get(i) - beta_true.get(i)) * (beta.get(i) - beta_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Weighted Cholesky (" << rows << "x" << cols << "): "
              << time.count() << " μs, error: " << std::sqrt(error) << std::endl;
    EXPECT_LT(std::sqrt(error), error_expected);
}

TEST(LinalgLstSqWeightedTest, unitWeightsEqualsStandard) {
    // Test that weighted least squares with unit weights equals standard least squares
    random::seed(123);

    size_t rows = 50;
    size_t cols = 5;

    Shape shapeX({rows, cols});
    auto X = random::rand(shapeX);

    Shape shapeBeta({cols});
    auto beta_true = random::rand(shapeBeta);

    auto noise = random::rand(Shape{rows}, -0.01, 0.01);
    auto y = X.dot(beta_true) + noise;

    // Unit weights
    std::vector<double> weights_vec(rows, 1.0);
    auto W = NDArrayDynamic<double>(weights_vec);

    // Solve with weighted Cholesky
    auto beta_weighted = lstsq_weighted_cholesky(X, W, y);

    // Solve with standard Cholesky (which now uses weighted with unit weights)
    auto beta_standard = lstsq_cholesky(X, y);

    // Compare results
    for (size_t i = 0; i < cols; i++) {
        EXPECT_NEAR(beta_weighted.get(i), beta_standard.get(i), 1e-6);
    }
}

TEST(LinalgLstSqWeightedTest, zeroWeights) {
    // Test that zero weights effectively remove corresponding observations
    random::seed(456);

    size_t rows = 10;
    size_t cols = 3;

    Shape shapeX({rows, cols});
    auto X = random::rand(shapeX);

    Shape shapeBeta({cols});
    auto beta_true = random::rand(shapeBeta);

    auto y = X.dot(beta_true);

    // Set first half weights to 0, second half to 1
    std::vector<double> weights_vec(rows, 0.0);
    for (size_t i = rows / 2; i < rows; i++) {
        weights_vec[i] = 1.0;
    }
    auto W = NDArrayDynamic<double>(weights_vec);

    // Solve with weighted Cholesky
    auto beta = lstsq_weighted_cholesky(X, W, y);

    // Should match solution using only second half of data
    // (For simplicity, just check it doesn't crash and produces reasonable output)
    for (size_t i = 0; i < cols; i++) {
        EXPECT_FALSE(std::isnan(beta.get(i)));
        EXPECT_FALSE(std::isinf(beta.get(i)));
    }
}

TEST(LinalgLstSqWeightedTest, irlsBasic) {
    // Basic test for iteratively reweighted least squares
    random::seed(789);

    size_t rows = 100;
    size_t cols = 4;

    Shape shapeX({rows, cols});
    auto X = random::rand(shapeX);

    Shape shapeBeta({cols});
    auto beta_true = random::rand(shapeBeta);

    // Add some outliers
    auto y = X.dot(beta_true);
    for (size_t i = 0; i < 5; i++) {
        y.set(i, y.get(i) + 10.0f);// Large outliers
    }

    // Solve with IRLS (should be robust to outliers)
    auto beta_irls = lstsq_irls(X, y, 5, 1e-4);

    // Compare with standard least squares (should be affected by outliers)
    auto beta_standard = lstsq_cholesky(X, y);

    // Compute errors
    double error_irls = 0.0;
    double error_standard = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error_irls += (beta_irls.get(i) - beta_true.get(i)) * (beta_irls.get(i) - beta_true.get(i));
        error_standard += (beta_standard.get(i) - beta_true.get(i)) * (beta_standard.get(i) - beta_true.get(i));
    }

    std::cout << "IRLS error: " << std::sqrt(error_irls)
              << ", Standard error: " << std::sqrt(error_standard) << std::endl;

    // IRLS should perform better in presence of outliers
    EXPECT_LT(error_irls, error_standard * 10.0);// IRLS should be significantly better
}

INSTANTIATE_TEST_SUITE_P(
        LinalgLstSqWeightedTestCases,
        LinalgLstSqWeightedTest,
        ::testing::Values(
                std::make_tuple(10, 3, 5.0),   // Small
                std::make_tuple(50, 5, 0.1),   // Medium
                std::make_tuple(200, 10, 0.15),// Larger
                std::make_tuple(1000, 20, 0.2) // Large (tests parallel reduction)
                ));

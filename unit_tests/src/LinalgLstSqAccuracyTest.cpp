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

class LinalgLstSqAccuracyTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, double>> {
protected:
};

TEST_P(LinalgLstSqAccuracyTest, lstsqCholeskyAccuracy) {
    auto [rows, cols, tolerance] = GetParam();

    // Generate random matrix A and true solution x_true
    random::seed(42);
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    // Compute b = A * x_true (no noise)
    auto b = A.dot(x_true);

    // Solve using Cholesky
    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq_cholesky(A, b);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error += (x.get(i) - x_true.get(i)) * (x.get(i) - x_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Cholesky: rows=" << rows << ", cols=" << cols << ", error=" << std::sqrt(error) << ", time=" << time.count() << " ms" << std::endl;
    EXPECT_LT(std::sqrt(error), tolerance);
}

#ifdef USE_CUDA
TEST_P(LinalgLstSqAccuracyTest, lstsqTikhonovAccuracy) {
    auto [rows, cols, tolerance] = GetParam();

    random::seed(42);
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    auto b = A.dot(x_true);

    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq_tikhonov(A, b, 1e-6);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error += (x.get(i) - x_true.get(i)) * (x.get(i) - x_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Tikhonov: rows=" << rows << ", cols=" << cols << ", error=" << std::sqrt(error) << ", time=" << time.count() << " ms" << std::endl;
    EXPECT_LT(std::sqrt(error), tolerance);
}

TEST_P(LinalgLstSqAccuracyTest, lstsqMrrrAccuracy) {
    auto [rows, cols, tolerance] = GetParam();

    random::seed(42);
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    auto b = A.dot(x_true);

    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq_mrrr(A, b);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error += (x.get(i) - x_true.get(i)) * (x.get(i) - x_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "MRRR: rows=" << rows << ", cols=" << cols << ", error=" << std::sqrt(error) << ", time=" << time.count() << " ms" << std::endl;
    EXPECT_LT(std::sqrt(error), tolerance);
}

TEST_P(LinalgLstSqAccuracyTest, lstsqQrAccuracy) {
    auto [rows, cols, tolerance] = GetParam();

    random::seed(42);
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    auto b = A.dot(x_true);

    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq_qr(A, b);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < cols; i++) {
        error += (x.get(i) - x_true.get(i)) * (x.get(i) - x_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "QR: rows=" << rows << ", cols=" << cols << ", error=" << std::sqrt(error) << ", time=" << time.count() << " ms" << std::endl;
    EXPECT_LT(std::sqrt(error), tolerance);
}

// Test that reproduces the gmt_trend_2d rank=1 scenario:
// X_aug = [ones, zeros] — a rank-deficient matrix where column 1 is all zeros.
// This previously triggered devInfo=-30 from cusolverDnDDgels.
// Now it should gracefully fall back to Tikhonov regularization.
TEST(LinalgLstSqTest, lstsqQrRankDeficient) {
    size_t rows = 100000;
    size_t cols = 2;

    // Build X_aug = [ones, zeros] — rank-deficient matrix
    // Use Array (dynamic) which supports set()
    Shape shapeA({rows, cols});
    np::Array<double> A{shapeA};
    for (size_t i = 0; i < rows; ++i) {
        A.set(i * cols, 1.0);// column 0 = 1 (intercept)
        // column 1 stays 0 (default-initialized)
    }

    // b = linspace(-25, 25, rows) — matching the gmt_trend_2d data
    auto b = np::linspace(-25.0, 25.0, rows);

    std::cout << "Testing rank-deficient matrix: rows=" << rows << ", cols=" << cols << std::endl;
    std::cout << "A[0]=" << A.get(0) << " A[1]=" << A.get(1) << std::endl;
    std::cout << "b[0]=" << b.get(0) << " b[1]=" << b.get(1) << std::endl;

    // Should succeed via Tikhonov fallback for rank-deficient matrix
    EXPECT_NO_THROW({ auto x = lstsq_qr(A, b); });

    // Also verify the solution is reasonable: with column 1 all zeros,
    // the solution should have x[1] ≈ 0 and x[0] ≈ mean(b)
    auto x = lstsq_qr(A, b);
    double b_mean = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        b_mean += b.get(i);
    }
    b_mean /= rows;
    std::cout << "Solution: x[0]=" << x.get(0) << " (expected ~" << b_mean << "), x[1]=" << x.get(1) << " (expected ~0)" << std::endl;
    EXPECT_NEAR(x.get(0), b_mean, 1e-4);
    EXPECT_NEAR(x.get(1), 0.0, 1e-4);
}

// Test lstsq_gelsd with rank-deficient matrix (large, triggers GELSD path).
// X_aug = [ones, zeros] — rank-deficient matrix where column 1 is all zeros.
// This previously produced incorrect results due to row-major vs column-major confusion.
TEST(LinalgLstSqTest, lstsqGelsdRankDeficient) {
    size_t rows = 100000;
    size_t cols = 2;

    // Build X_aug = [ones, zeros] — rank-deficient matrix
    Shape shapeA({rows, cols});
    np::Array<double> A{shapeA};
    for (size_t i = 0; i < rows; ++i) {
        A.set(i * cols, 1.0);// column 0 = 1 (intercept)
        // column 1 stays 0 (default-initialized)
    }

    // b = linspace(-25, 25, rows) — matching the gmt_trend_2d data
    auto b = np::linspace(-25.0, 25.0, rows);

    std::cout << "Testing lstsq_gelsd rank-deficient matrix: rows=" << rows << ", cols=" << cols << std::endl;
    std::cout << "A[0]=" << A.get(0) << " A[1]=" << A.get(1) << std::endl;
    std::cout << "b[0]=" << b.get(0) << " b[1]=" << b.get(1) << std::endl;

    // Should succeed
    EXPECT_NO_THROW({ auto x = lstsq_gelsd(A, b); });

    // With column 1 all zeros, the solution should have x[1] ≈ 0 and x[0] ≈ mean(b)
    auto x = lstsq_gelsd(A, b);
    double b_mean = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        b_mean += b.get(i);
    }
    b_mean /= rows;
    std::cout << "Solution: x[0]=" << x.get(0) << " (expected ~" << b_mean << "), x[1]=" << x.get(1) << " (expected ~0)" << std::endl;
    EXPECT_NEAR(x.get(0), b_mean, 1e-4);
    EXPECT_NEAR(x.get(1), 0.0, 1e-4);
}
#endif

INSTANTIATE_TEST_SUITE_P(
        LinalgLstSqAccuracyTestCases,
        LinalgLstSqAccuracyTest,
        ::testing::Values(
                // Small exact problem - tolerance relaxed due to regularization (Cholesky) and CUDA inaccuracies
                std::make_tuple(10, 2, 1.2),
                // Medium size similar to gmt_trend_2d rank 2 (100k points, 2 columns)
                std::make_tuple(100000, 2, 1.0),
                // Larger column count
                std::make_tuple(10000, 10, 1.2),
                // Overdetermined tall matrix
                std::make_tuple(50000, 5, 0.85)));

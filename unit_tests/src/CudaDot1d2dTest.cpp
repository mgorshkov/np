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

#ifdef USE_CUDA

#include <np/Array.hpp>
#include <np/internal/cuda/Dot1d2d.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace np;

class CudaDot1d2dTest : public ::testing::Test {
protected:
    /// CPU reference implementation of dot1d2d: y[j] = sum_i x[i] * W[i * cols + j]
    static void dot1d2dRef(const float *x, const float *W, std::size_t rows, std::size_t cols, float *result) {
        for (std::size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < rows; ++i) {
                sum += x[i] * W[i * cols + j];
            }
            result[j] = sum;
        }
    }

    /// Compare two float arrays element-wise with a tolerance
    /// Uses a relaxed tolerance (1e-3f) because the CUDA kernel and CPU reference
    /// may accumulate floating-point sums in different order, producing ULP-level
    /// differences (up to 2^-13 ≈ 1.2e-4 at magnitude ~1000).
    static void expectArraysClose(const float *expected, const float *actual, std::size_t n, float tolerance = 1e-3f) {
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_NEAR(expected[i], actual[i], tolerance) << " at index " << i;
        }
    }
};

/// Test with a small known matrix and vector
TEST_F(CudaDot1d2dTest, smallKnownValues) {
    // x = [1, 2, 3]
    // W = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    // result[0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
    // result[1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 2;

    float x[rows] = {1.0f, 2.0f, 3.0f};
    float W[rows * cols] = {1.0f, 2.0f,
                            3.0f, 4.0f,
                            5.0f, 6.0f};
    float result[cols] = {0.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    float expected[cols] = {22.0f, 28.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with a 1x1 matrix (single element)
TEST_F(CudaDot1d2dTest, singleElement) {
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 1;

    float x[rows] = {5.0f};
    float W[rows * cols] = {3.0f};
    float result[cols] = {0.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    float expected[cols] = {15.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with a single row (x is scalar effectively)
TEST_F(CudaDot1d2dTest, singleRow) {
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 4;

    float x[rows] = {2.0f};
    float W[rows * cols] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result[cols] = {0.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    // result[j] = x[0] * W[0 * cols + j] = 2 * W[j]
    float expected[cols] = {2.0f, 4.0f, 6.0f, 8.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with a single column (result is a scalar)
TEST_F(CudaDot1d2dTest, singleColumn) {
    constexpr std::size_t rows = 4;
    constexpr std::size_t cols = 1;

    float x[rows] = {1.0f, 2.0f, 3.0f, 4.0f};
    float W[rows * cols] = {2.0f, 3.0f, 4.0f, 5.0f};
    float result[cols] = {0.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    // result[0] = 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    float expected[cols] = {40.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with zero values
TEST_F(CudaDot1d2dTest, zeroValues) {
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;

    float x[rows] = {0.0f, 0.0f, 0.0f};
    float W[rows * cols] = {1.0f, 2.0f, 3.0f,
                            4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f};
    float result[cols] = {-1.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    float expected[cols] = {0.0f, 0.0f, 0.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with negative values
TEST_F(CudaDot1d2dTest, negativeValues) {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 3;

    float x[rows] = {-1.0f, 2.0f};
    float W[rows * cols] = {3.0f, -4.0f, 5.0f,
                            -6.0f, 7.0f, -8.0f};
    float result[cols] = {0.0f};

    internal::cuda::dot1d2d(x, W, rows, cols, result);

    // result[0] = (-1)*3 + 2*(-6) = -3 - 12 = -15
    // result[1] = (-1)*(-4) + 2*7 = 4 + 14 = 18
    // result[2] = (-1)*5 + 2*(-8) = -5 - 16 = -21
    float expected[cols] = {-15.0f, 18.0f, -21.0f};
    expectArraysClose(expected, result, cols);
}

/// Test with a rectangular matrix where rows > cols (tall matrix)
TEST_F(CudaDot1d2dTest, tallMatrix) {
    constexpr std::size_t rows = 100;
    constexpr std::size_t cols = 10;

    std::vector<float> x(rows);
    std::vector<float> W(rows * cols);
    std::vector<float> result(cols);
    std::vector<float> expected(cols);

    // Fill with known values
    for (std::size_t i = 0; i < rows; ++i) {
        x[i] = static_cast<float>(i + 1);
        for (std::size_t j = 0; j < cols; ++j) {
            W[i * cols + j] = static_cast<float>((i + 1) * (j + 1));
        }
    }

    internal::cuda::dot1d2d(x.data(), W.data(), rows, cols, result.data());
    dot1d2dRef(x.data(), W.data(), rows, cols, expected.data());

    expectArraysClose(expected.data(), result.data(), cols);
}

/// Test with a rectangular matrix where cols > rows (wide matrix)
TEST_F(CudaDot1d2dTest, wideMatrix) {
    constexpr std::size_t rows = 10;
    constexpr std::size_t cols = 100;

    std::vector<float> x(rows);
    std::vector<float> W(rows * cols);
    std::vector<float> result(cols);
    std::vector<float> expected(cols);

    // Fill with known values
    for (std::size_t i = 0; i < rows; ++i) {
        x[i] = static_cast<float>(i * 2 + 1);
        for (std::size_t j = 0; j < cols; ++j) {
            W[i * cols + j] = static_cast<float>((i + 1) * (j + 1) % 100);
        }
    }

    internal::cuda::dot1d2d(x.data(), W.data(), rows, cols, result.data());
    dot1d2dRef(x.data(), W.data(), rows, cols, expected.data());

    expectArraysClose(expected.data(), result.data(), cols);
}

/// Test with a larger matrix to exercise multiple thread blocks
TEST_F(CudaDot1d2dTest, largeMatrixMultipleBlocks) {
    // The kernel uses blockSize = 256, so cols > 256 exercises multiple blocks
    constexpr std::size_t rows = 50;
    constexpr std::size_t cols = 512;

    std::vector<float> x(rows);
    std::vector<float> W(rows * cols);
    std::vector<float> result(cols);
    std::vector<float> expected(cols);

    // Fill with random-ish values
    for (std::size_t i = 0; i < rows; ++i) {
        x[i] = static_cast<float>((i * 7 + 3) % 100) / 10.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            W[i * cols + j] = static_cast<float>((i * 13 + j * 17 + 5) % 100) / 10.0f;
        }
    }

    internal::cuda::dot1d2d(x.data(), W.data(), rows, cols, result.data());
    dot1d2dRef(x.data(), W.data(), rows, cols, expected.data());

    expectArraysClose(expected.data(), result.data(), cols);
}

/// Test with a very large matrix to stress-test the kernel
TEST_F(CudaDot1d2dTest, veryLargeMatrix) {
    constexpr std::size_t rows = 1000;
    constexpr std::size_t cols = 1000;

    std::vector<float> x(rows);
    std::vector<float> W(rows * cols);
    std::vector<float> result(cols);
    std::vector<float> expected(cols);

    // Fill with values
    for (std::size_t i = 0; i < rows; ++i) {
        x[i] = 1.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            W[i * cols + j] = 1.0f;
        }
    }

    internal::cuda::dot1d2d(x.data(), W.data(), rows, cols, result.data());

    // All ones: result[j] = sum_i 1 * 1 = rows
    for (std::size_t j = 0; j < cols; ++j) {
        expected[j] = static_cast<float>(rows);
    }

    expectArraysClose(expected.data(), result.data(), cols);
}

/// Test that the result matches the CPU reference for random data
TEST_F(CudaDot1d2dTest, randomValuesMatchCpu) {
    constexpr std::size_t rows = 75;
    constexpr std::size_t cols = 33;

    std::vector<float> x(rows);
    std::vector<float> W(rows * cols);
    std::vector<float> result(cols);
    std::vector<float> expected(cols);

    // Fill with deterministic pseudo-random values
    for (std::size_t i = 0; i < rows; ++i) {
        x[i] = static_cast<float>((i * 31 + 17) % 1000) / 100.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            W[i * cols + j] = static_cast<float>((i * 37 + j * 41 + 23) % 1000) / 100.0f;
        }
    }

    internal::cuda::dot1d2d(x.data(), W.data(), rows, cols, result.data());
    dot1d2dRef(x.data(), W.data(), rows, cols, expected.data());

    expectArraysClose(expected.data(), result.data(), cols);
}

#endif// USE_CUDA

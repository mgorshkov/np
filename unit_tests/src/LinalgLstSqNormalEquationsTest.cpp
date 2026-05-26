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

#include <gtest/gtest.h>

using namespace np;
using namespace np::linalg;

class LinalgLstSqNormalEquationsTest : public ::testing::Test {
protected:
};

//
// Known-answer tests for lstsq_cholesky with n=2, n=3, n=4
// These exercise the unrolled fast paths in compute_normal_equations
// and would have caught the incorrect flat-index bug.
//
// The expected values are computed by solving X*beta = y using
// the normal equations beta = (X^T X)^{-1} X^T y.
//

TEST_F(LinalgLstSqNormalEquationsTest, choleskyKnownAnswerN2) {
    // X is 4x2, y is 4
    // OLS without intercept: beta = (X^T X)^{-1} X^T y
    np::float_ X_arr[4][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ y_arr[4] = {6.0, 8.0, 9.0, 11.0};
    np::Array<np::float_> y{y_arr};

    // Solve X*beta = y using Cholesky (n=2, triggers unrolled fast path)
    auto beta = lstsq_cholesky(X, y);

    // Expected: beta = [29/14, 25/14] ≈ [2.07142857, 1.78571429]
    EXPECT_NEAR(beta.get(0), 29.0 / 14.0, 1e-7);
    EXPECT_NEAR(beta.get(1), 25.0 / 14.0, 1e-7);
}

TEST_F(LinalgLstSqNormalEquationsTest, choleskyKnownAnswerN3) {
    // X is 5x3, y is 5
    // Known coefficients: beta = [2.0, -1.0, 0.5]
    np::float_ X_arr[5][3] = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 1.0},
            {5.0, 1.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[3] = {2.0, -1.0, 0.5};
    np::Array<np::float_> beta_true{beta_true_arr};

    // y = X * beta_true
    auto y = X.dot(beta_true);

    // Solve using Cholesky (n=3, triggers unrolled fast path)
    auto beta = lstsq_cholesky(X, y);

    EXPECT_NEAR(beta.get(0), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(1), -1.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 0.5, 1e-7);
}

TEST_F(LinalgLstSqNormalEquationsTest, choleskyKnownAnswerN4) {
    // X is 6x4, y is 6
    // Use a well-conditioned matrix with known coefficients beta = [1, 2, 3, 4]
    np::float_ X_arr[6][4] = {
            {2.0, 1.0, 0.0, 1.0},
            {1.0, 3.0, 1.0, 0.0},
            {0.0, 1.0, 4.0, 1.0},
            {1.0, 0.0, 1.0, 3.0},
            {3.0, 2.0, 0.0, 1.0},
            {1.0, 1.0, 2.0, 0.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[4] = {1.0, 2.0, 3.0, 4.0};
    np::Array<np::float_> beta_true{beta_true_arr};

    // y = X * beta_true
    auto y = X.dot(beta_true);

    // Solve using Cholesky (n=4, triggers unrolled fast path)
    auto beta = lstsq_cholesky(X, y);

    EXPECT_NEAR(beta.get(0), 1.0, 1e-7);
    EXPECT_NEAR(beta.get(1), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 3.0, 1e-7);
    EXPECT_NEAR(beta.get(3), 4.0, 1e-7);
}

//
// Known-answer tests for lstsq_weighted_cholesky with n=2, n=3, n=4
// These exercise the unrolled fast paths in compute_weighted_normal_equations
//

TEST_F(LinalgLstSqNormalEquationsTest, weightedCholeskyKnownAnswerN2) {
    // X is 4x2, y is 4, with sample weights
    np::float_ X_arr[4][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ y_arr[4] = {6.0, 8.0, 9.0, 11.0};
    np::Array<np::float_> y{y_arr};
    np::float_ w_arr[4] = {4.0, 0.5, 2.0, 3.0};
    np::Array<np::float_> w{w_arr};

    // Solve weighted least squares (n=2, triggers unrolled fast path)
    auto beta = lstsq_weighted_cholesky(X, w, y);

    // Expected: beta = [X^T W X]^{-1} X^T W y
    // X^T W X = [[39.5, 49.0], [49.0, 62.0]]
    // X^T W y = [163, 200]
    // det = 39.5*62.0 - 49.0*49.0 = 2449 - 2401 = 48
    // beta[0] = (62.0*163 - 49.0*200) / 48 = (10106 - 9800) / 48 = 306/48 = 6.375
    // beta[1] = (-49.0*163 + 39.5*200) / 48 = (-7987 + 7900) / 48 = -87/48 = -1.8125
    EXPECT_NEAR(beta.get(0), 306.0 / 48.0, 1e-6);
    EXPECT_NEAR(beta.get(1), -87.0 / 48.0, 1e-6);
}

TEST_F(LinalgLstSqNormalEquationsTest, weightedCholeskyKnownAnswerN3) {
    // X is 5x3, y is 5, with sample weights
    np::float_ X_arr[5][3] = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 1.0},
            {5.0, 1.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[3] = {2.0, -1.0, 0.5};
    np::Array<np::float_> beta_true{beta_true_arr};

    // y = X * beta_true
    auto y = X.dot(beta_true);

    // Non-uniform weights
    np::float_ w_arr[5] = {0.5, 1.0, 2.0, 0.8, 1.5};
    np::Array<np::float_> w{w_arr};

    // Solve weighted least squares (n=3, triggers unrolled fast path)
    auto beta = lstsq_weighted_cholesky(X, w, y);

    // With exact data (no noise), weighted should recover exact coefficients
    EXPECT_NEAR(beta.get(0), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(1), -1.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 0.5, 1e-7);
}

TEST_F(LinalgLstSqNormalEquationsTest, weightedCholeskyKnownAnswerN4) {
    // X is 6x4, y is 6, with sample weights
    np::float_ X_arr[6][4] = {
            {2.0, 1.0, 0.0, 1.0},
            {1.0, 3.0, 1.0, 0.0},
            {0.0, 1.0, 4.0, 1.0},
            {1.0, 0.0, 1.0, 3.0},
            {3.0, 2.0, 0.0, 1.0},
            {1.0, 1.0, 2.0, 0.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[4] = {1.0, 2.0, 3.0, 4.0};
    np::Array<np::float_> beta_true{beta_true_arr};

    // y = X * beta_true
    auto y = X.dot(beta_true);

    // Non-uniform weights
    np::float_ w_arr[6] = {0.3, 1.2, 0.7, 2.0, 0.5, 1.8};
    np::Array<np::float_> w{w_arr};

    // Solve weighted least squares (n=4, triggers unrolled fast path)
    auto beta = lstsq_weighted_cholesky(X, w, y);

    // With exact data (no noise), weighted should recover exact coefficients
    EXPECT_NEAR(beta.get(0), 1.0, 1e-7);
    EXPECT_NEAR(beta.get(1), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 3.0, 1e-7);
    EXPECT_NEAR(beta.get(3), 4.0, 1e-7);
}

//
// Verify that lstsq (auto-dispatch) also works correctly for small n
// where it would use GELSD (m <= 100 && n <= 10)
//

TEST_F(LinalgLstSqNormalEquationsTest, lstsqKnownAnswerN2) {
    np::float_ X_arr[4][2] = {{1.0, 1.0}, {1.0, 2.0}, {2.0, 2.0}, {3.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ y_arr[4] = {6.0, 8.0, 9.0, 11.0};
    np::Array<np::float_> y{y_arr};

    // lstsq auto-dispatch: m=4 <= 100, n=2 <= 10 → uses GELSD
    auto beta = lstsq(X, y);

    EXPECT_NEAR(beta.get(0), 29.0 / 14.0, 1e-7);
    EXPECT_NEAR(beta.get(1), 25.0 / 14.0, 1e-7);
}

TEST_F(LinalgLstSqNormalEquationsTest, lstsqKnownAnswerN3) {
    np::float_ X_arr[5][3] = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 1.0},
            {5.0, 1.0, 4.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[3] = {2.0, -1.0, 0.5};
    np::Array<np::float_> beta_true{beta_true_arr};
    auto y = X.dot(beta_true);

    // lstsq auto-dispatch: m=5 <= 100, n=3 <= 10 → uses GELSD
    auto beta = lstsq(X, y);

    EXPECT_NEAR(beta.get(0), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(1), -1.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 0.5, 1e-7);
}

TEST_F(LinalgLstSqNormalEquationsTest, lstsqKnownAnswerN4) {
    np::float_ X_arr[6][4] = {
            {2.0, 1.0, 0.0, 1.0},
            {1.0, 3.0, 1.0, 0.0},
            {0.0, 1.0, 4.0, 1.0},
            {1.0, 0.0, 1.0, 3.0},
            {3.0, 2.0, 0.0, 1.0},
            {1.0, 1.0, 2.0, 0.0}};
    np::Array<np::float_> X{X_arr};
    np::float_ beta_true_arr[4] = {1.0, 2.0, 3.0, 4.0};
    np::Array<np::float_> beta_true{beta_true_arr};
    auto y = X.dot(beta_true);

    // lstsq auto-dispatch: m=6 <= 100, n=4 <= 10 → uses GELSD
    auto beta = lstsq(X, y);

    EXPECT_NEAR(beta.get(0), 1.0, 1e-7);
    EXPECT_NEAR(beta.get(1), 2.0, 1e-7);
    EXPECT_NEAR(beta.get(2), 3.0, 1e-7);
    EXPECT_NEAR(beta.get(3), 4.0, 1e-7);
}

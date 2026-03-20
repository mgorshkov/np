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

#pragma once

#include <cmath>
#include <np/Array.hpp>
#include <np/Exception.hpp>
#include <np/linalg/Inv.hpp>
#include <np/linalg/Pinv.hpp>
#include <vector>

namespace np {
    namespace linalg {
        namespace internal {
            // Cholesky decomposition: A = L * L^T where L is lower triangular
            // Returns L (lower triangular) or throws if matrix is not positive definite
            template<typename DType>
            std::vector<DType> cholesky_decompose(const DType *A, size_t n) {
                std::vector<DType> L(n * n, 0);

                // Fast path for n <= 4: manually unrolled to avoid loop overhead
                // and enable better compiler optimization for tiny matrices
                if (n <= 4) {
                    switch (n) {
                        case 3: {
                            // L[0][0] = sqrt(A[0][0])
                            L[0] = std::sqrt(A[0]);
                            // L[1][0] = A[1][0] / L[0][0]
                            L[3] = A[3] / L[0];
                            // L[1][1] = sqrt(A[1][1] - L[1][0]^2)
                            L[4] = std::sqrt(A[4] - L[3] * L[3]);
                            // L[2][0] = A[2][0] / L[0][0]
                            L[6] = A[6] / L[0];
                            // L[2][1] = (A[2][1] - L[2][0]*L[1][0]) / L[1][1]
                            L[7] = (A[7] - L[6] * L[3]) / L[4];
                            // L[2][2] = sqrt(A[2][2] - L[2][0]^2 - L[2][1]^2)
                            L[8] = std::sqrt(A[8] - L[6] * L[6] - L[7] * L[7]);
                            break;
                        }
                        case 2: {
                            // L[0][0] = sqrt(A[0][0])
                            L[0] = std::sqrt(A[0]);
                            // L[1][0] = A[1][0] / L[0][0]
                            L[2] = A[2] / L[0];
                            // L[1][1] = sqrt(A[1][1] - L[1][0]^2)
                            L[3] = std::sqrt(A[3] - L[2] * L[2]);
                            break;
                        }
                        case 1: {
                            L[0] = std::sqrt(A[0]);
                            break;
                        }
                        default:
                            break;
                    }
                    return L;
                }

                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        DType sum = 0;
#ifdef USE_OPENMP
#pragma omp simd reduction(+ : sum)
#endif
                        for (size_t k = 0; k < j; ++k) {
                            sum += L[i * n + k] * L[j * n + k];
                        }

                        if (i == j) {
                            DType diag = A[i * n + i] - sum;
                            if (diag <= 0) {
                                NP_THROW_WITH_STACKTRACE(std::runtime_error,
                                                         "Matrix is not positive definite in Cholesky decomposition");
                            }
                            L[i * n + i] = std::sqrt(diag);
                        } else {
                            L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
                        }
                    }
                }
                return L;
            }

            // Solve L * L^T * x = b using forward and backward substitution
            template<typename DType>
            std::vector<DType> cholesky_solve(const DType *L, const DType *b, size_t n) {
                std::vector<DType> y(n, 0);
                std::vector<DType> x(n, 0);

                // Fast path for n <= 4: manually unrolled to avoid loop overhead
                if (n <= 4) {
                    switch (n) {
                        case 3: {
                            // Forward substitution: L * y = b
                            // L is lower triangular, stored column-major in the flat array
                            // L[0]=L00, L[3]=L10, L[6]=L20, L[4]=L11, L[7]=L21, L[8]=L22
                            y[0] = b[0] / L[0];
                            y[1] = (b[1] - L[3] * y[0]) / L[4];
                            y[2] = (b[2] - L[6] * y[0] - L[7] * y[1]) / L[8];
                            // Backward substitution: L^T * x = y
                            // L^T is upper triangular: L[0]=L00, L[3]=L01, L[6]=L02, L[4]=L11, L[7]=L12, L[8]=L22
                            x[2] = y[2] / L[8];
                            x[1] = (y[1] - L[7] * x[2]) / L[4];
                            x[0] = (y[0] - L[3] * x[1] - L[6] * x[2]) / L[0];
                            break;
                        }
                        case 2: {
                            // Forward substitution
                            y[0] = b[0] / L[0];
                            y[1] = (b[1] - L[2] * y[0]) / L[3];
                            // Backward substitution
                            x[1] = y[1] / L[3];
                            x[0] = (y[0] - L[2] * x[1]) / L[0];
                            break;
                        }
                        case 1: {
                            x[0] = b[0] / L[0];
                            break;
                        }
                        default:
                            break;
                    }
                    return x;
                }

                // Forward substitution: L * y = b
                for (size_t i = 0; i < n; ++i) {
                    DType sum = 0;
#ifdef USE_OPENMP
#pragma omp simd reduction(+ : sum)
#endif
                    for (size_t j = 0; j < i; ++j) {
                        sum += L[i * n + j] * y[j];
                    }
                    y[i] = (b[i] - sum) / L[i * n + i];
                }

                // Backward substitution: L^T * x = y
                for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
                    DType sum = 0;
#ifdef USE_OPENMP
#pragma omp simd reduction(+ : sum)
#endif
                    for (size_t j = i + 1; j < n; ++j) {
                        sum += L[j * n + i] * x[j];// Note: L[j * n + i] is L^T[i][j]
                    }
                    x[i] = (y[i] - sum) / L[i * n + i];
                }

                return x;
            }

            // Compute X^T * X and X^T * y (unweighted normal equations)
            // X is m x n, y is size m
            // Returns pair (A, b) where A = X^T X (n x n) and b = X^T y (n)
            //
            // For small n (<= 4), uses a fast single-threaded path with explicit
            // loop unrolling to avoid OpenMP overhead. For larger n, uses OpenMP
            // parallel reduction over rows.
            template<typename DType>
            std::pair<std::vector<DType>, std::vector<DType>> compute_normal_equations(
                    const DType *X, const DType *y, size_t m, size_t n) {

                std::vector<DType> A(n * n, 0);
                std::vector<DType> b(n, 0);

                // Fast path for small n (typical for IRLS with few features)
                if (n <= 4) {
                    for (size_t i = 0; i < m; ++i) {
                        const DType *row = X + i * n;
                        DType yi = y[i];

                        switch (n) {
                            case 4: {
                                DType r0 = row[0], r1 = row[1], r2 = row[2], r3 = row[3];
                                b[0] += r0 * yi;
                                b[1] += r1 * yi;
                                b[2] += r2 * yi;
                                b[3] += r3 * yi;
                                A[0] += r0 * r0;
                                A[1] += r1 * r0;
                                A[5] += r1 * r1;
                                A[2] += r2 * r0;
                                A[6] += r2 * r1;
                                A[10] += r2 * r2;
                                A[3] += r3 * r0;
                                A[7] += r3 * r1;
                                A[11] += r3 * r2;
                                A[15] += r3 * r3;
                                break;
                            }
                            case 3: {
                                DType r0 = row[0], r1 = row[1], r2 = row[2];
                                b[0] += r0 * yi;
                                b[1] += r1 * yi;
                                b[2] += r2 * yi;
                                A[0] += r0 * r0;
                                A[1] += r1 * r0;
                                A[4] += r1 * r1;
                                A[2] += r2 * r0;
                                A[5] += r2 * r1;
                                A[8] += r2 * r2;
                                break;
                            }
                            case 2: {
                                DType r0 = row[0], r1 = row[1];
                                b[0] += r0 * yi;
                                b[1] += r1 * yi;
                                A[0] += r0 * r0;
                                A[1] += r1 * r0;
                                A[3] += r1 * r1;
                                break;
                            }
                            case 1: {
                                b[0] += row[0] * yi;
                                A[0] += row[0] * row[0];
                                break;
                            }
                        }
                    }
                    // Fill upper triangular part (symmetric matrix)
                    for (size_t j = 0; j < n; ++j)
                        for (size_t k = j + 1; k < n; ++k)
                            A[j * n + k] = A[k * n + j];
                    return {A, b};
                }

// Parallel reduction over rows using OpenMP (for larger n)
#ifdef USE_OPENMP
#pragma omp parallel default(none) shared(X, y, A, b, m, n)
#endif
                {
                    std::vector<DType> local_A(n * n, 0);
                    std::vector<DType> local_b(n, 0);

#ifdef USE_OPENMP
#pragma omp for nowait
#endif
                    for (std::int64_t i = 0; i < static_cast<std::int64_t>(m); ++i) {
                        const DType *row = X + i * n;
                        DType yi = y[i];

                        for (size_t j = 0; j < n; ++j) {
                            local_b[j] += row[j] * yi;
                            for (size_t k = 0; k <= j; ++k) {
                                local_A[j * n + k] += row[j] * row[k];
                            }
                        }
                    }

#ifdef USE_OPENMP
#pragma omp critical
#endif
                    {
                        for (size_t j = 0; j < n; ++j) {
                            b[j] += local_b[j];
                            for (size_t k = 0; k <= j; ++k) {
                                A[j * n + k] += local_A[j * n + k];
                            }
                        }
                    }
                }

                // Fill upper triangular part (symmetric matrix)
                for (size_t j = 0; j < n; ++j)
                    for (size_t k = j + 1; k < n; ++k)
                        A[j * n + k] = A[k * n + j];

                return {A, b};
            }

            // Compute X^T * W * X and X^T * W * y using parallel reduction
            // X is m x n, W is diagonal (size m), y is size m
            // Returns pair (A, b) where A = X^T W X (n x n) and b = X^T W y (n)
            //
            // For small n (<= 4), uses a fast single-threaded path with explicit
            // loop unrolling to avoid OpenMP overhead. For larger n, uses OpenMP
            // parallel reduction over rows.
            template<typename DType>
            std::pair<std::vector<DType>, std::vector<DType>> compute_weighted_normal_equations(
                    const DType *X, const DType *W, const DType *y, size_t m, size_t n) {

                std::vector<DType> A(n * n, 0);
                std::vector<DType> b(n, 0);

                // Fast path for small n (typical for IRLS with few features):
                // Avoid OpenMP thread management overhead and use flat loops.
                // For n <= 4, the inner loops are small enough that the compiler
                // can auto-vectorize effectively, and OpenMP overhead dominates.
                if (n <= 4) {
                    for (size_t i = 0; i < m; ++i) {
                        DType weight = W[i];
                        if (weight == 0) continue;

                        const DType *row = X + i * n;

                        // Manually unrolled for n <= 4 to help compiler vectorize
                        switch (n) {
                            case 4: {
                                DType w_row_0 = weight * row[0];
                                DType w_row_1 = weight * row[1];
                                DType w_row_2 = weight * row[2];
                                DType w_row_3 = weight * row[3];
                                b[0] += w_row_0 * y[i];
                                b[1] += w_row_1 * y[i];
                                b[2] += w_row_2 * y[i];
                                b[3] += w_row_3 * y[i];
                                A[0] += w_row_0 * row[0];
                                A[1] += w_row_1 * row[0];
                                A[5] += w_row_1 * row[1];
                                A[2] += w_row_2 * row[0];
                                A[6] += w_row_2 * row[1];
                                A[10] += w_row_2 * row[2];
                                A[3] += w_row_3 * row[0];
                                A[7] += w_row_3 * row[1];
                                A[11] += w_row_3 * row[2];
                                A[15] += w_row_3 * row[3];
                                break;
                            }
                            case 3: {
                                DType w_row_0 = weight * row[0];
                                DType w_row_1 = weight * row[1];
                                DType w_row_2 = weight * row[2];
                                b[0] += w_row_0 * y[i];
                                b[1] += w_row_1 * y[i];
                                b[2] += w_row_2 * y[i];
                                A[0] += w_row_0 * row[0];
                                A[1] += w_row_1 * row[0];
                                A[4] += w_row_1 * row[1];
                                A[2] += w_row_2 * row[0];
                                A[5] += w_row_2 * row[1];
                                A[8] += w_row_2 * row[2];
                                break;
                            }
                            case 2: {
                                DType w_row_0 = weight * row[0];
                                DType w_row_1 = weight * row[1];
                                b[0] += w_row_0 * y[i];
                                b[1] += w_row_1 * y[i];
                                A[0] += w_row_0 * row[0];
                                A[1] += w_row_1 * row[0];
                                A[3] += w_row_1 * row[1];
                                break;
                            }
                            case 1: {
                                DType w_row_0 = weight * row[0];
                                b[0] += w_row_0 * y[i];
                                A[0] += w_row_0 * row[0];
                                break;
                            }
                        }
                    }
                    // Fill upper triangular part (symmetric matrix)
                    for (size_t j = 0; j < n; ++j) {
                        for (size_t k = j + 1; k < n; ++k) {
                            A[j * n + k] = A[k * n + j];
                        }
                    }
                    return {A, b};
                }

// Parallel reduction over rows using OpenMP (for larger n)
#ifdef USE_OPENMP
#pragma omp parallel default(none) shared(X, W, y, A, b, m, n)
#endif
                {
                    // Thread-local accumulators
                    std::vector<DType> local_A(n * n, 0);
                    std::vector<DType> local_b(n, 0);

#ifdef USE_OPENMP
#pragma omp for nowait
#endif
                    // index variable in OpenMP 'for' statement must have signed integral type
                    for (std::int64_t i = 0; i < static_cast<std::int64_t>(m); ++i) {
                        DType weight = W[i];
                        if (weight == 0) continue;

                        const DType *row = X + i * n;

                        // Update X^T W X: A += weight * (row^T * row)
                        // Update X^T W y: b += weight * row * y[i]
                        for (size_t j = 0; j < n; ++j) {
                            DType w_row_j = weight * row[j];
                            local_b[j] += w_row_j * y[i];

                            for (size_t k = 0; k <= j; ++k) {
                                local_A[j * n + k] += w_row_j * row[k];
                            }
                        }
                    }

// Reduce thread-local accumulators into global accumulators
#ifdef USE_OPENMP
#pragma omp critical
#endif
                    {
                        for (size_t j = 0; j < n; ++j) {
                            b[j] += local_b[j];
                            for (size_t k = 0; k <= j; ++k) {
                                A[j * n + k] += local_A[j * n + k];
                            }
                        }
                    }
                }

                // Fill upper triangular part (symmetric matrix)
                for (size_t j = 0; j < n; ++j) {
                    for (size_t k = j + 1; k < n; ++k) {
                        A[j * n + k] = A[k * n + j];
                    }
                }

                return {A, b};
            }
        }// namespace internal

        // Weighted least squares using Cholesky decomposition
        // Solves: argmin ||W^(1/2)(y - Xβ)||²
        // where W is diagonal weight matrix (size m)
        template<typename DType, typename DerivedX, typename StorageX,
                 typename DerivedW, typename StorageW,
                 typename DerivedY, typename StorageY>
        auto lstsq_weighted_cholesky(
                const ndarray::internal::NDArrayBase<DType, DerivedX, StorageX> &X,
                const ndarray::internal::NDArrayBase<DType, DerivedW, StorageW> &W,
                const ndarray::internal::NDArrayBase<DType, DerivedY, StorageY> &y,
                DType regularization = 1e-8) {

            if (X.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array for X.");
            }
            if (W.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array for weights.");
            }
            if (y.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array for y.");
            }

            auto m = X.shape()[0];
            auto n = X.shape()[1];

            if (W.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument,
                                         "Weight vector size must match number of rows in X.");
            }
            if (y.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument,
                                         "y vector size must match number of rows in X.");
            }

            // Compute normal equations: A = X^T W X, b = X^T W y
            // If y is already contiguous, use its data() directly to avoid a copy.
            // Otherwise, evaluate expression templates to a concrete array first.
            const DType *y_data;
            NDArrayDynamic<DType> y_eval;// holds evaluated copy if needed
            if constexpr (StorageY::is_contiguous) {
                y_data = y.data();
            } else {
                y_eval = y.derived().eval();
                y_data = y_eval.data();
            }

            auto [A_vec, b_vec] = internal::compute_weighted_normal_equations(
                    X.data(), W.data(), y_data, m, n);

            // Try Cholesky with progressively stronger regularization if the matrix
            // is not positive definite (e.g., due to near-zero weights in IRLS).
            DType current_reg = regularization;
            const DType max_reg = 1e-2;
            const DType reg_growth = 10.0;

            while (true) {
                // Make a copy of A_vec so we can retry with different regularization
                std::vector<DType> A_reg = A_vec;

                // Add regularization to ensure positive definiteness
                for (size_t i = 0; i < n; ++i) {
                    A_reg[i * n + i] += current_reg;
                }

                try {
                    // Perform Cholesky decomposition
                    auto L = internal::cholesky_decompose(A_reg.data(), n);

                    // Solve using Cholesky
                    auto x_vec = internal::cholesky_solve(L.data(), b_vec.data(), n);

                    // Convert to NDArray
                    return NDArrayDynamic<DType>(x_vec);
                } catch (const std::runtime_error &) {
                    if (current_reg >= max_reg) {
                        // Fallback: use pseudoinverse via Tikhonov-regularized normal equations
                        // Solve (A^T W A + λI) β = A^T W y using matrix inversion
                        // A_reg already has the regularization added at current_reg level
                        // Try to invert using the regularized matrix
                        NDArrayDynamic<DType> A_reg_array(A_reg, Shape{n, n});
                        NDArrayDynamic<DType> b_vec_array(b_vec, Shape{n});
                        try {
                            auto A_inv = inv(A_reg_array);
                            auto x_vec = A_inv.dot(b_vec_array);
                            std::vector<DType> x(n);
                            for (size_t i = 0; i < n; ++i) {
                                x[i] = x_vec.get(i);
                            }
                            return NDArrayDynamic<DType>(x);
                        } catch (...) {
                            // Last resort: use pinv with very strong regularization
                            auto A_reg_arr = NDArrayDynamic<DType>(A_reg, Shape{n, n});
                            auto A_pinv = pinv(A_reg_arr, 1e-4);
                            auto b_vec_arr = NDArrayDynamic<DType>(b_vec, Shape{n});
                            auto x_vec = A_pinv.dot(b_vec_arr);
                            std::vector<DType> x(n);
                            for (size_t i = 0; i < n; ++i) {
                                x[i] = x_vec.get(i);
                            }
                            return NDArrayDynamic<DType>(x);
                        }
                    }
                    // Increase regularization and retry
                    current_reg *= reg_growth;
                }
            }
        }

        // Iteratively reweighted least squares (IRLS)
        // Performs weighted least squares with weight updates each iteration
        template<typename DType, typename DerivedX, typename StorageX,
                 typename DerivedY, typename StorageY>
        auto lstsq_irls(
                const ndarray::internal::NDArrayBase<DType, DerivedX, StorageX> &X,
                const ndarray::internal::NDArrayBase<DType, DerivedY, StorageY> &y,
                size_t max_iter = 10,
                DType tolerance = 1e-6,
                DType regularization = 1e-8) {

            auto m = X.shape()[0];
            auto n = X.shape()[1];

            // Initialize weights to 1
            std::vector<DType> weights_vec(m, 1.0);
            auto weights = NDArrayDynamic<DType>(weights_vec);

            std::vector<DType> beta_prev(n, 0);
            std::vector<DType> beta(n, 0);

            for (size_t iter = 0; iter < max_iter; ++iter) {
                // Solve weighted least squares
                auto beta_array = lstsq_weighted_cholesky(X, weights, y, regularization);

                // Copy to vector
                for (size_t i = 0; i < n; ++i) {
                    beta[i] = beta_array.get(i);
                }

                // Check convergence
                if (iter > 0) {
                    DType diff = 0;
                    for (size_t i = 0; i < n; ++i) {
                        DType delta = beta[i] - beta_prev[i];
                        diff += delta * delta;
                    }
                    diff = std::sqrt(diff);
                    if (diff < tolerance) {
                        break;
                    }
                }

                // Update weights based on residuals (example: Huber weights)
                std::vector<DType> residuals(m);
                for (size_t i = 0; i < m; ++i) {
                    DType pred = 0;
                    for (size_t j = 0; j < n; ++j) {
                        pred += X.get(i * n + j) * beta[j];
                    }
                    residuals[i] = std::abs(y.get(i) - pred);
                }

                // Update weights (Huber weighting: w = 1 / max(1, |r|/c))
                DType c = 1.345;// Huber constant
                for (size_t i = 0; i < m; ++i) {
                    DType r = residuals[i];
                    weights_vec[i] = 1.0 / std::max(1.0, r / c);
                }
                weights = NDArrayDynamic<DType>(weights_vec);

                // Store current beta for next iteration
                beta_prev = beta;
            }

            return NDArrayDynamic<DType>(beta);
        }
    }// namespace linalg
}// namespace np

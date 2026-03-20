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

#include <cstring>

#include <np/Array.hpp>
#include <np/Exception.hpp>
#include <np/internal/cpu/LstSqGelsd.hpp>
#include <np/internal/cuda/Mrrr.hpp>
#include <np/internal/cuda/Tikhonov.hpp>
#include <np/internal/cuda/lstsqQr.hpp>
#include <np/linalg/Cholesky.hpp>
#include <np/linalg/Inv.hpp>
#include <np/linalg/Pinv.hpp>

namespace np {
    namespace linalg {
        // Threshold constants for algorithm selection in lstsq()
        // Both m (rows) and n (cols) are considered independently.
        //
        // GELSD (SIMD SVD) is fastest for very tiny problems (e.g. 100×10)
        static constexpr size_t kGelsdMaxRows = 100;
        static constexpr size_t kGelsdMaxCols = 10;
        // Cholesky is the default CPU solver for most problem sizes
        //
        // CUDA solvers are used for large matrices where Cholesky's O(cols²) workspace grows.
        // Tikhonov (EVD-based) is preferred when both dimensions are large (≥10000 rows, ≥500 cols).
        // QR (cuSOLVER gels) is used for other large matrices.
        static constexpr size_t kCudaTikhonovMinRows = 10000;
        static constexpr size_t kCudaTikhonovMinCols = 500;

        // Least squares with Cholesky (Use Cholesky for small matrices)
        // Solves A x = b for x, where A is (m x n), b is (m)
        // Returns coefficients x (including intercept if A has intercept column)
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq_cholesky(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                                   const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b,
                                   DType regularization = 1e-8) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }

            // Compute normal equations directly (unweighted) to avoid
            // creating a unit weight vector and multiplying by 1.0
            const DType *b_data;
            NDArrayDynamic<DType> b_eval;
            if constexpr (StorageB::is_contiguous) {
                b_data = b.data();
            } else {
                b_eval = b.derived().eval();
                b_data = b_eval.data();
            }

            auto [A_vec, b_vec] = internal::compute_normal_equations(
                    A.data(), b_data, m, n);

            // Try Cholesky with progressively stronger regularization
            DType current_reg = regularization;
            const DType max_reg = 1e-2;
            const DType reg_growth = 10.0;

            while (true) {
                std::vector<DType> A_reg = A_vec;
                for (size_t i = 0; i < n; ++i) {
                    A_reg[i * n + i] += current_reg;
                }

                try {
                    auto L = internal::cholesky_decompose(A_reg.data(), n);
                    auto x_vec = internal::cholesky_solve(L.data(), b_vec.data(), n);
                    return NDArrayDynamic<DType>(x_vec);
                } catch (const std::runtime_error &) {
                    if (current_reg >= max_reg) {
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
                    current_reg *= reg_growth;
                }
            }
        }

        // Forward declarations for functions defined in Cholesky.hpp
        template<typename DType, typename DerivedX, typename StorageX,
                 typename DerivedW, typename StorageW,
                 typename DerivedY, typename StorageY>
        auto lstsq_weighted_cholesky(
                const ndarray::internal::NDArrayBase<DType, DerivedX, StorageX> &X,
                const ndarray::internal::NDArrayBase<DType, DerivedW, StorageW> &W,
                const ndarray::internal::NDArrayBase<DType, DerivedY, StorageY> &y,
                DType regularization);

        template<typename DType, typename DerivedX, typename StorageX,
                 typename DerivedY, typename StorageY>
        auto lstsq_irls(
                const ndarray::internal::NDArrayBase<DType, DerivedX, StorageX> &X,
                const ndarray::internal::NDArrayBase<DType, DerivedY, StorageY> &y,
                size_t max_iter,
                DType tolerance,
                DType regularization);

        // Helper to materialize an NDArrayBase into a contiguous NDArrayDynamic for data() access.
        // This is needed because the input may be an expression template (e.g., BinaryExpression)
        // which does not have contiguous storage.
        template<typename DType, typename Derived, typename Storage>
        inline NDArrayDynamic<DType> materialize(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
            if constexpr (Storage::is_contiguous) {
                // Fast path: copy the contiguous data directly via memcpy
                NDArrayDynamic<DType> result{array.shape()};
                std::memcpy(result.data(), array.data(), array.size() * sizeof(DType));
                return result;
            } else {
                // Slow path: copy element-by-element for non-contiguous storage
                NDArrayDynamic<DType> result{array.shape()};
                for (Size i = 0; i < array.size(); ++i) {
                    result.set(i, array.get(i));
                }
                return result;
            }
        }

        // Least squares with GELSD (divide-and-conquer SVD) on CPU.
        // This is the same algorithm as numpy.linalg.lstsq (LAPACK DGELSD/SGELSD).
        // Uses SIMD-optimized implementation with automatic dispatch.
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq_gelsd(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                                const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b,
                                DType rcond = -1) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }
            // Materialize to ensure contiguous storage for data() access
            auto A_mat = materialize(A);
            auto b_mat = materialize(b);
            std::vector<DType> x(n);
            if constexpr (std::is_same_v<DType, double>) {
                np::internal::cpu::lstsq_gelsd_double(A_mat.data(), b_mat.data(), x.data(), m, n, static_cast<double>(rcond));
            } else {
                np::internal::cpu::lstsq_gelsd_float(A_mat.data(), b_mat.data(), x.data(), m, n, static_cast<float>(rcond));
            }
            return NDArrayDynamic<DType>(x);
        }

#ifdef USE_CUDA
        // Least squares with Tikhonov Regularized EVD
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq_tikhonov(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                                   const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b, DType lambda = 1e-6) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }
            // Materialize to ensure contiguous storage for data() access
            auto A_mat = materialize(A);
            auto b_mat = materialize(b);
            std::vector<DType> x(n);
            np::internal::cuda::lstsqTikhonov(A_mat.data(), b_mat.data(), x.data(), m, n, lambda);
            return NDArrayDynamic<DType>(x);
        }

        // Least squares with MRRR
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq_mrrr(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                               const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }
            // Materialize to ensure contiguous storage for data() access
            auto A_mat = materialize(A);
            auto b_mat = materialize(b);
            std::vector<DType> x(n);
            np::internal::cuda::lstsqMrrr(A_mat.data(), b_mat.data(), x.data(), m, n);
            return NDArrayDynamic<DType>(x);
        }

        // Least squares with QR decomposition (cuSOLVER gels)
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq_qr(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                             const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }
            // Materialize to ensure contiguous storage for data() access
            auto A_mat = materialize(A);
            auto b_mat = materialize(b);
            std::vector<DType> x(n);
            np::internal::cuda::lstsqQr(A_mat.data(), b_mat.data(), x.data(), m, n);
            return NDArrayDynamic<DType>(x);
        }
#endif

        // Unified least‑squares solver with automatic backend selection.
        // Both dimensions m (rows) and n (cols) are considered independently.
        //
        // Selection logic (based on benchmarking):
        //   m ≤ 100  && n ≤ 10   → GELSD (SIMD SVD) — fastest for tiny problems
        //   m ≥ 10000 && n ≥ 500  → CUDA Tikhonov (EVD) — best when both dimensions are large
        //   else with CUDA        → CUDA QR (cuSOLVER gels) — for other large matrices
        //   else                  → CPU Cholesky — fastest CPU solver for most sizes
        template<typename DType, typename DerivedA, typename StorageA, typename DerivedB, typename StorageB>
        inline auto lstsq(const ndarray::internal::NDArrayBase<DType, DerivedA, StorageA> &A,
                          const ndarray::internal::NDArrayBase<DType, DerivedB, StorageB> &b) {
            if (A.ndim() != 2) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 2D array.");
            }
            if (b.ndim() != 1) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Expected 1D array.");
            }
            auto m = A.shape()[0];
            auto n = A.shape()[1];
            if (b.shape()[0] != m) {
                NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid size.");
            }

            // GELSD (SIMD-optimized divide-and-conquer SVD) is fastest for very tiny problems
            // e.g. 100×10: GELSD=102µs vs Cholesky=2929µs
            if (m <= kGelsdMaxRows && n <= kGelsdMaxCols) {
                return lstsq_gelsd(A, b);
            }

#ifdef USE_CUDA
            // CUDA Tikhonov (EVD-based) for large matrices where both dimensions are big
            // e.g. 10000×500: Tikhonov=84845µs vs Cholesky=228562µs, QR=80872µs
            // Tikhonov provides regularization benefits with competitive performance
            if (m >= kCudaTikhonovMinRows && n >= kCudaTikhonovMinCols) {
                return lstsq_tikhonov(A, b);
            }

            // CUDA QR (cuSOLVER gels) for other large matrices where Cholesky's
            // O(cols²) workspace grows
            if (m * n >= kCudaTikhonovMinRows * kCudaTikhonovMinCols) {
                return lstsq_qr(A, b);
            }
#endif
            // Default: CPU Cholesky — fastest CPU solver for the vast majority of problem sizes
            // Beats GELSD on all tested sizes from 1000×50 through 100000×2
            return lstsq_cholesky(A, b);
        }
    }// namespace linalg

}// namespace np

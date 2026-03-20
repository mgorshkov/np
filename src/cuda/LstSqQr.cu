/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov

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
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>

#include <np/internal/cuda/Tikhonov.hpp>
#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/lstsqQr.hpp>
#include <np/Exception.hpp>

// Least Squares with QR decomposition via cuSOLVER's gels
namespace np {
    namespace internal {
        namespace cuda {
            template<typename DType>
            void lstsqQr(const DType* A, const DType* b, DType* x, size_t m, size_t n) {
                // Allocate device memory
                DType *cuda_A;
                checkCudaError(cudaMalloc(&cuda_A, m * n * sizeof(DType)));
                DType *cuda_b;
                checkCudaError(cudaMalloc(&cuda_b, m * sizeof(DType)));
                DType *cuda_x;
                checkCudaError(cudaMalloc(&cuda_x, n * sizeof(DType)));

                // Copy input to device
                checkCudaError(cudaMemcpy(cuda_A, A, m * n * sizeof(DType), cudaMemcpyHostToDevice));
                checkCudaError(cudaMemcpy(cuda_b, b, m * sizeof(DType), cudaMemcpyHostToDevice));

                // Convert row-major to column-major layout
                DType *cuda_A_col;
                checkCudaError(cudaMalloc(&cuda_A_col, m * n * sizeof(DType)));
                dim3 blockDim(16, 16);
                dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
                rowMajorToColMajor<<<gridDim, blockDim>>>(cuda_A, cuda_A_col, static_cast<int>(m), static_cast<int>(n));
                checkCudaError(cudaGetLastError());
                checkCudaError(cudaDeviceSynchronize());

                // Create cuSOLVER and cuBLAS handles
                CusolverWrapper cusolver;
                CublasWrapper cublas;

                // Prepare parameters for gels
                int rows = static_cast<int>(m);
                int cols = static_cast<int>(n);
                int nrhs = 1;  // single right-hand side
                int lda = rows;
                int ldb = (rows > cols) ? rows : cols;  // gels expects ldb >= max(m, n)
                int lddx = (rows > cols) ? rows : cols;  // gels expects lddx >= max(m, n)

                // Allocate device memory for b with size ldb (gels expects ldb >= max(m, n))
                DType *cuda_b_ext;
                checkCudaError(cudaMalloc(&cuda_b_ext, ldb * sizeof(DType)));
                // Copy b into the first m elements
                checkCudaError(cudaMemcpy(cuda_b_ext, cuda_b, m * sizeof(DType), cudaMemcpyDeviceToDevice));

                // Allocate device memory for solution X (size lddx * nrhs)
                // gels requires lddx >= max(m, n); the solution is returned in the first n rows.
                DType *cuda_dX;
                checkCudaError(cudaMalloc(&cuda_dX, static_cast<size_t>(lddx) * nrhs * sizeof(DType)));

                // Query workspace size in bytes using a temporary buffer
                size_t lwork_bytes = 0;
                // Allocate a small temporary workspace for the bufferSize query
                char *d_work_tmp;
                checkCudaError(cudaMalloc(&d_work_tmp, 1024));
                cusolverStatus_t status = cusolverDnGels_bufferSize(
                    cusolver, rows, cols, nrhs, cuda_A_col, lda, cuda_b_ext, ldb,
                    cuda_dX, lddx, d_work_tmp, &lwork_bytes);
                checkCudaError(cudaFree(d_work_tmp));
                if (status != CUSOLVER_STATUS_SUCCESS) {
                    checkCusolverError(status);
                }

                // Allocate workspace in bytes
                char *d_work = nullptr;
                if (lwork_bytes > 0) {
                    checkCudaError(cudaMalloc(&d_work, lwork_bytes));
                }
                int *d_devInfo;
                checkCudaError(cudaMalloc(&d_devInfo, sizeof(int)));
                // Initialize devInfo to 0 on device
                checkCudaError(cudaMemset(d_devInfo, 0, sizeof(int)));
                int host_iter = 0;

                // Solve least squares via QR
                status = cusolverDnGels(
                    cusolver, rows, cols, nrhs, cuda_A_col, lda, cuda_b_ext, ldb,
                    cuda_dX, lddx, d_work, lwork_bytes,
                    &host_iter, d_devInfo);
                if (status != CUSOLVER_STATUS_SUCCESS) {
                    checkCusolverError(status);
                }

                // Check convergence
                int host_devInfo;
                checkCudaError(cudaMemcpy(&host_devInfo, d_devInfo, sizeof(int), cudaMemcpyDeviceToHost));
                if (host_devInfo < 0) {
                    // Negative devInfo indicates an invalid parameter (LAPACK-style error).
                    // This can happen when the matrix is numerically singular or ill-conditioned
                    // (e.g., during IRLS iterations with near-zero weights).
                    // Fall back to Tikhonov regularization instead of throwing.
                    cudaFree(d_devInfo);
                    cudaFree(d_work);
                    cudaFree(cuda_dX);
                    cudaFree(cuda_b_ext);
                    cudaFree(cuda_A_col);
                    cudaFree(cuda_A);
                    cudaFree(cuda_b);
                    cudaFree(cuda_x);

                    DType lambda = 1e-6;
                    lstsqTikhonov(A, b, x, m, n, lambda);
                    return;
                }
                // host_devInfo == 0 indicates success
                // host_devInfo > 0 indicates rank-deficient matrix (the i-th diagonal element of the
                // triangular factor is zero). Fall back to Tikhonov regularization for a minimum-norm solution.
                if (host_devInfo > 0) {
                    // Clean up QR-specific allocations before Tikhonov fallback
                    cudaFree(d_devInfo);
                    cudaFree(d_work);
                    cudaFree(cuda_dX);
                    cudaFree(cuda_b_ext);
                    cudaFree(cuda_A_col);
                    cudaFree(cuda_A);
                    cudaFree(cuda_b);
                    cudaFree(cuda_x);

                    // Fall back to Tikhonov regularization with a small lambda
                    DType lambda = 1e-6;
                    lstsqTikhonov(A, b, x, m, n, lambda);
                    return;
                }

                // Extract solution from dX
                checkCudaError(cudaMemcpy(cuda_x, cuda_dX, n * sizeof(DType), cudaMemcpyDeviceToDevice));

                // Copy result back to host
                checkCudaError(cudaMemcpy(x, cuda_x, n * sizeof(DType), cudaMemcpyDeviceToHost));

                // Cleanup
                cudaFree(d_devInfo);
                cudaFree(d_work);
                cudaFree(cuda_dX);
                cudaFree(cuda_b_ext);
                cudaFree(cuda_A_col);
                cudaFree(cuda_A);
                cudaFree(cuda_b);
                cudaFree(cuda_x);
            }

            // Explicit instantiations for float and double
            template void lstsqQr<float>(const float* A, const float* b, float* x, size_t m, size_t n);
            template void lstsqQr<double>(const double* A, const double* b, double* x, size_t m, size_t n);
        }// namespace cuda
    }// namespace internal
}// namespace np

#endif

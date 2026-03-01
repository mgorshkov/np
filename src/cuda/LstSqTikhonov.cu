/*
C++ numpy-like template-based array implementation

Copyright (c) 2023 Mikhail Gorshkov

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

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <thrust/device_vector.h>

#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>

#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/Tikhonov.cuh>

// Least Squares with Tikhonov Regularized EVD, lambda=1e-6 by default
namespace np {
    namespace internal {
        namespace cuda {
            template<typename DType>
            void lstsqTikhonov(const DType* A, const DType* b, DType* x, size_t m, size_t n, DType lambda) {
                DType *cuda_A;
                checkCudaError(cudaMalloc(&cuda_A, m * n * sizeof(DType)));
                DType *cuda_b;
                checkCudaError(cudaMalloc(&cuda_b, m * sizeof(DType)));
                DType *cuda_x;
                checkCudaError(cudaMalloc(&cuda_x, n * sizeof(DType)));
                checkCudaError(cudaMemcpy(cuda_A, A, m * n * sizeof(DType), cudaMemcpyHostToDevice));
                checkCudaError(cudaMemcpy(cuda_b, b, m * sizeof(DType), cudaMemcpyHostToDevice));

                CublasWrapper cublas;
                CusolverWrapper cusolver;

                // 1: AT A + lambda * I = AT B (Tikhonov)
                thrust::device_vector<DType> cuda_AtA(n * n);
                checkCublasError(cublasGemm<DType>(cublas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    (int)n,
                    (int)n,
                    (int)m,
                    cuda_A,
                    (int)m,
                    cuda_A,
                    (int)m,
                    cuda_AtA.data().get(),
                    (int)n));

                // Tikhonov diagonal
                size_t gridSize = (n + blockSize - 1) / blockSize;
                addConstantToDiagonalKernel<<<gridSize, blockSize>>>(cuda_AtA.data().get(), lambda, n);

                thrust::device_vector<DType> cuda_Atb(n);
                checkCublasError(cublasGemv<DType>(cublas, CUBLAS_OP_T, (int)m, (int)n,
                        cuda_A, (int)m, cuda_b, 1, cuda_Atb.data().get(), 1));

                // Scale A and b if overflow
                DType maxAtb;
                DType scaleFactor = 1.0;
                int maxIdx;
                checkCublasError(cublasIamax<DType>(cublas, n, cuda_Atb.data().get(), 1, &maxIdx));
                checkCudaError(cudaMemcpy(&maxAtb, cuda_Atb.data().get() + maxIdx, sizeof(DType), cudaMemcpyDeviceToHost));

                DType threshold = 1e4;
                if (maxAtb > threshold) {
                    DType scale = sqrtf(threshold / maxAtb);
                    checkCublasError(cublasScal<DType>(cublas, m * n, &scale, cuda_A, 1));
                    checkCublasError(cublasScal<DType>(cublas, m, &scale, cuda_b, 1));
                    // Recalc Atb after scaling
                    checkCublasError(cublasGemv<DType>(cublas, CUBLAS_OP_T, (int)m, (int)n,
                        cuda_A, (int)m, cuda_b, 1, cuda_Atb.data().get(), 1));
                    scaleFactor = 1.0 / (scale * scale);
                }

                // 2: Direct EVD: VΛVᵀ
                thrust::device_vector<DType> evals(n);
                int lwork;
                cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER, (int)n, nullptr, (int)n, nullptr, &lwork);
                thrust::device_vector<DType> work(lwork);
                thrust::device_vector<int> devInfo(1);
                checkCusolverError(cusolverDnSyevd<DType>(cusolver,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    (int)n,
                    cuda_AtA.data().get(),
                    (int)n,
                    evals.data().get(),
                    work.data().get(),
                    (int)lwork,
                    devInfo.data().get()));

                int host_devInfo;
                cudaMemcpy(&host_devInfo, devInfo.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
                if (host_devInfo < 0) {
                    std::ostringstream oss;
                    oss << "Invalid parameter: " << host_devInfo;
                    throw std::runtime_error(oss.str());
                }
                normalizeEigenvectorsKernel<<<gridSize, blockSize>>>(work.data().get(), (int)n);
                checkCudaError(cudaGetLastError());
                checkCudaError(cudaDeviceSynchronize());

                // 3. Spectral filtering: x = V Σ⁺ VT (AT b)
                thrust::device_vector<DType> filtered(n);
                spectralFilterKernel<<<gridSize, blockSize>>>(filtered.data().get(), evals.data().get(), cuda_Atb.data().get(), n);

                // V * filtered
                checkCublasError(cublasGemv<DType>(cublas,
                    CUBLAS_OP_N,
                    (int)n,
                    (int)n,
                    work.data().get(),
                    (int)n,
                    filtered.data().get(),
                    1,
                    cuda_x,
                    1));

                // Unscale solution
                cublasScal<DType>(cublas, n, &scaleFactor, cuda_x, 1);

                checkCudaError(cudaMemcpy(x, cuda_x, n * sizeof(DType), cudaMemcpyDeviceToHost));

                checkCudaError(cudaFree(cuda_A));
                checkCudaError(cudaFree(cuda_b));
                checkCudaError(cudaFree(cuda_x));
            }

            template void lstsqTikhonov(const float* A, const float* b, float* x, size_t m, size_t n, float lambda);
            template void lstsqTikhonov(const double* A, const double* b, double* x, size_t m, size_t n, double lambda);
        }
    }
}

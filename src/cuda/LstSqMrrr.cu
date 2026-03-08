/*
C++ numpy-like template-based array implementation

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

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>

#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/Mrrr.cuh>

// Least Squares with MRRR
namespace np {
    namespace internal {
        namespace cuda {
            // Compute Gershgorin interval for symmetric tridiagonal matrix
            template<typename DType>
            void gershgorinBounds(const DType* d, const DType* e, int n, DType* lower, DType* upper) {
                DType minBound = d[0] - std::abs(e[0]);
                DType maxBound = d[0] + std::abs(e[0]);
                for (int i = 1; i < n - 1; ++i) {
                    DType rowMin = d[i] - std::abs(e[i-1]) - std::abs(e[i]);
                    DType rowMax = d[i] + std::abs(e[i-1]) + std::abs(e[i]);
                    if (rowMin < minBound) minBound = rowMin;
                    if (rowMax > maxBound) maxBound = rowMax;
                }
                if (n > 1) {
                    DType lastMin = d[n-1] - std::abs(e[n-2]);
                    DType lastMax = d[n-1] + std::abs(e[n-2]);
                    if (lastMin < minBound) minBound = lastMin;
                    if (lastMax > maxBound) maxBound = lastMax;
                }
                *lower = minBound;
                *upper = maxBound;
            }

            // Bisection to compute i-th eigenvalue (0-indexed) using Sturm count
            template<typename DType>
            DType bisectionEigenvalue(const DType* d_dev, const DType* e_dev, const DType* l_dev, const DType* delta_dev, int n, int i, DType tol) {
                // Copy d and e to host for bounds
                std::vector<DType> d_host(n);
                std::vector<DType> e_host(n-1);
                cudaMemcpy(d_host.data(), d_dev, n * sizeof(DType), cudaMemcpyDeviceToHost);
                cudaMemcpy(e_host.data(), e_dev, (n-1) * sizeof(DType), cudaMemcpyDeviceToHost);
                DType lower, upper;
                gershgorinBounds(d_host.data(), e_host.data(), n, &lower, &upper);
                // Ensure we have a little padding
                lower -= 1.0;
                upper += 1.0;

                DType lo = lower, hi = upper;
                thrust::device_vector<int> count_dev(1);
                int* count_ptr = count_dev.data().get();
                const int max_iter = 30;
                int iter = 0;
                while (hi - lo > tol && iter < max_iter) {
                    DType mid = (lo + hi) / 2;
                    sturmCountKernel<<<1,1>>>(l_dev, delta_dev, n, mid, count_ptr);
                    cudaDeviceSynchronize();
                    int count;
                    cudaMemcpy(&count, count_ptr, sizeof(int), cudaMemcpyDeviceToHost);
                    if (count <= i) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    ++iter;
                }
                // Convergence failed, return current approximation
                return (lo + hi) / 2;
            }

            template<typename DType>
            void lstsqMrrr(const DType* A, const DType* b, DType* x, size_t m, size_t n) {
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

                // 1: Compute A^T A and A^T b
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

                thrust::device_vector<DType> cuda_Atb(n);
                checkCublasError(cublasGemv<DType>(cublas, CUBLAS_OP_T, (int)m, (int)n,
                        cuda_A, (int)m, cuda_b, 1, cuda_Atb.data().get(), 1));

                // Scale AtA and Atb if overflow to improve numerical stability
                DType maxAtb;
                int maxIdx;
                checkCublasError(cublasIamax<DType>(cublas, n, cuda_Atb.data().get(), 1, &maxIdx));
                if (maxIdx == 0) {
                    throw std::runtime_error("cublasIamax returned zero idx");
                }
                // maxIdx is 1‑based, convert to 0‑based
                size_t idx = maxIdx - 1;
                checkCudaError(cudaMemcpy(&maxAtb, cuda_Atb.data().get() + idx, sizeof(DType), cudaMemcpyDeviceToHost));

                DType threshold = 1e4;
                if (maxAtb > threshold) {
                    DType scale = std::sqrt(threshold / maxAtb);
                    DType scaleSq = scale * scale;
                    // Scale AtA and Atb by scaleSq
                    checkCublasError(cublasScal<DType>(cublas, n * n, &scaleSq, cuda_AtA.data().get(), 1));
                    checkCublasError(cublasScal<DType>(cublas, n, &scaleSq, cuda_Atb.data().get(), 1));
                }

                // 2: Reduce AtA to tridiagonal form (d, e)
                thrust::device_vector<DType> d(n); // diagonal
                thrust::device_vector<DType> e(n-1); // subdiagonal
                thrust::device_vector<DType> tau(n-1); // Householder scalars
                // workspace for sytrd
                int lwork_sytrd;
                cusolverDnSytrd_bufferSize<DType>(cusolver, CUBLAS_FILL_MODE_LOWER, (int)n,
                                            nullptr, (int)n, nullptr, nullptr, nullptr, &lwork_sytrd);
                thrust::device_vector<DType> work_sytrd(lwork_sytrd);
                thrust::device_vector<int> devInfo_sytrd(1);
                // Copy AtA to a temporary matrix because sytrd overwrites it
                thrust::device_vector<DType> AtA_copy = cuda_AtA;
                checkCusolverError(cusolverDnSytrd<DType>(cusolver, CUBLAS_FILL_MODE_LOWER, (int)n,
                                                    AtA_copy.data().get(), (int)n,
                                                    d.data().get(), e.data().get(), tau.data().get(),
                                                    work_sytrd.data().get(), lwork_sytrd,
                                                    devInfo_sytrd.data().get()));
                int info_sytrd;
                cudaMemcpy(&info_sytrd, devInfo_sytrd.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
                if (info_sytrd != 0) {
                    std::ostringstream oss;
                    oss << "sytrd failed with info = " << info_sytrd;
                    throw std::runtime_error(oss.str());
                }

                // Compute LDL^T factorization of tridiagonal matrix
                thrust::device_vector<DType> l(n-1);
                thrust::device_vector<DType> delta(n);
                {
                    std::vector<DType> d_host(n);
                    std::vector<DType> e_host(n-1);
                    checkCudaError(cudaMemcpy(d_host.data(), d.data().get(), n * sizeof(DType), cudaMemcpyDeviceToHost));
                    checkCudaError(cudaMemcpy(e_host.data(), e.data().get(), (n-1) * sizeof(DType), cudaMemcpyDeviceToHost));
                    std::vector<DType> l_host(n-1);
                    std::vector<DType> delta_host(n);
                    delta_host[0] = d_host[0];
                    for (size_t i = 1; i < n; ++i) {
                        l_host[i-1] = e_host[i-1] / delta_host[i-1];
                        delta_host[i] = d_host[i] - l_host[i-1] * e_host[i-1];
                    }
                    checkCudaError(cudaMemcpy(l.data().get(), l_host.data(), (n-1) * sizeof(DType), cudaMemcpyHostToDevice));
                    checkCudaError(cudaMemcpy(delta.data().get(), delta_host.data(), n * sizeof(DType), cudaMemcpyHostToDevice));
                }

                // 3: Compute eigenvalues via parallel bisection (MRRR)
                thrust::device_vector<DType> evals(n);
                {
                    // Compute Gershgorin bounds on host
                    std::vector<DType> d_host(n);
                    std::vector<DType> e_host(n-1);
                    checkCudaError(cudaMemcpy(d_host.data(), d.data().get(), n * sizeof(DType), cudaMemcpyDeviceToHost));
                    checkCudaError(cudaMemcpy(e_host.data(), e.data().get(), (n-1) * sizeof(DType), cudaMemcpyDeviceToHost));
                    DType lower, upper;
                    gershgorinBounds(d_host.data(), e_host.data(), (int)n, &lower, &upper);
                    
                    DType tol = DType(1000.0);
                    const int max_iter = 1;
                    const int blockSize = 256;
                    size_t gridSize = (n + blockSize - 1) / blockSize;
                    bisectionEigenvaluesKernel<<<gridSize, blockSize>>>(
                        l.data().get(), delta.data().get(),
                        (int)n, lower, upper, tol, max_iter,
                        evals.data().get());
                    checkCudaError(cudaDeviceSynchronize());
                }

                // 4: Compute eigenvectors of the original dense matrix using syevd
                // We'll use the original AtA (cuda_AtA) which is still intact.
                thrust::device_vector<DType> evals_syevd(n); // dummy, not used later
                int lwork;
                cusolverDnSyevd_bufferSize<DType>(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER, (int)n, nullptr, (int)n, nullptr, &lwork);
                thrust::device_vector<DType> work(lwork);
                thrust::device_vector<int> devInfo(1);
                // syevd overwrites cuda_AtA with eigenvectors
                checkCusolverError(cusolverDnSyevd<DType>(cusolver,
                    CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER,
                    (int)n,
                    cuda_AtA.data().get(),
                    (int)n,
                    evals_syevd.data().get(),
                    work.data().get(),
                    lwork,
                    devInfo.data().get()));
                int host_devInfo;
                cudaMemcpy(&host_devInfo, devInfo.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
                if (host_devInfo < 0) {
                    std::ostringstream oss;
                    oss << "Invalid parameter: " << host_devInfo;
                    throw std::runtime_error(oss.str());
                }

                // Compute min, max from bisection eigenvalues
                std::vector<DType> evals_host(n);
                cudaMemcpy(evals_host.data(), evals.data().get(), n * sizeof(DType), cudaMemcpyDeviceToHost);
                DType minEval = evals_host[0];
                DType maxEval = evals_host[0];
                for (int i = 1; i < (int)n; ++i) {
                    if (evals_host[i] < minEval) minEval = evals_host[i];
                    if (evals_host[i] > maxEval) maxEval = evals_host[i];
                }
                // Determine filter threshold relative to max eigenvalue
                DType filterThreshold = maxEval * DType(1e-12);
                // 5. Spectral filtering: x = V Σ⁺ V^T * (AT b)
                // Compute y = V^T * (AT b)
                thrust::device_vector<DType> y(n);
                checkCublasError(cublasGemv<DType>(cublas,
                    CUBLAS_OP_T,
                    (int)n,
                    (int)n,
                    cuda_AtA.data().get(),
                    (int)n,
                    cuda_Atb.data().get(),
                    1,
                    y.data().get(),
                    1));

                // Define block size for kernel
                const int blockSize = 256;
                // Regularization parameter (relative to max eigenvalue)
                DType reg = maxEval * DType(2e-4);
                // Filter y by eigenvalues: filtered[i] = y[i] / (evals[i] + reg) (if evals[i] > filterThreshold)
                thrust::device_vector<DType> filtered(n);
                size_t gridSize = (n + blockSize - 1) / blockSize;
                spectralFilterKernel<<<gridSize, blockSize>>>(filtered.data().get(), evals.data().get(), y.data().get(), n, filterThreshold, reg);

                // x = V * filtered
                checkCublasError(cublasGemv<DType>(cublas,
                    CUBLAS_OP_N,
                    (int)n,
                    (int)n,
                    cuda_AtA.data().get(),
                    (int)n,
                    filtered.data().get(),
                    1,
                    cuda_x,
                    1));

                checkCudaError(cudaMemcpy(x, cuda_x, n * sizeof(DType), cudaMemcpyDeviceToHost));


                checkCudaError(cudaFree(cuda_A));
                checkCudaError(cudaFree(cuda_b));
                checkCudaError(cudaFree(cuda_x));
            }

            template void lstsqMrrr(const float* A, const float* b, float* x, size_t m, size_t n);
            template void lstsqMrrr(const double* A, const double* b, double* x, size_t m, size_t n);
        }// namespace cuda
    }// namespace internal
}// namespace np

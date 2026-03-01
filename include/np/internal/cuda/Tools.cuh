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

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <sstream>

namespace np {
    namespace internal {
        namespace cuda {
            const size_t blockSize = 256;

            template<typename DType>
            inline cublasStatus_t cublasGemv(cublasHandle_t handle, cublasOperation_t trans,
                int m, int n,
                const DType *A, int lda,
                const DType *x, int incx,
                DType *y, int incy);

            template<>
            inline cublasStatus_t cublasGemv<float>(cublasHandle_t handle, cublasOperation_t trans,
                int m, int n,
                const float *A, int lda,
                const float *x, int incx,
                float       *y, int incy) {
                    const float alpha1 = 1.0f, beta0 = 0.0f;
                    return cublasSgemv(handle, trans, m, n, &alpha1, A, lda, x, incx, &beta0, y, incy);
                }


            template<>
            inline cublasStatus_t cublasGemv<double>(cublasHandle_t handle, cublasOperation_t trans,
                int m, int n,
                const double *A, int lda,
                const double *x, int incx,
                double       *y, int incy) {
                    const double alpha1 = 1.0, beta0 = 0.0;
                    return cublasDgemv(handle, trans, m, n, &alpha1, A, lda, x, incx, &beta0, y, incy);
                }

            template<typename DType>
            inline cublasStatus_t cublasGemm(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                int m, int n, int k,
                const DType *A, int lda,
                const DType *B, int ldb,
                DType       *C, int ldc);

            template<>
            inline cublasStatus_t cublasGemm<float>(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                int m, int n, int k,
                const float *A, int lda,
                const float *B, int ldb,
                float       *C, int ldc) {
                const float alpha1 = 1.0f, beta0 = 0.0f;

                return cublasSgemm(handle, transa,transb,
                        m, n, k,
                        &alpha1,
                        A, lda,
                        B, ldb,
                        &beta0,
                        C, ldc);
            }

            template<>
            inline cublasStatus_t cublasGemm<double>(cublasHandle_t handle,
                cublasOperation_t transa, cublasOperation_t transb,
                int m, int n, int k,
                const double *A, int lda,
                const double *B, int ldb,
                double       *C, int ldc) {
                const double alpha1 = 1.0, beta0 = 0.0;

                return cublasDgemm(handle, transa,transb,
                    m, n, k,
                    &alpha1,
                    A, lda,
                    B, ldb,
                    &beta0,
                    C, ldc);
            }

            template<typename DType>
            inline cublasStatus_t cublasAsum(cublasHandle_t handle, int n,
                const DType *x, int incx, DType *result);

            template<>
            inline cublasStatus_t cublasAsum<float>(cublasHandle_t handle, int n,
                const float *x, int incx, float *result) {
                return cublasSasum(handle, n, x, incx, result);
            }

            template<>
            inline cublasStatus_t cublasAsum<double>(cublasHandle_t handle, int n,
                const double *x, int incx, double *result) {
                return cublasDasum(handle, n, x, incx, result);
            }

            template<typename DType>
            inline cublasStatus_t cublasIamax(cublasHandle_t handle, int n, const DType *x, int incx, int *result);

            template<>
            inline cublasStatus_t cublasIamax<float>(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
                return cublasIsamax(handle, n, x, incx, result);
            }

            template<>
            inline cublasStatus_t cublasIamax<double>(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
                return cublasIdamax(handle, n, x, incx, result);
            }

            template<typename DType>
            inline cublasStatus_t cublasScal(cublasHandle_t handle, int n,
                const DType *alpha,
                DType *x, int incx);

            template<>
            inline cublasStatus_t cublasScal<float>(cublasHandle_t handle, int n,
                const float *alpha,
                float *x, int incx) {
                return cublasSscal(handle, n, alpha, x, incx);
            }

            template<>
            inline cublasStatus_t cublasScal<double>(cublasHandle_t handle, int n,
                const double *alpha,
                double *x, int incx) {
                return cublasDscal(handle, n, alpha, x, incx);
            }

            template<typename DType>
            inline cusolverStatus_t cusolverDnSyevd(cusolverDnHandle_t handle,
                cusolverEigMode_t jobz,
                cublasFillMode_t uplo,
                int n,
                DType *A,
                int lda,
                DType *W,
                DType *work,
                int lwork,
                int *devInfo);

            template<>
            inline cusolverStatus_t cusolverDnSyevd<float>(cusolverDnHandle_t handle,
                cusolverEigMode_t jobz,
                cublasFillMode_t uplo,
                int n,
                float *A,
                int lda,
                float *W,
                float *work,
                int lwork,
                int *devInfo) {
                return cusolverDnSsyevd(
                    handle,
                    jobz,
                    uplo,
                    n,
                    A,
                    lda,
                    W,
                    work,
                    lwork,
                    devInfo);
            }

            template<>
            inline cusolverStatus_t cusolverDnSyevd<double>(cusolverDnHandle_t handle,
                cusolverEigMode_t jobz,
                cublasFillMode_t uplo,
                int n,
                double *A,
                int lda,
                double *W,
                double *work,
                int lwork,
                int *devInfo) {
                return cusolverDnDsyevd(
                    handle,
                    jobz,
                    uplo,
                    n,
                    A,
                    lda,
                    W,
                    work,
                    lwork,
                    devInfo);
            }

            inline void checkCudaError(cudaError_t err) {
                if (err != cudaSuccess) {
                    std::ostringstream oss;
                    oss << "CUDA error: " << cudaGetErrorString(err);
                    throw std::runtime_error(oss.str());
                }
            }

            inline void checkCublasError(cublasStatus_t status) {
                if (status != CUBLAS_STATUS_SUCCESS) {
                    const char* msg;
                    switch(status) {
                    case CUBLAS_STATUS_NOT_INITIALIZED:
                        msg = "CUBLAS_STATUS_NOT_INITIALIZED";
                        break;
                    case CUBLAS_STATUS_ALLOC_FAILED:
                        msg = "CUBLAS_STATUS_ALLOC_FAILED";
                        break;
                    case CUBLAS_STATUS_INVALID_VALUE:
                        msg = "CUBLAS_STATUS_INVALID_VALUE";
                        break;
                    case CUBLAS_STATUS_ARCH_MISMATCH:
                        msg = "CUBLAS_STATUS_ARCH_MISMATCH";
                        break;
                    case CUBLAS_STATUS_MAPPING_ERROR:
                        msg = "CUBLAS_STATUS_MAPPING_ERROR";
                        break;
                    case CUBLAS_STATUS_EXECUTION_FAILED:
                        msg = "CUBLAS_STATUS_EXECUTION_FAILED";
                        break;
                    case CUBLAS_STATUS_INTERNAL_ERROR:
                        msg = "CUBLAS_STATUS_INTERNAL_ERROR";
                        break;
                    case CUBLAS_STATUS_NOT_SUPPORTED:
                        msg = "CUBLAS_STATUS_NOT_SUPPORTED";
                        break;
                    case CUBLAS_STATUS_LICENSE_ERROR:
                        msg = "CUBLAS_STATUS_LICENSE_ERROR";
                        break;
                    default:
                        msg = "Unknown cuBLAS error";
                        break;
                    }
                    std::ostringstream oss;
                    oss << "cuBLAS error: " << status << " (" << msg << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            inline void checkCusolverError(cusolverStatus_t status) {
                if (status != CUSOLVER_STATUS_SUCCESS) {
                    const char* msg;
                    switch(status) {
                    case CUSOLVER_STATUS_SUCCESS:
                        msg = "SUCCESS";
                        break;
                    case CUSOLVER_STATUS_NOT_INITIALIZED:
                        msg = "CUSOLVER_STATUS_NOT_INITIALIZED";
                        break;
                    case CUSOLVER_STATUS_ALLOC_FAILED:
                        msg = "CUSOLVER_STATUS_ALLOC_FAILED";
                        break;
                    case CUSOLVER_STATUS_INVALID_VALUE:
                        msg = "CUSOLVER_STATUS_INVALID_VALUE";
                        break;
                    case CUSOLVER_STATUS_ARCH_MISMATCH:
                        msg = "CUSOLVER_STATUS_ARCH_MISMATCH";
                        break;
                    case CUSOLVER_STATUS_MAPPING_ERROR:
                        msg = "CUSOLVER_STATUS_MAPPING_ERROR";
                        break;
                    case CUSOLVER_STATUS_EXECUTION_FAILED:
                        msg = "CUSOLVER_STATUS_EXECUTION_FAILED";
                        break;
                    case CUSOLVER_STATUS_INTERNAL_ERROR:
                        msg = "CUSOLVER_STATUS_INTERNAL_ERROR";
                        break;
                    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                        msg = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
                        break;
                    case CUSOLVER_STATUS_NOT_SUPPORTED:
                        msg = "CUSOLVER_STATUS_NOT_SUPPORTED";
                        break;
                    case CUSOLVER_STATUS_ZERO_PIVOT:
                        msg = "CUSOLVER_STATUS_ZERO_PIVOT";
                        break;
                    case CUSOLVER_STATUS_INVALID_LICENSE:
                        msg = "CUSOLVER_STATUS_INVALID_LICENSE";
                        break;
                    case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:
                        msg = "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
                        break;
                    case CUSOLVER_STATUS_IRS_PARAMS_INVALID:
                        msg = "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
                        break;
                    default:
                        msg = "Unknown cuSOLVER error";
                        break;
                    }
                    std::ostringstream oss;
                    oss << "cuSOLVER error: " << status << " (" << msg << ")";
                    throw std::runtime_error(oss.str());
                }
            }

            struct CublasWrapper {
                CublasWrapper() {
                    checkCublasError(cublasCreate(&cublas));
                }
                ~CublasWrapper() {
                    checkCublasError(cublasDestroy(cublas));
                }

                operator cublasHandle_t() {
                    return cublas;
                }

                cublasHandle_t cublas;
            };

            struct CusolverWrapper {
                CusolverWrapper() {
                    checkCusolverError(cusolverDnCreate(&cusolver));
                }
                ~CusolverWrapper() {
                    checkCusolverError(cusolverDnDestroy(cusolver));
                }

                operator cusolverDnHandle_t() {
                    return cusolver;
                }

                cusolverDnHandle_t cusolver;
            };
        }
    }
}


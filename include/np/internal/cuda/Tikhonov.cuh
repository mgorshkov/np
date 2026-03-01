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

template<typename DType>
__global__ void addConstantToDiagonalKernel(DType* AtA, DType lambda, size_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        AtA[i * n + i] += lambda;
    }
}

template<typename DType>
__global__ void normalizeEigenvectorsKernel(DType* V, int n) {
    int col = blockIdx.x;  // One block per column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Threads per column

    if (col >= n) return;

    // Shared memory for column sum-of-squares
    __shared__ double col_norm_sq[1];
    if (threadIdx.y == 0) col_norm_sq[0] = 0.0;
    __syncthreads();

    // Compute column norm² (parallel reduction)
    if (row < n) {
        DType v = V[row * n + col];
        atomicAdd(&col_norm_sq[0], (double)v * v);
    }
    __syncthreads();

    // Normalize column
    if (row < n && col_norm_sq[0] > 1e-20) {
        DType scale = rsqrtf((float)col_norm_sq[0]);
        V[row * n + col] *= scale;
    }
}

template<typename DType>
__global__ void spectralFilterKernel(DType* filtered, const DType* evals, const DType* atb, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        filtered[i] = (evals[i] > 1e-8f) ? atb[i] / evals[i] : 0.0f;
    }
}


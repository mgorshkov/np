/*
C++ numpy-like template-based array implementation

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

#include <cuda_runtime.h>

template<typename DType>
__global__ void spectralFilterKernel(DType *filtered, const DType *evals, const DType *atb, int n, DType threshold = DType(1e-8), DType reg = DType(0)) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        DType denom = evals[i] + reg;
        filtered[i] = (denom > threshold) ? atb[i] / denom : DType(0);
    }
}

// Sturm count: number of eigenvalues less than mu
template<typename DType>
__device__ int sturmCount(int n, const DType *l, const DType *d, DType mu) {
    DType s = -mu;
    int count = 0;
    for (int i = 0; i < n - 1; ++i) {
        DType dpi = d[i] + s;
        if (dpi < 0) ++count;
        DType lpi = (d[i] * l[i]) / dpi;
        s = lpi * l[i] * s - mu;
    }
    DType dpi = d[n - 1] + s;
    if (dpi < 0) ++count;
    return count;
}

// Kernel to compute Sturm counts for a shift
template<typename DType>
__global__ void sturmCountKernel(const DType *l, const DType *d, int n, DType shift, int *count) {
    *count = sturmCount(n, l, d, shift);
}

// Kernel to compute all eigenvalues via parallel bisection
template<typename DType>
__global__ void bisectionEigenvaluesKernel(const DType *l, const DType *d, int n, DType lower, DType upper, DType tol, int max_iter, DType *evals) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    DType lo = lower;
    DType hi = upper;
    for (int iter = 0; iter < max_iter && (hi - lo) > tol; ++iter) {
        DType mid = (lo + hi) / 2;
        int count = sturmCount(n, l, d, mid);
        if (count <= i) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    evals[i] = (lo + hi) / 2;
}

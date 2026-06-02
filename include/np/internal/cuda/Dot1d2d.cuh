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

#ifdef USE_CUDA

#include <cuda_runtime.h>

/// CUDA kernel for 1D·2D dot product: y[j] = sum_i x[i] * W[i * cols + j]
///
/// Each block handles a range of columns. Within each block, threads cooperate
/// to compute the dot product for their assigned column using shared memory
/// for partial sums.
///
/// @param x      1D input vector of size rows (in device memory)
/// @param W      2D row-major matrix of shape (rows, cols) (in device memory)
/// @param rows   Number of rows in W
/// @param cols   Number of columns in W
/// @param result Output vector of size cols (in device memory)
template<typename T>
__global__ void dot1d2dKernel(const T *x, const T *W, std::size_t rows, std::size_t cols, T *result) {
    // Each thread handles one column
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;

    T sum = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        sum += x[i] * W[i * cols + j];
    }
    result[j] = sum;
}

#endif

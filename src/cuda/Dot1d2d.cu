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

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/Dot1d2d.cuh>

namespace np {
    namespace internal {
        namespace cuda {
            void dot1d2d(const float *x, const float *W, std::size_t rows, std::size_t cols, float *result) {
                float *d_x, *d_W, *d_result;

                checkCudaError(cudaMalloc(&d_x, rows * sizeof(float)));
                checkCudaError(cudaMalloc(&d_W, rows * cols * sizeof(float)));
                checkCudaError(cudaMalloc(&d_result, cols * sizeof(float)));

                checkCudaError(cudaMemcpy(d_x, x, rows * sizeof(float), cudaMemcpyHostToDevice));
                checkCudaError(cudaMemcpy(d_W, W, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

                constexpr int blockSize = 256;
                int gridSize = static_cast<int>((cols + blockSize - 1) / blockSize);
                dot1d2dKernel<<<gridSize, blockSize>>>(d_x, d_W, rows, cols, d_result);
                checkCudaError(cudaDeviceSynchronize());

                checkCudaError(cudaMemcpy(result, d_result, cols * sizeof(float), cudaMemcpyDeviceToHost));

                checkCudaError(cudaFree(d_x));
                checkCudaError(cudaFree(d_W));
                checkCudaError(cudaFree(d_result));
            }
        }
    }
}

#endif

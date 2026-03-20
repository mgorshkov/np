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

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <curand.h>

#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/Random.cuh>
#include <np/internal/cuda/Random.hpp>

namespace np {
    namespace internal {
        namespace cuda {
            template<typename DType>
            void randUniform(DType *result, size_t size, DType minValue, DType maxValue, unsigned long long seed) {
                DType *d_result;
                checkCudaError(cudaMalloc(&d_result, size * sizeof(DType)));

                constexpr size_t elementsPerThread = 4;
                size_t blockSize = 256;
                size_t totalThreads = (size + elementsPerThread - 1) / elementsPerThread;
                size_t gridSize = (totalThreads + blockSize - 1) / blockSize;
                if (gridSize == 0) gridSize = 1; // ensure at least one block for small sizes
                randUniformKernel<<<gridSize, blockSize>>>(d_result, size, minValue, maxValue, seed);
                checkCudaError(cudaDeviceSynchronize());

                checkCudaError(cudaMemcpy(result, d_result, size * sizeof(DType), cudaMemcpyDeviceToHost));
                checkCudaError(cudaFree(d_result));
            }

            template void randUniform<float>(float*, size_t, float, float, unsigned long long);
            template void randUniform<double>(double*, size_t, double, double, unsigned long long);
        }
    }
}
#endif

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
#include <curand.h>
#include <curand_kernel.h>
#include <type_traits>

namespace np {
    namespace internal {
        namespace cuda {
            template<typename DType>
            __global__ void randUniformKernel(DType *result, size_t size, DType minValue, DType maxValue, unsigned long long seed) {
                constexpr int elementsPerThread = 4;
                int tid = blockIdx.x * blockDim.x + threadIdx.x;
                int startIdx = tid * elementsPerThread;

                // Vectorized generation for float and double
                if constexpr (std::is_same_v<DType, float>) {
                    curandStatePhilox4_32_10_t state;
                    curand_init(seed, tid, 0, &state);
                    float4 rand4 = curand_uniform4(&state);
                    float rands[4] = {rand4.x, rand4.y, rand4.z, rand4.w};
                    for (int i = 0; i < elementsPerThread; ++i) {
                        int idx = startIdx + i;
                        if (idx >= static_cast<int>(size)) break;
                        result[idx] = rands[i] * (maxValue - minValue) + minValue;
                    }
                } else if constexpr (std::is_same_v<DType, double>) {
                    curandStatePhilox4_32_10_t state;
                    curand_init(seed, tid, 0, &state);
                    double4 rand4 = curand_uniform4_double(&state);
                    double rands[4] = {rand4.x, rand4.y, rand4.z, rand4.w};
                    for (int i = 0; i < elementsPerThread; ++i) {
                        int idx = startIdx + i;
                        if (idx >= static_cast<int>(size)) break;
                        result[idx] = rands[i] * (maxValue - minValue) + minValue;
                    }
                } else {
                    // Fallback for other types (e.g., int) - generate scalar per element
                    curandState state;
                    curand_init(seed, tid, 0, &state);
                    for (int i = 0; i < elementsPerThread; ++i) {
                        int idx = startIdx + i;
                        if (idx >= static_cast<int>(size)) break;
                        DType uniform = static_cast<DType>(curand_uniform(&state));
                        result[idx] = uniform * (maxValue - minValue) + minValue;
                    }
                }
            }

        }// namespace cuda
    }// namespace internal
}// namespace np
#endif

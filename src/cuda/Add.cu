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

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>

#include <np/internal/cuda/Tools.cuh>
#include <np/internal/cuda/Add.cuh>

namespace np {
    namespace internal {
        namespace cuda {
            template<typename DType1, typename DType2, typename DTypeResult>
            void add(const DType1* array1, size_t size1, const DType2* array2, size_t size2, DTypeResult* result, size_t resultSize) {
                DType1 *cuda_array1;
                DType2 *cuda_array2;
                DTypeResult *cuda_result;

                checkCudaError(cudaMalloc(&cuda_array1, size1 * sizeof(DType1)));
                checkCudaError(cudaMalloc(&cuda_array2, size2 * sizeof(DType2)));
                checkCudaError(cudaMalloc(&cuda_result, resultSize * sizeof(DTypeResult)));
                checkCudaError(cudaMemcpy(cuda_array1, array1, size1 * sizeof(DType1), cudaMemcpyHostToDevice));
                checkCudaError(cudaMemcpy(cuda_array2, array2, size2 * sizeof(DType2), cudaMemcpyHostToDevice));

                size_t gridSize = (resultSize + blockSize - 1) / blockSize;
                addKernel<<<gridSize, blockSize>>>(cuda_array1, size1 * sizeof(DType1), cuda_array2,
                    size2 * sizeof(DType2), cuda_result, resultSize * sizeof(DTypeResult));
                checkCudaError(cudaDeviceSynchronize());

                checkCudaError(cudaMemcpy(result, cuda_result, resultSize * sizeof(DTypeResult), cudaMemcpyDeviceToHost));
                checkCudaError(cudaFree(cuda_array1));
                checkCudaError(cudaFree(cuda_array2));
                checkCudaError(cudaFree(cuda_result));
            }

            template void add<long, long, long>(const long*, size_t, const long*, size_t, long*, size_t);
            template void add<float, float, float>(const float*, size_t, const float*, size_t, float*, size_t);
            template void add<double, double, double>(const double*, size_t, const double*, size_t, double*, size_t);
       }
    }
}

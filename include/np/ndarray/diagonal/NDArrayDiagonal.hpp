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

#include <cstddef>
#include <functional>
#include <optional>
#include <ostream>
#include <vector>

#include <np/Axis.hpp>
#include <np/Shape.hpp>

#include <np/ndarray/internal/NDArrayIndex.hpp>
#include <np/ndarray/internal/NDArrayShaped.hpp>

#include <np/ndarray/diagonal/internal/NDArrayDiagonalStorage.hpp>
#include <np/ndarray/diagonal/internal/NDArrayDiagonalStorageStreamIo.hpp>

namespace np {
    namespace ndarray {
        namespace array_diagonal {
            // N-dimensional diagonal array
            template<typename DType, typename Derived, typename Storage, Size Dims>
            class NDArrayDiagonal;

            template<typename DType, typename Derived, typename Storage, Size Dims>
            using NDArrayDiagonalStorage = internal::NDArrayDiagonalStorage<DType, Derived, Storage, Dims>;

            template<typename DType, typename Derived, typename Storage, Size Dims>
            using NDArrayDiagonalBase = ndarray::internal::NDArrayShaped<DType, NDArrayDiagonal<DType, Derived, Storage, Dims>, NDArrayDiagonalStorage<DType, Derived, Storage, Dims>>;

            template<typename DType, typename Derived, typename Storage, Size Dims>
            class NDArrayDiagonal final : public NDArrayDiagonalBase<DType, Derived, Storage, Dims> {
            public:
                // Creating Arrays
                NDArrayDiagonal() noexcept
                    : NDArrayDiagonalBase<DType, Storage, Storage, Dims>{} {
                }

                NDArrayDiagonal(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &v, int k)
                    : NDArrayDiagonalBase<DType, Derived, Storage, Dims>{calcShape(v, k), v, k} {
                }

                NDArrayDiagonal(const NDArrayDiagonal &another) noexcept = default;

                NDArrayDiagonal(NDArrayDiagonal &&another) noexcept = default;

                ~NDArrayDiagonal() noexcept = default;

                NDArrayDiagonal &operator=(const NDArrayDiagonal &another) noexcept {
                    if (&another != this) {
                        NDArrayDiagonalBase<DType, Derived, Storage, Dims>::operator=(another);
                    }
                }

                NDArrayDiagonal &operator=(NDArrayDiagonal &&another) noexcept {
                    NDArrayDiagonalBase<DType, Derived, Storage, Dims>::operator=(another);
                    return *this;
                }

            private:
                static Shape calcShape(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &v, int k) {
                    auto size = static_cast<Size>(std::abs(k));
                    if (Dims != v.ndim()) {
                        throw std::runtime_error("Incorrect Dims");
                    }
                    // empty array
                    if constexpr (Dims == 0) {
                        if (size == 0) {
                            return Shape{};
                        }
                        return Shape{size, size};
                    }
                    // 1D array
                    if constexpr (Dims == 1) {
                        size += v.size();
                        return Shape{size, size};
                    }
                    // 2D array
                    if (k >= 0) {
                        size = std::min(v.shape()[0], v.shape()[1] - k);
                    } else {
                        size = std::min(v.shape()[0] + k, v.shape()[1]);
                    }
                    return Shape{size};
                }
            };

        }// namespace array_diagonal
    }    // namespace ndarray
}// namespace np

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
#include <ostream>

#include <np/Shape.hpp>

#include <np/ndarray/internal/NDArrayShaped.hpp>

#include <np/ndarray/constant/internal/NDArrayConstantStorage.hpp>
#include <np/ndarray/constant/internal/NDArrayConstantStorageStreamIo.hpp>

namespace np {
    namespace ndarray {
        namespace array_constant {
            // N-dimensional constant matrix
            template<typename DType>
            class NDArrayConstant;

            template<typename DType>
            using NDArrayConstantStorage = internal::NDArrayConstantStorage<DType>;

            template<typename DType>
            using NDArrayConstantBase = ndarray::internal::NDArrayShaped<DType, NDArrayConstant<DType>, NDArrayConstantStorage<DType>>;

            template<typename DType>
            class NDArrayConstant final : public NDArrayConstantBase<DType> {
            public:
                // Creating Arrays
                NDArrayConstant() noexcept
                    : NDArrayConstant<DType>{} {
                }

                NDArrayConstant(const Shape &shape, const DType &value) noexcept
                    : NDArrayConstantBase<DType>{shape, shape.calcSizeByShape(), value} {
                }

                NDArrayConstant(const NDArrayConstant &another) noexcept = default;

                NDArrayConstant(NDArrayConstant &&another) noexcept = default;

                ~NDArrayConstant() noexcept = default;

                NDArrayConstant &operator=(const NDArrayConstant &another) noexcept {
                    if (&another != this) {
                        NDArrayConstantBase<DType>::operator=(another);
                    }
                }

                NDArrayConstant &operator=(NDArrayConstant &&another) noexcept {
                    NDArrayConstantBase<DType>::operator=(another);
                    return *this;
                }
            };

        }// namespace array_constant
    }    // namespace ndarray
}// namespace np

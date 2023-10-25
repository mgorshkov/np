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

#include <np/ndarray/diagonal/internal/NDArrayIdentityStorage.hpp>
#include <np/ndarray/diagonal/internal/NDArrayIdentityStorageStreamIo.hpp>

namespace np {
    namespace ndarray {
        namespace array_diagonal {
            // N-dimensional identity matrix
            template<typename DType>
            class NDArrayIdentity;

            template<typename DType>
            using NDArrayIdentityStorage = internal::NDArrayIdentityStorage<DType>;

            template<typename DType>
            using NDArrayIdentityBase = ndarray::internal::NDArrayShaped<DType, NDArrayIdentity<DType>, NDArrayIdentityStorage<DType>>;

            template<typename DType>
            class NDArrayIdentity final : public NDArrayIdentityBase<DType> {
            public:
                // Creating Arrays
                NDArrayIdentity() noexcept
                    : NDArrayIdentity<DType>{} {
                }

                explicit NDArrayIdentity(Size size) noexcept
                    : NDArrayIdentityBase<DType>{Shape{size, size}, size} {
                }

                NDArrayIdentity(const NDArrayIdentity &another) noexcept = default;

                NDArrayIdentity(NDArrayIdentity &&another) noexcept = default;

                ~NDArrayIdentity() noexcept = default;

                NDArrayIdentity &operator=(const NDArrayIdentity &another) noexcept {
                    if (&another != this) {
                        NDArrayIdentityBase<DType>::operator=(another);
                    }
                }

                NDArrayIdentity &operator=(NDArrayIdentity &&another) noexcept {
                    NDArrayIdentityBase<DType>::operator=(another);
                    return *this;
                }
            };

        }// namespace array_diagonal
    }    // namespace ndarray
}// namespace np

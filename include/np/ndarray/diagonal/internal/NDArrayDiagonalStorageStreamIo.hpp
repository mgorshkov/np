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
#include <iomanip>
#include <iostream>

#include <np/ndarray/diagonal/internal/NDArrayDiagonalStorage.hpp>

namespace np {
    namespace ndarray {
        namespace array_diagonal {
            namespace internal {
                template<typename DType, typename Storage, typename Parent, Size Dims>
                std::ostream &operator<<(std::ostream &stream, const NDArrayDiagonalStorage<DType, Storage, Parent, Dims> &array);

                template<typename Storage, typename Parent, Size Dims>
                inline std::ostream &
                operator<<(std::ostream &stream, const NDArrayDiagonalStorage<std::wstring, Storage, Parent, Dims> &array);

                template<typename Storage, typename Parent, Size Dims>
                inline std::wostream &
                operator<<(std::wostream &stream, const NDArrayDiagonalStorage<std::wstring, Storage, Parent, Dims> &array);
            }// namespace internal
        }    // namespace array_diagonal
    }        // namespace ndarray
}// namespace np

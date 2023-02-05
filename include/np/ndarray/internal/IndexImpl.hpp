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
#include <utility>
#include <vector>

#include <np/DType.hpp>
#include <np/Shape.hpp>
#include <np/ndarray/internal/Index.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            Index<DType, Parent, Storage, ParentStorage>::Index(const Index<DType, Parent, Storage, ParentStorage> &another)
                : NDArrayShaped<DType, Parent, IndexStorage<DType, Parent, ParentStorage>>{another} {
            }

            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            Index<DType, Parent, Storage, ParentStorage>::Index(Index<DType, Parent, Storage, ParentStorage> &&another) noexcept
                : NDArrayShaped<DType, Parent, IndexStorage<DType, Parent, ParentStorage>>{std::move(another)} {
            }

            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            Index<DType, Parent, Storage, ParentStorage>::Index(NDArrayBase<DType, Parent, ParentStorage> *parent, Size indexStart, Shape shape)
                : NDArrayShaped<DType, Parent, IndexStorage<DType, Parent, ParentStorage>>{std::move(shape), parent, indexStart, indexStart + shape.calcSizeByShape()} {
            }

            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            Index<DType, Parent, Storage, ParentStorage>::Index(NDArrayBase<DType, Parent, ParentStorage> *parent, const std::vector<Size> &indices, Shape shape)
                : NDArrayShaped<DType, Parent, IndexStorage<DType, Parent, ParentStorage>>{std::move(shape), parent, indices} {
            }
        }// namespace internal
    }    // namespace ndarray
}// namespace np
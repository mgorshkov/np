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
#include <memory>
#include <utility>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage, typename ParentStorage, typename Parent>
            class NDArrayIndex final : public NDArrayBase<DType, Derived, Storage> {
            public:
                NDArrayIndex() = default;

                NDArrayIndex(const NDArrayIndex &another)
                    : NDArrayBase<DType, Derived, Storage>{another.getStorage()} {
                }

                NDArrayIndex(NDArrayIndex &&another) noexcept
                    : NDArrayBase<DType, Derived, Storage>{std::move(another.getStorage())} {
                }

                NDArrayIndex &operator=(const NDArrayIndex &another) = default;
                NDArrayIndex &operator=(NDArrayIndex &&another) noexcept = default;

                NDArrayIndex(Parent parent, const IndicesType<DType> &indices)
                    : NDArrayBase<DType, Derived, NDArrayIndexStorage<DType, ParentStorage, Parent>>{parent, indices} {
                }
            };

            template<typename DType, typename Derived, typename ParentStorage, typename ParentType>
            using IndexParentPtr = std::shared_ptr<IndexParent<DType, Derived, ParentStorage, ParentType>>;

            template<typename DType, typename Derived, typename ParentStorage, typename Parent>
            using IndexParent = NDArrayIndex<DType, Derived, NDArrayIndexStorage<DType, ParentStorage, Parent>, ParentStorage, Parent>;
        }// namespace internal
    }    // namespace ndarray
}// namespace np

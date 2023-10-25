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

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>


namespace np {
    namespace ndarray {
        namespace internal {

            template<typename DType, typename Derived, typename Storage>
            class NDArrayShaped : public NDArrayBase<DType, Derived, Storage> {
            public:
                NDArrayShaped() = default;

                NDArrayShaped(const NDArrayShaped &another) = default;

                NDArrayShaped(NDArrayShaped &&another) noexcept = default;

                NDArrayShaped(const std::vector<DType> &vector, Shape shape);

                NDArrayShaped(DType *data, Shape shape);

                template<typename... Args>
                explicit NDArrayShaped(Shape shape, Args &&...args);

                template<typename... Args>
                explicit NDArrayShaped(bool isColumnVector, Args &&...args);

                NDArrayShaped &operator=(const NDArrayShaped &another);

                NDArrayShaped &operator=(NDArrayShaped &&another) noexcept;

                using Base = NDArrayBase<DType, Derived, Storage>;
                using BasePtrBool = NDArrayBasePtr<bool_, Derived, Storage>;
                using BasePtr = NDArrayBasePtr<DType, Derived, Storage>;
                using BaseConstPtr = NDArrayBaseConstPtr<DType, Derived, Storage>;

                // Array dimensions
                [[nodiscard]] Shape shape() const override;
                void setShape(const Shape &shape) override;

                using value_type = DType;// for std::back_inserter
                void push_back(const value_type &value) {
                    Base::push(value);
                    if (m_shape.empty()) {
                        m_shape = Shape{1};
                    } else {
                        m_shape.flatten();
                        ++m_shape[0];
                    }
                }

            protected:
                Shape m_shape;
            };

        }// namespace internal
    }    // namespace ndarray
}// namespace np
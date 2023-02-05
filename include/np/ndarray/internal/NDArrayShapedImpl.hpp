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
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>
#include <np/ndarray/internal/NDArrayShaped.hpp>

namespace np {
    namespace ndarray {
        namespace internal {

            template<typename DType, typename Derived, typename Storage>
            NDArrayShaped<DType, Derived, Storage>::NDArrayShaped(const std::vector<DType> &vector, Shape shape)
                : Base{vector, shape.calcSizeByShape()}, m_shape{std::move(shape)} {
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename... Args>
            NDArrayShaped<DType, Derived, Storage>::NDArrayShaped(Shape shape, Args &&...args)
                : Base{std::forward<Args>(args)...}, m_shape{std::move(shape)} {
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename... Args>
            NDArrayShaped<DType, Derived, Storage>::NDArrayShaped(bool isColumnVector, Args &&...args)
                : Base{std::forward<Args>(args)...} {
                if (isColumnVector) {
                    m_shape.singleElement();
                }
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayShaped<DType, Derived, Storage> &NDArrayShaped<DType, Derived, Storage>::operator=(const NDArrayShaped &another) {
                if (this != &another) {
                    m_shape = another.m_shape;
                    Base::operator=(another);
                }
                return *this;
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayShaped<DType, Derived, Storage> &NDArrayShaped<DType, Derived, Storage>::operator=(NDArrayShaped &&another) noexcept {
                if (this != &another) {
                    m_shape = another.m_shape;
                    another.m_shape.clear();
                    Base::operator=(another);
                }
                return *this;
            }

            template<typename DType, typename Derived, typename Storage>
            Shape NDArrayShaped<DType, Derived, Storage>::shape() const {
                return m_shape;
            }

            template<typename DType, typename Derived, typename Storage>
            void NDArrayShaped<DType, Derived, Storage>::setShape(Shape shape) {
                m_shape = std::move(shape);
            }

            template<typename DType, typename Derived, typename Storage>
            Size NDArrayShaped<DType, Derived, Storage>::len() const {
                return m_shape.empty() ? 0 : m_shape[0];
            }

            template<typename DType, typename Derived, typename Storage>
            Size NDArrayShaped<DType, Derived, Storage>::ndim() const {
                return m_shape.size();
            }

            template<typename DType, typename Derived, typename Storage>
            Size NDArrayShaped<DType, Derived, Storage>::size() const {
                return m_shape.calcSizeByShape();
            }

        }// namespace internal
    }    // namespace ndarray
}// namespace np
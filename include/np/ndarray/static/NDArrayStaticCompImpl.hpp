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

#include <np/DType.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            template<typename DType, Size SizeT>
            NDArrayStatic<bool_, SizeT> NDArrayStatic<DType, SizeT>::operator==(const NDArrayStatic<DType, SizeT> &array) const {
                NDArrayStatic<bool_, SizeT> result{NDArrayStaticBase<DType, SizeT>::shape()};
                for (Size i = 0; i < NDArrayStaticBase<DType, SizeT>::size(); ++i) {
                    const auto equals = NDArrayStaticBase<DType, SizeT>::get(i) == array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, Size SizeT>
            NDArrayStatic<bool_, SizeT> NDArrayStatic<DType, SizeT>::operator<(const NDArrayStatic<DType, SizeT> &array) const {
                NDArrayStatic<bool_, SizeT> result{NDArrayStaticBase<DType, SizeT>::shape()};
                for (Size i = 0; i < NDArrayStaticBase<DType, SizeT>::size(); ++i) {
                    auto equals = NDArrayStaticBase<DType, SizeT>::get(i) < array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, Size SizeT>
            NDArrayStatic<bool_, SizeT> NDArrayStatic<DType, SizeT>::operator>(const NDArrayStatic<DType, SizeT> &array) const {
                NDArrayStatic<bool_, SizeT> result{NDArrayStaticBase<DType, SizeT>::shape()};
                for (Size i = 0; i < NDArrayStaticBase<DType, SizeT>::size(); ++i) {
                    auto equals = NDArrayStaticBase<DType, SizeT>::get(i) > array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

        }// namespace array_static
    }    // namespace ndarray
}// namespace np
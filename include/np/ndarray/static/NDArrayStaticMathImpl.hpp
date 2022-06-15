/*
C++ numpy-like template-based array implementation

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/ndarray/static/NDArrayStaticDecl.hpp>


namespace np {
    namespace ndarray {
        namespace array_static {
            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::operator+(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return add(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::add(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) + array.get(i));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::operator-(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return subtract(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::subtract(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) - array.get(i));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::operator*(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return multiply(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::multiply(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) * array.get(i));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::operator/(
                    const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                return divide(array);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::divide(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i) / array.get(i));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::exp(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < SizeT; ++i) {
                    result.set(i, ReducedType{m_ArrayImpl[i]}.exp(ReducedType{array[i]}));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::sqrt() const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < SizeT; ++i) {
                    result.set(i, sqrt(ReducedType{m_ArrayImpl[i]}));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::sin() const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < SizeT; ++i) {
                    result.set(i, sin(ReducedType{m_ArrayImpl[i]}));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::cos() const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < SizeT; ++i) {
                    result.set(i, cos(ReducedType{m_ArrayImpl[i]}));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayStatic<DType, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::log() const {
                NDArrayStatic<DType, SizeT, SizeTs...> result;
                for (Size i = 0; i < SizeT; ++i) {
                    result.set(i, log(ReducedType{m_ArrayImpl[i]}));
                }
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType
            NDArrayStatic<DType, SizeT, SizeTs...>::dot(const NDArrayStatic<DType, SizeT, SizeTs...> &array) const {
                if (shape().size() != 1 || array.shape().size() != 1 || shape() != array.shape()) {
                    throw std::runtime_error("Shapes are different or arguments are not 1D arrays");
                }
                DType result{0};
                for (Size i = 0; i < size(); ++i) {
                    result += get(i) * array.get(i);
                }
                return result;
            }
        }// namespace array_static
    }// namespace ndarray
}// namespace np

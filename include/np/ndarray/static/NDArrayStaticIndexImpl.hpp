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

#include <type_traits>

#include <np/ndarray/static/NDArrayStaticDecl.hpp>


namespace np {
    namespace ndarray {
        namespace array_static {
            // Subsetting
            template<typename DType, Size SizeT, Size... SizeTs>
            inline void
            set(NDArrayStatic<DType, SizeT, SizeTs...> &array, Size i,
                const typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType &data) {
                array.m_ArrayImpl.set(i, data.m_ArrayImpl);
            }

            // Select an element at an index
            // a[2]
            template<typename DType, Size SizeT, Size... SizeTs>
            inline typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::operator[](Size i) const {
                return ReducedType{m_ArrayImpl[i]};
            }

            // a[1,2]                               Select the element at row and column (same as a[1][2])
            // Slicing
            // a[0:2]                               Select items at index 0 and 1
            // b[0:2,1]                             Select items at rows 0 and 1 in column 1
            // b[:1]                                Select all items at row 0 (equivalent to b[0:1,:])
            // c[1,...]                             Same as [1,:,:]
            // a[::-1]                              Reversed array
            // Boolean indexing
            // a[a < 2]                             Select elements from a less than 2
            // Fancy indexing
            // b[[1, 0, 1, 0], [0, 1, 2, 0]]        Select elements (1,0),(0,1),(1,2) and (0,0)
            // b[[1, 0, 1, 0]][: ,[0, 1, 2, 0]]     Select a subset of the matrix’s rows and columns
            /* TODO
            template <typename DType, Size SizeT, Size... SizeTs>
            inline ReducedType&
            NDArrayStatic<DType, SizeT, SizeTs...>::operator [] (const std::string& i) {
                return ReducedType{m_ArrayImpl[i]};
            }

            template <typename DType, Size SizeT, Size... SizeTs>
            inline ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::operator [] (const std::string& i) const {
                return ReducedType{m_ArrayImpl[i]};
            }
            */

            template<typename DType, Size SizeT, Size... SizeTs>
            inline typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::at(Size i) const {
                return ReducedType(m_ArrayImpl[i]);
            }

            /* TODO
            template <typename DType, Size SizeT, Size... SizeTs>
            inline ReducedType&
            NDArrayStatic<DType, SizeT, SizeTs...>::at(const std::string& i) {
                return ReducedType{m_ArrayImpl[i]};
            }

            template <typename DType, Size SizeT, Size... SizeTs>
            inline ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::at(const std::string& i) const {
                return ReducedType{m_ArrayImpl[i]};
            }
            */

            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::get(std::size_t i) const {
                return m_ArrayImpl.get(i);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::set(std::size_t i, const DType &value) {
                return m_ArrayImpl.set(i, value);
            }
        }// namespace array_static
    }    // namespace ndarray
}// namespace np

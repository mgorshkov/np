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
#include <np/ndarray/static/internal/Tools.hpp>


namespace np {
    namespace ndarray {
        namespace array_static {
            // Array dimensions
            template<typename DType, Size SizeT, Size... SizeTs>
            Shape NDArrayStatic<DType, SizeT, SizeTs...>::shape() const {
                return {SizeT, SizeTs...};
            }

            // Length of array
            template<typename DType, Size SizeT, Size... SizeTs>
            Size NDArrayStatic<DType, SizeT, SizeTs...>::len() const {
                return SizeT;
            }

            // Number of array dimensions
            template<typename DType, Size SizeT, Size... SizeTs>
            Size NDArrayStatic<DType, SizeT, SizeTs...>::ndim() const {
                return 1 + sizeof...(SizeTs);
            }

            // Number of array elements
            template<typename DType, Size SizeT, Size... SizeTs>
            Size NDArrayStatic<DType, SizeT, SizeTs...>::size() const {
                size_t result{SizeT};
                if constexpr ((sizeof...(SizeTs)) != 0)
                    for (auto i: {SizeTs...}) result *= i;
                return result;
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline constexpr DType NDArrayStatic<DType, SizeT, SizeTs...>::dtype() const {
                return DType{};
            }

            // Convert an array to a different type
            template<typename DType, Size SizeT, Size... SizeTs>
            template<typename DTypeNew>
            NDArrayStatic<DTypeNew, SizeT, SizeTs...>
            NDArrayStatic<DType, SizeT, SizeTs...>::astype() const {
                return NDArrayStatic<DTypeNew, SizeT, SizeTs...>{};
            }
        }
    }
}

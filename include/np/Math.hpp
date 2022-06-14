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

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    using ndarray::array_dynamic::NDArrayDynamic;
    using ndarray::array_static::NDArrayStatic;

    //////////////////////////////////////////////////////////////
    /// \brief Arrays sum
    ///
    /// Calculate the array-wise element-by-element sum of the arrays.
    ///
    /// \param array1 array to sum
    /// \param array2 array to sum
    ///
    /// \return The sum of the arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> add(const Array<DType, SizeTs, Sizes...> &array1, const Array<DType, SizeTs, Sizes...> &array2) {
        return array1.add(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Arrays subtraction
    ///
    /// Calculate the array-wise element-by-element difference of the arrays.
    ///
    /// \param array1 array to subtract from
    /// \param array2 array to subtract
    ///
    /// \return The difference of the arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> subtract(const Array<DType, SizeTs, Sizes...> &array1, const Array<DType, SizeTs, Sizes...> &array2) {
        return array1.subtract(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Arrays multiplication
    ///
    /// Calculate the array-wise element-by-element product of the arrays.
    ///
    /// \param array1 array to multiply
    /// \param array2 array to multiply
    ///
    /// \return The product of the arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> multiply(const Array<DType, SizeTs, Sizes...> &array1, const Array<DType, SizeTs, Sizes...> &array2) {
        return array1.multiply(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Arrays division
    ///
    /// Calculate the array-wise element-by-element ratio of the arrays.
    ///
    /// \param array1 array to multiply
    /// \param array2 array to multiply
    ///
    /// \warning Division by zero is not handled
    ///
    /// \return The ratio of the arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> divide(const Array<DType, SizeTs, Sizes...> &array1, const Array<DType, SizeTs, Sizes...> &array2) {
        return array1.divide(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the exponent of the arrays
    ///
    /// Find array1 ^ array2 element by element.
    ///
    /// \param array1 array to exp
    /// \param array2 array to exp
    ///
    /// \return The exponent of an array1
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> exp(const Array<DType, SizeTs, Sizes...> &array1, const Array<DType, SizeTs, Sizes...> &array2) {
        return array1.exp(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the square root of the array
    ///
    /// Calculate array-wise sqrt element by element.
    ///
    /// \param array array to calculate sqrt
    ///
    /// \return The square root of an array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> sqrt(const Array<DType, SizeTs, Sizes...> &array) {
        return array.sqrt();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the sine of the array
    ///
    /// Find array-wise sine element by element.
    ///
    /// \param array array to calculate sine
    ///
    /// \return The sine of an array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> sin(const Array<DType, SizeTs, Sizes...> &array) {
        return array.sin();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the cosine of the array
    ///
    /// Find array-wise cosine element by element.
    ///
    /// \param array array to calculate cosine
    ///
    /// \return The cosine of an array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> cos(const Array<DType, SizeTs, Sizes...> &array) {
        return array.cos();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the log of the array
    ///
    /// Find array-wise log element by element.
    ///
    /// \param array array to calculate log
    ///
    /// \return The log of an array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeTs = SIZE_DEFAULT, Size... Sizes>
    inline Array<DType, SizeTs, Sizes...> log(const Array<DType, SizeTs, Sizes...> &array) {
        return array.log();
    }
}// namespace np

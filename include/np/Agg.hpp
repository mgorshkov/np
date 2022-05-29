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

#include <optional>

#include <np/Constants.hpp>
#include <np/DType.hpp>
#include <np/Axis.hpp>
#include <np/Array.hpp>

#include <np/ndarray/static/NDArrayStatic.hpp>
#include <np/ndarray/dynamic/NDArrayDynamic.hpp>

////////////////////////////////////////////////////////////
/// \brief Aggregate functions
///
////////////////////////////////////////////////////////////
namespace np {
    //////////////////////////////////////////////////////////////
    /// \brief Sum of array elements
    ///
    /// This function sums up all elements of an array.
    ///
    /// \param array An array to calculate the sum
    ///
    /// \return Sum of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto sum(const Array<DType, SizeT, SizeTs...> &array) {
        return array.sum();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Minimum of array elements
    ///
    /// This function finds a minimum among all elements of an array.
    ///
    /// \warning This function is currently implemented for dynamic arrays only
    ///
    /// \param array An array to calculate the minimum
    ///
    /// \return Minimum of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto min(const Array<DType, SizeT, SizeTs...> &array) {
        return array.min();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Maximum of array elements
    ///
    /// This function finds a maximum among all elements of an array.
    ///
    /// \param array An array to calculate the maximum
    ///
    /// \return Maximum of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto max(const Array<DType, SizeT, SizeTs...> &array) {
        return array.max();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Cumulative sum of the elements
    ///
    /// This function calculates a cumulative sum among all elements of an array.
    ///
    /// \param array An array to calculate the cumulative sum
    ///
    /// \return Cumulative sum of array elements as a 1D array
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto cumsum(const Array<DType, SizeT, SizeTs...> &array) {
        return array.cumsum();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Mean of the elements
    ///
    /// This function calculates mean among all elements of an array.
    ///
    /// \param array An array to calculate the mean
    ///
    /// \return Mean of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto mean(const Array<DType, SizeT, SizeTs...> &array) {
        return array.mean();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Median of the elements
    ///
    /// This function calculates median among all elements of an array.
    /// Given an array A of length N, the median of A is the middle value of a sorted copy of A, A_sorted[(N-1)/2],
    /// when N is odd, and the average of the two middle values of A_sorted when N is even.
    ///
    /// \param array An array to calculate the median
    ///
    /// \return Median of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto median(const Array<DType, SizeT, SizeTs...> &array) {
        return array.median();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Calculate a covariance matrix
    ///
    /// Covariance indicates the level to which two variables vary together.
    ///
    /// \param array An array to calculate the covariance
    ///
    /// \return Covariance of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto cov(const Array<DType, SizeT, SizeTs...> &array) {
        return array.cov();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Return Pearson product-moment correlation coefficients
    ///
    /// The relationship between the correlation coefficient matrix, R, and the covariance matrix, C, is
    /// R_ij = C_ij / sqrt(C_ii * C_jj)
    ///
    /// \param array An array to calculate the correlation coefficients
    ///
    /// \return Correlation coefficients of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto corrcoef(const Array<DType, SizeT, SizeTs...> &array) {
        return array.corrcoef();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Compute the standard deviation along the specified axis
    ///
    /// Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
    /// The standard deviation is computed for the flattened array
    ///
    /// \param array An array to calculate the standard deviation
    ///
    /// \return Standard deviation of array elements
    ///
    //////////////////////////////////////////////////////////////
    template <typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    inline auto std_(const Array<DType, SizeT, SizeTs...> &array) {
        return array.std_();
    }
}


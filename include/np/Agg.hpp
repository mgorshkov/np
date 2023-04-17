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

#include <optional>

#include <np/Array.hpp>
#include <np/Axis.hpp>
#include <np/Constants.hpp>
#include <np/DType.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> sumImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> sumImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.sum();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto sum(const Array<DType, SizeT> &array) {
        float_ result{};
        sumImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Sum of array elements, treating NaNs as zeros
    ///
    /// This function sums up all elements of an array, treating NaNs as zeros.
    ///
    /// \param array An array to calculate the sum
    ///
    /// \return Sum of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nansumImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nansumImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.nansum();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nansum(const Array<DType, SizeT> &array) {
        float_ result{};
        nansumImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Minimum of array elements
    ///
    /// This function finds a minimum among all elements of an array.
    ///
    /// \param array An array to calculate the minimum
    ///
    /// \return Minimum of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> minImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> minImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.min();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto min(const Array<DType, SizeT> &array) {
        float_ result{};
        minImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> maxImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> maxImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.max();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto max(const Array<DType, SizeT> &array) {
        float_ result{};
        maxImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> cumsumImpl(const Array<DType, SizeT> &, Array<DType> &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> cumsumImpl(const Array<DType, SizeT> &array, Array<DType> &result) {
        result = array.cumsum();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto cumsum(const Array<DType, SizeT> &array) {
        Array<DType> result{};
        cumsumImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Cumulative sum of the elements treating NaNs as zeros
    ///
    /// This function calculates a cumulative sum among all elements of an array treating NaNs as zeros.
    ///
    /// \param array An array to calculate the cumulative sum
    ///
    /// \return Cumulative sum of array elements as a 1D array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nancumsumImpl(const Array<DType, SizeT> &, Array<DType> &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nancumsumImpl(const Array<DType, SizeT> &array, Array<DType> &result) {
        result = array.nancumsum();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nancumsum(const Array<DType, SizeT> &array) {
        Array<DType> result{};
        nancumsumImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> meanImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> meanImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.mean();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto mean(const Array<DType, SizeT> &array) {
        float_ result{};
        meanImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Mean of the elements ignoring NaNs
    ///
    /// This function calculates mean among all elements of an array except NaNs.
    ///
    /// \param array An array to calculate the mean
    ///
    /// \return Mean of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nanmeanImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nanmeanImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.nanmean();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nanmean(const Array<DType, SizeT> &array) {
        float_ result{};
        nanmeanImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> medianImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> medianImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.median();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto median(const Array<DType, SizeT> &array) {
        float_ result{};
        medianImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Median of the elements ignoring NaNs
    ///
    /// This function calculates median among all elements of an array except NaNs.
    /// Given an array A of length N, the median of A is the middle value of a sorted copy of A, A_sorted[(N-1)/2],
    /// when N is odd, and the average of the two middle values of A_sorted when N is even.
    ///
    /// \param array An array to calculate the median
    ///
    /// \return Median of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nanmedianImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nanmedianImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.nanmedian();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nanmedian(const Array<DType, SizeT> &array) {
        float_ result{};
        nanmedianImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> covImpl(const Array<DType, SizeT> &, Array<float_> &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> covImpl(const Array<DType, SizeT> &array, Array<float_> &result) {
        result = array.cov();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto cov(const Array<DType, SizeT> &array) {
        Array<float_> result{};
        covImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> corrcoefImpl(const Array<DType, SizeT> &, Array<float_> &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> corrcoefImpl(const Array<DType, SizeT> &array, Array<float_> &result) {
        result = array.corrcoef();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto corrcoef(const Array<DType, SizeT> &array) {
        Array<float_> result{};
        corrcoefImpl<DType, SizeT>(array, result);
        return result;
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
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> stdImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> stdImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.std_();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto std_(const Array<DType, SizeT> &array) {
        float_ result{};
        stdImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Compute the standard deviation along the specified axis except NaNs
    ///
    /// Returns the standard deviation, a measure of the spread of a distribution, of the array elements.
    /// The standard deviation is computed for the flattened array
    ///
    /// \param array An array to calculate the standard deviation
    ///
    /// \return Standard deviation of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nanstdImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nanstdImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.nanstd();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nanstd(const Array<DType, SizeT> &array) {
        float_ result{};
        nanstdImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Compute the variance along the specified axis
    ///
    /// Returns the variance of the array elements, a measure of the spread of a distribution.
    /// The variance is computed for the flattened array
    ///
    /// \param array An array to calculate the variance
    ///
    /// \return Variance of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> varImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> varImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.var();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto var(const Array<DType, SizeT> &array) {
        float_ result{};
        varImpl<DType, SizeT>(array, result);
        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Compute the variance along the specified axis except NaNs
    ///
    /// Returns the variance of the array elements, a measure of the spread of a distribution.
    /// The variance is computed for the flattened array
    ///
    /// \param array An array to calculate the variance
    ///
    /// \return Variance of array elements
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<!std::is_arithmetic_v<DType>> nanvarImpl(const Array<DType, SizeT> &, float_ &) {
        throw std::runtime_error("Invalid argument");
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    std::enable_if_t<std::is_arithmetic_v<DType>> nanvarImpl(const Array<DType, SizeT> &array, float_ &result) {
        result = array.nanvar();
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline auto nanvar(const Array<DType, SizeT> &array) {
        float_ result{};
        nanvarImpl<DType, SizeT>(array, result);
        return result;
    }

}// namespace np

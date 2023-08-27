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
    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto add(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.add(array2);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator+(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.add(array2);
    }

    template<Arithmetic DType1, typename Derived2, typename Storage2, Arithmetic DType2>
    inline auto operator+(const DType1 &value, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array) {
        return array.add(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator+(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.add(value);
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
    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto subtract(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.subtract(array2);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator-(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.subtract(array2);
    }

    template<Arithmetic DType1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator-(const DType1 &value, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array) {
        return array.subtract(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator-(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.subtract(value);
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
    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto multiply(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.multiply(array2);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator*(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.multiply(array2);
    }

    template<Arithmetic DType1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator*(const DType1 &value, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array) {
        return array.multiply(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator*(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.multiply(value);
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
    template<Arithmetic DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    inline auto divide(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
        return array1.divide(array2);
    }

    template<Arithmetic DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    inline auto operator/(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
        return array1.divide(array2);
    }

    template<Arithmetic DType1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator/(const DType1 &value, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array) {
        return array.divide(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator/(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.divide(value);
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
    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto exp(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto sqrt(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto sin(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto cos(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto log(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.log();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Find the abs of the array
    ///
    /// Find array-wise absolute value element by element.
    ///
    /// \param array array to calculate abs
    ///
    /// \return The absolute value of an array
    ///
    //////////////////////////////////////////////////////////////
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto abs(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.abs();
    }

    //////////////////////////////////////////////////////////////
    /// \brief One-dimensional linear interpolation for monotonically increasing sample points.
    ///
    /// Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.
    ///
    /// \param x The x-coordinates at which to evaluate the interpolated values.
    /// \param xp 1-D sequence of floats
    /// The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted
    /// after normalizing the periodic boundaries with xp = xp % period.
    /// \param fp 1-D sequence of float or complex
    /// The y-coordinates of the data points, same length as xp.
    /// \param left optional float or complex corresponding to fp
    /// Value to return for x < xp[0], default is fp[0].
    /// \param right optional float or complex corresponding to fp
    /// Value to return for x > xp[-1], default is fp[-1].
    /// \param period None or float, optional
    /// A period for the x-coordinates. This parameter allows the proper interpolation of angular x-coordinates. Parameters left and right are ignored if period is specified.
    ///
    /// \return
    // The interpolated values, same shape as x.
    ///
    //////////////////////////////////////////////////////////////
    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2, Arithmetic DType3, typename Derived3, typename Storage3>
    inline auto interp(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &x,
                       const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &xp,
                       const ndarray::internal::NDArrayBase<DType3, Derived3, Storage3> &fp,
                       std::optional<DType1> = std::nullopt,
                       std::optional<DType1> = std::nullopt,
                       std::optional<DType1> = std::nullopt) {
        if (xp.empty()) {
            throw std::runtime_error("Array of sample points is empty");
        }
        if (xp.ndim() != 1) {
            throw std::runtime_error("xp must be 1 dimensional array");
        }
        if (fp.ndim() != 1) {
            throw std::runtime_error("fp must be 1 dimensional array");
        }
        if (xp.size() != fp.size()) {
            throw std::runtime_error("fp and xp are not of the same length");
        }
        NDArrayDynamic<std::pair<DType2, DType3>> target{Shape{xp.size()}};
        for (Size i = 0; i < xp.size(); ++i) {
            target.set(i, {xp.get(i), fp.get(i)});
        }
        target.sort();

        NDArrayDynamic<DType3> result{x.shape()};
        for (Size i = 0; i < x.size(); ++i) {
            auto element = x.get(i);
            auto it = std::upper_bound(target.cbegin(), target.cend(), element, [](const auto &c1, const auto &c2) {
                return c1 < c2.first;
            });
            if (it == target.cend()) {
                result.set(i, target.get(target.size() - 1).second);
            } else {
                auto x0 = (*std::prev(it)).first;
                auto y0 = (*std::prev(it)).second;
                auto x1 = (*it).first;
                auto y1 = (*it).second;
                auto derivative = static_cast<DType3>((y1 - y0)) / (x1 - x0);
                result.set(i, y0 + derivative * (element - x0));
            }
        }

        return result;
    }
}// namespace np

/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/Constants.hpp>

#include <np/Exception.hpp>
#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/internal/Expression.hpp>
#include <np/ndarray/internal/Math.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    using ndarray::array_dynamic::NDArrayDynamic;
    using ndarray::array_static::NDArrayStatic;

    //////////////////////////////////////////////////////////////
    /// \brief Sum of array elements
    ///
    /// Calculate the array elements sum
    ///
    /// \param array array to sum
    ///
    /// \return The sum of the elements
    ///
    //////////////////////////////////////////////////////////////
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto sum(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.sum();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Sum of expression elements matching a boolean condition
    ///
    /// Count elements of an expression matching a condition like "dist<1".
    /// For expression trees (e.g., rx*rx + ry*ry), this uses the fused
    /// AVX2 count_if() path with no intermediate array allocations.
    ///
    /// \param cond A condition string (e.g., "dist<1", "x>0.5")
    /// \param expr An expression to evaluate the condition on
    ///
    /// \return Count of elements matching the condition as float_
    ///
    //////////////////////////////////////////////////////////////
    template<typename Derived, typename DType>
    inline auto sum(const std::string &cond, const ndarray::internal::ExpressionBase<Derived, DType> &expr) {
        return expr.sum(cond);
    }

    // Operator overloads that return expressions
    // lvalue + lvalue
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    auto operator+(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<AddOp, decltype(makeExpression(a)), decltype(makeExpression(b))>(
                makeExpression(a), makeExpression(b));
    }

    // rvalue + lvalue: use makeExpression rvalue overload which creates an ArrayExpression
    // that owns its data via shared_ptr, preventing dangling pointers.
    // Only enabled for concrete dynamic arrays (NDArrayDynamicStorage), not indexed views or expressions.
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>>
    auto operator+(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<AddOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(b))>(
                makeExpression(std::move(a)), makeExpression(b));
    }

    // lvalue + rvalue: use makeExpression rvalue overload
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator+(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<AddOp, decltype(makeExpression(a)), decltype(makeExpression(std::move(b)))>(
                makeExpression(a), makeExpression(std::move(b)));
    }

    // rvalue + rvalue: use makeExpression rvalue overload for both
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator+(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<AddOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(std::move(b)))>(
                makeExpression(std::move(a)), makeExpression(std::move(b)));
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator+=(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.addInplace(array2);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator+(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.add(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator+(const DType2 &value, const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array) {
        return array.add(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator+=(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.addInplace(value);
    }

    // lvalue - lvalue
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    auto operator-(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<SubtractOp, decltype(makeExpression(a)), decltype(makeExpression(b))>(
                makeExpression(a), makeExpression(b));
    }

    // rvalue - lvalue: use makeExpression rvalue overload
    // Only enabled for concrete dynamic arrays (NDArrayDynamicStorage)
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>>
    auto operator-(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<SubtractOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(b))>(
                makeExpression(std::move(a)), makeExpression(b));
    }

    // lvalue - rvalue: use makeExpression rvalue overload
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator-(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<SubtractOp, decltype(makeExpression(a)), decltype(makeExpression(std::move(b)))>(
                makeExpression(a), makeExpression(std::move(b)));
    }

    // rvalue - rvalue: use makeExpression rvalue overload for both
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator-(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<SubtractOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(std::move(b)))>(
                makeExpression(std::move(a)), makeExpression(std::move(b)));
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto operator-=(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.subtractInplace(array2);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator-(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.subtract(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator-=(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.subtractInplace(value);
    }


    // lvalue * lvalue
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    auto operator*(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<MultiplyOp, decltype(makeExpression(a)), decltype(makeExpression(b))>(
                makeExpression(a), makeExpression(b));
    }

    // rvalue * lvalue: use makeExpression rvalue overload
    // Only enabled for concrete dynamic arrays (NDArrayDynamicStorage)
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>>
    auto operator*(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<MultiplyOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(b))>(
                makeExpression(std::move(a)), makeExpression(b));
    }

    // lvalue * rvalue: use makeExpression rvalue overload
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator*(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<MultiplyOp, decltype(makeExpression(a)), decltype(makeExpression(std::move(b)))>(
                makeExpression(a), makeExpression(std::move(b)));
    }

    // rvalue * rvalue: use makeExpression rvalue overload for both
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator*(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<MultiplyOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(std::move(b)))>(
                makeExpression(std::move(a)), makeExpression(std::move(b)));
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator*(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.multiply(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator*(const DType2 &value, const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array) {
        return array.multiply(value);
    }

    // lvalue / lvalue
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    auto operator/(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<DivideOp, decltype(makeExpression(a)), decltype(makeExpression(b))>(
                makeExpression(a), makeExpression(b));
    }

    // rvalue / lvalue: use makeExpression rvalue overload
    // Only enabled for concrete dynamic arrays (NDArrayDynamicStorage)
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>>
    auto operator/(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        using namespace ndarray::internal;
        return BinaryExpression<DivideOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(b))>(
                makeExpression(std::move(a)), makeExpression(b));
    }

    // lvalue / rvalue: use makeExpression rvalue overload
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator/(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<DivideOp, decltype(makeExpression(a)), decltype(makeExpression(std::move(b)))>(
                makeExpression(a), makeExpression(std::move(b)));
    }

    // rvalue / rvalue: use makeExpression rvalue overload for both
    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2,
             typename = std::enable_if_t<std::is_same_v<Storage1, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType1>>>,
             typename = std::enable_if_t<std::is_same_v<Storage2, ndarray::array_dynamic::internal::NDArrayDynamicStorage<DType2>>>>
    auto operator/(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &&a,
                   ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &&b) {
        using namespace ndarray::internal;
        return BinaryExpression<DivideOp, decltype(makeExpression(std::move(a))), decltype(makeExpression(std::move(b)))>(
                makeExpression(std::move(a)), makeExpression(std::move(b)));
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator/(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array, const DType2 &value) {
        return array.divide(value);
    }

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2>
    inline auto operator/(const DType2 &value, const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array) {
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

    template<Arithmetic DType1, typename Derived1, typename Storage1, Arithmetic DType2, typename Derived2, typename Storage2>
    inline auto expInplace(ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &array2) {
        return array1.expInplace(array2);
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

    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto sqrtInplace(ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.sqrtInplace();
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

    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto sinInplace(ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.sinInplace();
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

    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto cosInplace(ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.cosInplace();
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto logInplace(ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.logInplace();
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
    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto absInplace(ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
        return array.absInplace();
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
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Array of sample points is empty");
        }
        if (xp.ndim() != 1) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "xp must be 1 dimensional array");
        }
        if (fp.ndim() != 1) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "fp must be 1 dimensional array");
        }
        if (xp.size() != fp.size()) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "fp and xp are not of the same length");
        }

        const auto x_size = x.size();
        const auto xp_size = xp.size();

        NDArrayDynamic<DType3> result{x.shape()};
        auto *result_data = result.data();

        // Fast path: use direct pointer access when all inputs are contiguous
        if constexpr (Storage1::is_contiguous && Storage2::is_contiguous && Storage3::is_contiguous) {
            const auto *xp_data = xp.data();
            const auto *fp_data = fp.data();
            const auto *x_data = x.data();

            if (xp_size <= 2) {
                // Fast path for tiny xp (size 1 or 2):
                // Avoid std::upper_bound binary search overhead (O(log n) for just 1-2 elements).
                // Use simple direct comparisons instead.
                if (xp_size == 1) {
                    // Single sample point: return fp[0] for all x
                    auto y0 = fp_data[0];
                    for (Size i = 0; i < x_size; ++i) {
                        result_data[i] = y0;
                    }
                } else {
                    // xp_size == 2: linear interpolation between two points
                    auto x0 = xp_data[0];
                    auto y0 = fp_data[0];
                    auto x1 = xp_data[1];
                    auto y1 = fp_data[1];
                    auto inv_dx = static_cast<DType3>(1.0) / static_cast<DType3>(x1 - x0);
                    // SIMD-accelerated path for double/float contiguous arrays
                    if constexpr (std::is_same_v<DType1, double> && std::is_same_v<DType3, double> && Storage1::is_contiguous) {
                        internal::interp_pd(reinterpret_cast<const double *>(x_data),
                                            static_cast<double>(x0),
                                            static_cast<double>(y0),
                                            static_cast<double>(x1),
                                            static_cast<double>(y1),
                                            static_cast<double>(inv_dx),
                                            reinterpret_cast<double *>(result_data),
                                            static_cast<std::size_t>(x_size));
                    } else if constexpr (std::is_same_v<DType1, float> && std::is_same_v<DType3, float> && Storage1::is_contiguous) {
                        internal::interp_ps(reinterpret_cast<const float *>(x_data),
                                            static_cast<float>(x0),
                                            static_cast<float>(y0),
                                            static_cast<float>(x1),
                                            static_cast<float>(y1),
                                            static_cast<float>(inv_dx),
                                            reinterpret_cast<float *>(result_data),
                                            static_cast<std::size_t>(x_size));
                    } else {
                        for (Size i = 0; i < x_size; ++i) {
                            auto element = x_data[i];
                            if (element <= x0) {
                                result_data[i] = y0;
                            } else if (element >= x1) {
                                result_data[i] = y1;
                            } else {
                                auto t = static_cast<DType3>(element - x0) * inv_dx;
                                result_data[i] = static_cast<DType3>(y0) + t * static_cast<DType3>(y1 - y0);
                            }
                        }
                    }
                }
            } else {
                for (Size i = 0; i < x_size; ++i) {
                    auto element = x_data[i];
                    auto it = std::upper_bound(xp_data, xp_data + xp_size, element);
                    if (it == xp_data + xp_size) {
                        result_data[i] = fp_data[xp_size - 1];
                    } else if (it == xp_data) {
                        result_data[i] = fp_data[0];
                    } else {
                        auto idx = static_cast<Size>(it - xp_data);
                        auto x0 = xp_data[idx - 1];
                        auto y0 = fp_data[idx - 1];
                        auto x1 = xp_data[idx];
                        auto y1 = fp_data[idx];
                        auto derivative = static_cast<DType3>((y1 - y0)) / (x1 - x0);
                        result_data[i] = static_cast<DType3>(y0 + derivative * (element - x0));
                    }
                }
            }
        } else {
            // Fallback for non-contiguous inputs: use virtual get() dispatch
            if (xp_size <= 2) {
                if (xp_size == 1) {
                    auto y0 = fp.get(0);
                    for (Size i = 0; i < x_size; ++i) {
                        result_data[i] = y0;
                    }
                } else {
                    auto x0 = xp.get(0);
                    auto y0 = fp.get(0);
                    auto x1 = xp.get(1);
                    auto y1 = fp.get(1);
                    auto inv_dx = static_cast<DType3>(1.0) / static_cast<DType3>(x1 - x0);
                    for (Size i = 0; i < x_size; ++i) {
                        auto element = x.get(i);
                        if (element <= x0) {
                            result_data[i] = y0;
                        } else if (element >= x1) {
                            result_data[i] = y1;
                        } else {
                            auto t = static_cast<DType3>(element - x0) * inv_dx;
                            result_data[i] = static_cast<DType3>(y0) + t * static_cast<DType3>(y1 - y0);
                        }
                    }
                }
            } else {
                // Build sorted index for binary search on non-contiguous xp/fp
                // Use a vector of pairs for binary search
                std::vector<std::pair<DType2, DType3>> target(xp_size);
                for (Size i = 0; i < xp_size; ++i) {
                    target[i] = {xp.get(i), fp.get(i)};
                }
                // xp is already monotonically increasing, so no sort needed
                for (Size i = 0; i < x_size; ++i) {
                    auto element = x.get(i);
                    auto it = std::upper_bound(target.cbegin(), target.cend(), element,
                                               [](const auto &c1, const auto &c2) {
                                                   return c1 < c2.first;
                                               });
                    if (it == target.cend()) {
                        result_data[i] = target[xp_size - 1].second;
                    } else if (it == target.cbegin()) {
                        result_data[i] = target[0].second;
                    } else {
                        auto idx = static_cast<Size>(it - target.cbegin());
                        auto x0 = target[idx - 1].first;
                        auto y0 = target[idx - 1].second;
                        auto x1 = target[idx].first;
                        auto y1 = target[idx].second;
                        auto derivative = static_cast<DType3>((y1 - y0)) / (x1 - x0);
                        result_data[i] = static_cast<DType3>(y0 + derivative * (element - x0));
                    }
                }
            }
        }

        return result;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Evenly round to the given number of decimals.
    ///
    /// \param a input data
    /// \param decimals Number of decimal places to round to (default: 0). If decimals is negative, it specifies the number of positions to the left of the decimal point.
    ///
    /// \return
    /// An array of the same type as a, containing the rounded values. Unless out was specified, a new array is created. A reference to the result is returned.
    ///
    /// The real and imaginary parts of complex numbers are rounded separately. The result of rounding a float is a float.
    ///
    //////////////////////////////////////////////////////////////
    template<Arithmetic DType>
    inline auto round(DType a, int decimals = 0) {
        auto digits = std::pow(10, decimals);
        return std::round(a * digits) / digits;
    }

    template<Arithmetic DType, typename Derived, typename Storage>
    inline auto round(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &a, int decimals = 0) {
        NDArrayDynamic<DType> result{a.shape()};
        auto multiplier = std::pow(10, decimals);
        for (Size i = 0; i < result.size(); ++i) {
            result.set(i, std::round(a.get(i) * multiplier) / multiplier);
        }
        return result;
    }
}// namespace np

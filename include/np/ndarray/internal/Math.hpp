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

#include <cstddef>
#include <istream>
#include <ostream>
#include <string>

#include <np/Constants.hpp>
#include <np/Shape.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> add(DType1, DType2, DType3 &) {
                throw std::runtime_error("Plus called for non-arithmetic types");
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> add(DType1 arg1, DType2 arg2, DType3 &result) {
                result = arg1 + arg2;
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> add(DType1 &, DType2) {
                throw std::runtime_error("add called for non-arithmetic type");
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> add(DType1 &arg1, DType2 arg2) {
                arg1 += static_cast<DType1>(arg2);
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> subtract(DType1, DType2, DType3 &) {
                throw std::runtime_error("subtract called for non-arithmetic types");
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> subtract(DType1 arg1, DType2 arg2, DType3 &result) {
                result = arg1 - arg2;
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> subtract(DType1 &, DType2) {
                throw std::runtime_error("subtract called for non-arithmetic type");
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> subtract(DType1 &arg1, DType2 arg2) {
                arg1 -= static_cast<DType1>(arg2);
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> multiply(DType1, DType2, DType3 &) {
                throw std::runtime_error("multiply called for non-arithmetic type");
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> multiply(DType1 arg1, DType2 arg2, DType3 &result) {
                result = arg1 * arg2;
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> multiply(DType1 &, DType2) {
                throw std::runtime_error("multiply called for non-arithmetic type");
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> multiply(DType1 &arg1, DType2 arg2) {
                arg1 *= arg2;
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> divide(DType1, DType2, DType3 &) {
                throw std::runtime_error("divide called for non-arithmetic type");
            }

            template<typename DType1, typename DType2, typename DType3>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> divide(DType1 arg1, DType2 arg2, DType3 &result) {
                result = arg1 / arg2;
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<!std::is_arithmetic_v<DType1> || !std::is_arithmetic_v<DType2>> divide(DType1 &, DType2) {
                throw std::runtime_error("divide called for non-arithmetic type");
            }

            template<typename DType1, typename DType2>
            std::enable_if_t<std::is_arithmetic_v<DType1> && std::is_arithmetic_v<DType2>> divide(DType1 &arg1, DType2 arg2) {
                arg1 /= arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> exp(DType, DType &) {
                throw std::runtime_error("exp called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> exp(DType arg, DType &result) {
                result = std::exp(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> exp(DType &) {
                throw std::runtime_error("exp called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> expInplace(DType &arg) {
                arg = std::exp(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> setDouble(float_, DType &) {
                throw std::runtime_error("setDouble called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> setDouble(float_ arg, DType &result) {
                result = arg;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sqrt(DType, DType &) {
                throw std::runtime_error("sqrt called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sqrt(DType arg, DType &result) {
                result = std::sqrt(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sqrt(DType &) {
                throw std::runtime_error("sqrt called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sqrt(DType &arg) {
                arg = std::sqrt(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sin(DType, DType &) {
                throw std::runtime_error("sin called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sin(DType arg, DType &result) {
                result = std::sin(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sin(DType &) {
                throw std::runtime_error("sin called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sin(DType &arg) {
                arg = std::sin(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> cos(DType, DType &) {
                throw std::runtime_error("cos called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> cos(DType arg, DType &result) {
                result = std::cos(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> cos(DType &) {
                throw std::runtime_error("cos called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> cos(DType &arg) {
                arg = std::cos(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> log(DType, DType &) {
                throw std::runtime_error("log called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> log(DType arg, DType &result) {
                result = std::log(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> log(DType &) {
                throw std::runtime_error("log called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> log(DType &arg) {
                arg = std::log(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> abs(DType, DType &) {
                throw std::runtime_error("abs called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> abs(DType arg, DType &result) {
                result = std::abs(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> abs(DType &) {
                throw std::runtime_error("abs called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> abs(DType &arg) {
                arg = std::abs(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_floating_point_v<DType>> isNaN(DType, bool &result) {
                result = false;
            }

            template<typename DType>
            std::enable_if_t<std::is_floating_point_v<DType>> isNaN(DType arg, bool &result) {
                result = std::isnan(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_floating_point_v<DType>> nanToZero(DType arg, DType &result) {
                result = arg;
            }

            template<typename DType>
            std::enable_if_t<std::is_floating_point_v<DType>> nanToZero(DType arg, DType &result) {
                result = std::isnan(arg) ? 0 : arg;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> isTrue(DType, bool &) {
                throw std::runtime_error("isTrue is called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> isTrue(DType arg, bool &result) {
                result = static_cast<bool>(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> isFalse(DType, bool &) {
                throw std::runtime_error("isFalse is called for non-arithmetic type");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> isFalse(DType arg, bool &result) {
                result = !static_cast<bool>(arg);
            }
        }// namespace internal
    }    // namespace ndarray
}// namespace np
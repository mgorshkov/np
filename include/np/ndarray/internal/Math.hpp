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
            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> add(DType, DType, DType &) {
                throw std::runtime_error("Plus called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> add(DType arg1, DType arg2, DType &result) {
                result = arg1 + arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> subtract(DType, DType, DType &) {
                throw std::runtime_error("Subtract called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> subtract(DType arg1, DType arg2, DType &result) {
                result = arg1 - arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> multiply(DType, DType, DType &) {
                throw std::runtime_error("Multiply called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> multiply(DType arg1, DType arg2, DType &result) {
                result = arg1 * arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> divide(DType, DType, DType &) {
                throw std::runtime_error("Divide called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> divide(DType arg1, DType arg2, DType &result) {
                result = arg1 / arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> divideBySize(DType, Size, float_ &) {
                throw std::runtime_error("Divide called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> divideBySize(DType arg1, Size arg2, float_ &result) {
                result = arg1 / arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> divideByDouble(DType, float_, DType &) {
                throw std::runtime_error("Divide called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> divideByDouble(DType arg1, float_ arg2, DType &result) {
                result = arg1 / arg2;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> exp(DType, DType &) {
                throw std::runtime_error("Exp called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> exp(DType arg, DType &result) {
                result = std::exp(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> setDouble(float_, DType &) {
                throw std::runtime_error("setDouble called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> setDouble(float_ arg, DType &result) {
                result = arg;
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sqrt(DType, DType &) {
                throw std::runtime_error("Sqrt called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sqrt(DType arg, DType &result) {
                result = std::sqrt(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> sin(DType, DType &) {
                throw std::runtime_error("Sin called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> sin(DType arg, DType &result) {
                result = std::sin(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> cos(DType, DType &) {
                throw std::runtime_error("Cos called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> cos(DType arg, DType &result) {
                result = std::cos(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> log(DType, DType &) {
                throw std::runtime_error("Log called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> log(DType arg, DType &result) {
                result = std::log(arg);
            }

            template<typename DType>
            std::enable_if_t<!std::is_arithmetic_v<DType>> abs(DType, DType &) {
                throw std::runtime_error("Abs called for non-arithmetic types");
            }

            template<typename DType>
            std::enable_if_t<std::is_arithmetic_v<DType>> abs(DType arg, DType &result) {
                result = std::abs(arg);
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
        }// namespace internal
    }    // namespace ndarray
}// namespace np
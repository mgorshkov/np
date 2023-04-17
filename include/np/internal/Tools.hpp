/*
MIT License

Copyright (c) 2023 Mikhail Gorshkov

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
#include <math.h>
#include <tuple>
#include <type_traits>

namespace np {
    namespace internal {
        template<typename DType>
        typename std::enable_if<!std::numeric_limits<DType>::is_integer, bool>::type almost_equal(DType x, DType y, int ulp) {
            // the machine epsilon has to be scaled to the magnitude of the values used
            // and multiplied by the desired precision in ULPs (units in the last place)
            // unless the result is subnormal
            return std::fabs(x - y) <= std::numeric_limits<DType>::epsilon() * std::fabs(x + y) * ulp ||
                   std::fabs(x - y) < std::numeric_limits<DType>::min();
        }

        template<typename DType1, typename DType2>
        inline static bool element_equal(const DType1 &value1, const DType2 &value2) {
            return value1 == value2;
        }

        static const constexpr int ULP_TOLERANCE = 7;

        inline static bool element_equal(const float_ &value1, const float_ &value2) {
            return (std::isnan(value1) && std::isnan(value2)) || almost_equal(value1, value2, ULP_TOLERANCE);
        }
    }// namespace internal
}// namespace np

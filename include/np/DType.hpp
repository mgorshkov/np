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

#include <array>
#include <complex>
#include <any>
#include <string>

namespace np {

    // Integer types
    // Signed 8-bit integer types
    using int8 = std::int8_t;
    // Signed 16-bit integer types
    using int16 = std::int16_t;
    // Signed 32-bit integer types
    using int32 = std::int32_t;
    // Signed 64-bit integer types
    using int64 = std::int64_t;

    // Unsigned 8-bit integer types
    using uint8 = std::uint8_t;
    // Unsigned 16-bit integer types
    using uint16 = std::uint16_t;
    // Unsigned 32-bit integer types
    using uint32 = std::uint32_t;
    // Unsigned 64-bit integer types
    using uint64 = std::uint64_t;

    // Standard double-precision floating point
    using float_ = double;
    using float32 = float;
    using float64 = long double;
    using longdouble = long double;

    // Complex numbers represented by 128 floats
    template <typename T>
    using complex_ = std::complex<T>;

    // Boolean type storing TRUE and FALSE values
    using bool_ = bool;

    using byte = signed char;
    using intc = int;
    using short_ = short;
    using int_ = long;
    using longlong = long long;

    using ubyte = unsigned char;
    using ushort = unsigned short;
    using uintc = unsigned int;
    using uint = unsigned long;
    using ulonglong = unsigned long long;

    // Python object type
    using object = std::any;

    // String type
    using string_ = std::string;

    // Unicode type
    using unicode_ = std::wstring;

    using DTypeDefault = float_;
}

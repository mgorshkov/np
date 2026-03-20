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

#include <limits>

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Type traits for GELSD scalar constants
            // ============================================================

            template<typename T>
            struct GelsdTraits;

            template<>
            struct GelsdTraits<double> {
                static constexpr double tol() { return 1e-15; }
                static constexpr double eps() { return 2.2e-16; }
                static constexpr double deflation_tol() { return 1e-12; }
            };

            template<>
            struct GelsdTraits<float> {
                static constexpr float tol() { return 1e-7f; }
                static constexpr float eps() { return 1.2e-7f; }
                static constexpr float deflation_tol() { return 1e-6f; }
            };

        }// namespace cpu
    }// namespace internal
}// namespace np

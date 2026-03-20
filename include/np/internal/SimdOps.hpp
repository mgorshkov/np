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

#include <cstddef>
#include <cstdint>

#include <np/internal/CpuDispatch.hpp>

namespace np {
    namespace internal {

        /// @defgroup simd_dispatch SIMD Dispatch Function Pointers
        ///
        /// These function pointers are initialized at startup by SimdOps_init.cpp
        /// based on the runtime CPU capabilities (capped by compile-time ENABLE_* macros).
        ///
        /// Each operation has implementations for different SIMD levels in separate
        /// translation units compiled with appropriate -march flags.

        // ---- Element-wise operations on double arrays ----

        /// Add two double arrays element-wise: result[i] = a[i] + b[i] for i in [0, n)
        using add_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n);
        extern add_pd_fn add_pd;

        /// Subtract two double arrays element-wise: result[i] = a[i] - b[i]
        using sub_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n);
        extern sub_pd_fn sub_pd;

        /// Multiply two double arrays element-wise: result[i] = a[i] * b[i]
        using mul_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n);
        extern mul_pd_fn mul_pd;

        /// Divide two double arrays element-wise: result[i] = a[i] / b[i]
        using div_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n);
        extern div_pd_fn div_pd;

        // ---- Element-wise operations on float arrays ----

        /// Add two float arrays element-wise: result[i] = a[i] + b[i]
        using add_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n);
        extern add_ps_fn add_ps;

        /// Subtract two float arrays element-wise: result[i] = a[i] - b[i]
        using sub_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n);
        extern sub_ps_fn sub_ps;

        /// Multiply two float arrays element-wise: result[i] = a[i] * b[i]
        using mul_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n);
        extern mul_ps_fn mul_ps;

        /// Divide two float arrays element-wise: result[i] = a[i] / b[i]
        using div_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n);
        extern div_ps_fn div_ps;

        // ---- Comparison operations ----

        /// Count elements where a[i] < threshold for double arrays.
        /// Returns the number of elements satisfying the condition.
        using count_lt_pd_fn = std::size_t (*)(const double *a, double threshold, std::size_t n);
        extern count_lt_pd_fn count_lt_pd;

        /// Count elements where a[i] < threshold for float arrays.
        using count_lt_ps_fn = std::size_t (*)(const float *a, float threshold, std::size_t n);
        extern count_lt_ps_fn count_lt_ps;

        // ---- Element-wise absolute value ----

        /// Compute absolute value of double array: result[i] = abs(a[i])
        using abs_pd_fn = void (*)(const double *a, double *result, std::size_t n);
        extern abs_pd_fn abs_pd;

        /// Compute absolute value of float array: result[i] = abs(a[i])
        using abs_ps_fn = void (*)(const float *a, float *result, std::size_t n);
        extern abs_ps_fn abs_ps;

        // ---- Fused operations (single pass, no intermediate arrays) ----

        /// Fused abs(a - b) for double arrays: result[i] = abs(a[i] - b[i])
        /// Single pass avoids intermediate array allocation.
        using abs_sub_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n);
        extern abs_sub_pd_fn abs_sub_pd;

        /// Fused abs(a - b) for float arrays: result[i] = abs(a[i] - b[i])
        using abs_sub_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n);
        extern abs_sub_ps_fn abs_sub_ps;

        /// Fused sum(a * a * w) for double arrays: returns sum(a[i]*a[i]*w[i])
        /// Single pass avoids intermediate array allocation for r*r*w.
        using sum_sq_weighted_pd_fn = double (*)(const double *a, const double *w, std::size_t n);
        extern sum_sq_weighted_pd_fn sum_sq_weighted_pd;

        /// Fused sum(a * a * w) for float arrays: returns sum(a[i]*a[i]*w[i])
        using sum_sq_weighted_ps_fn = float (*)(const float *a, const float *w, std::size_t n);
        extern sum_sq_weighted_ps_fn sum_sq_weighted_ps;

        // ---- Conditional selection (where) ----

        /// Tukey bisquare weight update for double arrays.
        /// For each element a[i]:
        ///   if a[i] <= k: result[i] = 1.0
        ///   else:         result[i] = 2*k/a[i] - k*k/(a[i]*a[i])
        using where_tukey_pd_fn = void (*)(const double *a, double k, double *result, std::size_t n);
        extern where_tukey_pd_fn where_tukey_pd;

        /// Tukey bisquare weight update for float arrays.
        using where_tukey_ps_fn = void (*)(const float *a, float k, float *result, std::size_t n);
        extern where_tukey_ps_fn where_tukey_ps;

        // ---- Linear interpolation (interp) ----

        /// Linear interpolation for the xp_size == 2 fast path (double).
        /// For each element x[i]:
        ///   if x[i] <= x0: result[i] = y0
        ///   if x[i] >= x1: result[i] = y1
        ///   else:          result[i] = y0 + (x[i] - x0) * (y1 - y0) * inv_dx
        using interp_pd_fn = void (*)(const double *x, double x0, double y0, double x1, double y1, double inv_dx, double *result, std::size_t n);
        extern interp_pd_fn interp_pd;

        /// Linear interpolation for the xp_size == 2 fast path (float).
        using interp_ps_fn = void (*)(const float *x, float x0, float y0, float x1, float y1, float inv_dx, float *result, std::size_t n);
        extern interp_ps_fn interp_ps;

        // ---- AMX operations (only available with AVX512 + AMX) ----

        /// Process a tile of double data using AMX + AVX512.
        /// Processes kAmxDoublesPerTile (128) doubles at once.
        using amx_process_pd_fn = void (*)(const double *a, const double *b, double *result, std::size_t n,
                                           void (*op)(const double *, const double *, double *, std::size_t));
        extern amx_process_pd_fn amx_process_pd;

        /// Process a tile of float data using AMX + AVX512.
        /// Processes kAmxFloatsPerTile (256) floats at once.
        using amx_process_ps_fn = void (*)(const float *a, const float *b, float *result, std::size_t n,
                                           void (*op)(const float *, const float *, float *, std::size_t));
        extern amx_process_ps_fn amx_process_ps;

        /// Initialize the SIMD function pointers based on runtime CPU detection.
        /// Called automatically at startup via static initialization.
        void init_simd_dispatch();

    }// namespace internal
}// namespace np

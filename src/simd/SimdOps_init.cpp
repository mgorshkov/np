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

/// Initializes SIMD function pointers based on runtime CPU detection.
/// Each function pointer is set to the best implementation supported
/// by the current CPU, capped by the compile-time ENABLE_* macros.

#include <np/internal/CpuDispatch.hpp>
#include <np/internal/SimdOps.hpp>

namespace np {
    namespace internal {

        // Scalar implementations (always available)
        void add_pd_scalar(const double *, const double *, double *, std::size_t);
        void sub_pd_scalar(const double *, const double *, double *, std::size_t);
        void mul_pd_scalar(const double *, const double *, double *, std::size_t);
        void div_pd_scalar(const double *, const double *, double *, std::size_t);
        void add_ps_scalar(const float *, const float *, float *, std::size_t);
        void sub_ps_scalar(const float *, const float *, float *, std::size_t);
        void mul_ps_scalar(const float *, const float *, float *, std::size_t);
        void div_ps_scalar(const float *, const float *, float *, std::size_t);
        std::size_t count_lt_pd_scalar(const double *, double, std::size_t);
        std::size_t count_lt_ps_scalar(const float *, float, std::size_t);
        void abs_pd_scalar(const double *, double *, std::size_t);
        void abs_ps_scalar(const float *, float *, std::size_t);
        void where_tukey_pd_scalar(const double *, double, double *, std::size_t);
        void where_tukey_ps_scalar(const float *, float, float *, std::size_t);
        void abs_sub_pd_scalar(const double *, const double *, double *, std::size_t);
        void abs_sub_ps_scalar(const float *, const float *, float *, std::size_t);
        double sum_sq_weighted_pd_scalar(const double *, const double *, std::size_t);
        float sum_sq_weighted_ps_scalar(const float *, const float *, std::size_t);
        void interp_pd_scalar(const double *, double, double, double, double, double, double *, std::size_t);
        void interp_ps_scalar(const float *, float, float, float, float, float, float *, std::size_t);

        // AVX2 implementations (compiled with -mavx2)
        void add_pd_avx2(const double *, const double *, double *, std::size_t);
        void sub_pd_avx2(const double *, const double *, double *, std::size_t);
        void mul_pd_avx2(const double *, const double *, double *, std::size_t);
        void div_pd_avx2(const double *, const double *, double *, std::size_t);
        void add_ps_avx2(const float *, const float *, float *, std::size_t);
        void sub_ps_avx2(const float *, const float *, float *, std::size_t);
        void mul_ps_avx2(const float *, const float *, float *, std::size_t);
        void div_ps_avx2(const float *, const float *, float *, std::size_t);
        std::size_t count_lt_pd_avx2(const double *, double, std::size_t);
        std::size_t count_lt_ps_avx2(const float *, float, std::size_t);
        void abs_pd_avx2(const double *, double *, std::size_t);
        void abs_ps_avx2(const float *, float *, std::size_t);
        void where_tukey_pd_avx2(const double *, double, double *, std::size_t);
        void where_tukey_ps_avx2(const float *, float, float *, std::size_t);
        void abs_sub_pd_avx2(const double *, const double *, double *, std::size_t);
        void abs_sub_ps_avx2(const float *, const float *, float *, std::size_t);
        double sum_sq_weighted_pd_avx2(const double *, const double *, std::size_t);
        float sum_sq_weighted_ps_avx2(const float *, const float *, std::size_t);
        void interp_pd_avx2(const double *, double, double, double, double, double, double *, std::size_t);
        void interp_ps_avx2(const float *, float, float, float, float, float, float *, std::size_t);

        // AVX-512 implementations (compiled with -mavx512f)
        void add_pd_avx512(const double *, const double *, double *, std::size_t);
        void sub_pd_avx512(const double *, const double *, double *, std::size_t);
        void mul_pd_avx512(const double *, const double *, double *, std::size_t);
        void div_pd_avx512(const double *, const double *, double *, std::size_t);
        void add_ps_avx512(const float *, const float *, float *, std::size_t);
        void sub_ps_avx512(const float *, const float *, float *, std::size_t);
        void mul_ps_avx512(const float *, const float *, float *, std::size_t);
        void div_ps_avx512(const float *, const float *, float *, std::size_t);
        std::size_t count_lt_pd_avx512(const double *, double, std::size_t);
        std::size_t count_lt_ps_avx512(const float *, float, std::size_t);
        void abs_pd_avx512(const double *, double *, std::size_t);
        void abs_ps_avx512(const float *, float *, std::size_t);
        void where_tukey_pd_avx512(const double *, double, double *, std::size_t);
        void where_tukey_ps_avx512(const float *, float, float *, std::size_t);
        void abs_sub_pd_avx512(const double *, const double *, double *, std::size_t);
        void abs_sub_ps_avx512(const float *, const float *, float *, std::size_t);
        double sum_sq_weighted_pd_avx512(const double *, const double *, std::size_t);
        float sum_sq_weighted_ps_avx512(const float *, const float *, std::size_t);
        void interp_pd_avx512(const double *, double, double, double, double, double, double *, std::size_t);
        void interp_ps_avx512(const float *, float, float, float, float, float, float *, std::size_t);

        // AMX implementations (compiled with -mamx-tile)
        void amx_process_pd_impl(const double *, const double *, double *, std::size_t,
                                 void (*)(const double *, const double *, double *, std::size_t));
        void amx_process_ps_impl(const float *, const float *, float *, std::size_t,
                                 void (*)(const float *, const float *, float *, std::size_t));
        void interp_pd_amx(const double *, double, double, double, double, double, double *, std::size_t);
        void interp_ps_amx(const float *, float, float, float, float, float, float *, std::size_t);

        // ---- Function pointer definitions ----

        add_pd_fn add_pd = add_pd_scalar;
        sub_pd_fn sub_pd = sub_pd_scalar;
        mul_pd_fn mul_pd = mul_pd_scalar;
        div_pd_fn div_pd = div_pd_scalar;
        add_ps_fn add_ps = add_ps_scalar;
        sub_ps_fn sub_ps = sub_ps_scalar;
        mul_ps_fn mul_ps = mul_ps_scalar;
        div_ps_fn div_ps = div_ps_scalar;
        count_lt_pd_fn count_lt_pd = count_lt_pd_scalar;
        count_lt_ps_fn count_lt_ps = count_lt_ps_scalar;
        abs_pd_fn abs_pd = abs_pd_scalar;
        abs_ps_fn abs_ps = abs_ps_scalar;
        where_tukey_pd_fn where_tukey_pd = where_tukey_pd_scalar;
        where_tukey_ps_fn where_tukey_ps = where_tukey_ps_scalar;
        abs_sub_pd_fn abs_sub_pd = abs_sub_pd_scalar;
        abs_sub_ps_fn abs_sub_ps = abs_sub_ps_scalar;
        sum_sq_weighted_pd_fn sum_sq_weighted_pd = sum_sq_weighted_pd_scalar;
        sum_sq_weighted_ps_fn sum_sq_weighted_ps = sum_sq_weighted_ps_scalar;
        interp_pd_fn interp_pd = interp_pd_scalar;
        interp_ps_fn interp_ps = interp_ps_scalar;
        amx_process_pd_fn amx_process_pd = nullptr;
        amx_process_ps_fn amx_process_ps = nullptr;

        void init_simd_dispatch() {
            SimdLevel level = max_simd_level();

            // Set the best available implementations based on SIMD level.
            // Higher levels override lower ones.

            // AVX2 level
            if (level >= SimdLevel::AVX2) {
                add_pd = add_pd_avx2;
                sub_pd = sub_pd_avx2;
                mul_pd = mul_pd_avx2;
                div_pd = div_pd_avx2;
                add_ps = add_ps_avx2;
                sub_ps = sub_ps_avx2;
                mul_ps = mul_ps_avx2;
                div_ps = div_ps_avx2;
                count_lt_pd = count_lt_pd_avx2;
                count_lt_ps = count_lt_ps_avx2;
                abs_pd = abs_pd_avx2;
                abs_ps = abs_ps_avx2;
                where_tukey_pd = where_tukey_pd_avx2;
                where_tukey_ps = where_tukey_ps_avx2;
                abs_sub_pd = abs_sub_pd_avx2;
                abs_sub_ps = abs_sub_ps_avx2;
                sum_sq_weighted_pd = sum_sq_weighted_pd_avx2;
                sum_sq_weighted_ps = sum_sq_weighted_ps_avx2;
                interp_pd = interp_pd_avx2;
                interp_ps = interp_ps_avx2;
            }

            // AVX-512 level (overrides AVX2)
            if (level >= SimdLevel::AVX512) {
                add_pd = add_pd_avx512;
                sub_pd = sub_pd_avx512;
                mul_pd = mul_pd_avx512;
                div_pd = div_pd_avx512;
                add_ps = add_ps_avx512;
                sub_ps = sub_ps_avx512;
                mul_ps = mul_ps_avx512;
                div_ps = div_ps_avx512;
                count_lt_pd = count_lt_pd_avx512;
                count_lt_ps = count_lt_ps_avx512;
                abs_pd = abs_pd_avx512;
                abs_ps = abs_ps_avx512;
                where_tukey_pd = where_tukey_pd_avx512;
                where_tukey_ps = where_tukey_ps_avx512;
                abs_sub_pd = abs_sub_pd_avx512;
                abs_sub_ps = abs_sub_ps_avx512;
                sum_sq_weighted_pd = sum_sq_weighted_pd_avx512;
                sum_sq_weighted_ps = sum_sq_weighted_ps_avx512;
                interp_pd = interp_pd_avx512;
                interp_ps = interp_ps_avx512;
            }

            // AMX level (adds tile processing on top of AVX-512)
            if (level >= SimdLevel::AMX) {
                amx_process_pd = amx_process_pd_impl;
                amx_process_ps = amx_process_ps_impl;
                interp_pd = interp_pd_amx;
                interp_ps = interp_ps_amx;
            }
        }

        // Static initializer to call init_simd_dispatch() at startup
        namespace {
            struct SimdInit {
                SimdInit() {
                    init_simd_dispatch();
                }
            };
            SimdInit g_simd_init;
        }

    } // namespace internal
} // namespace np

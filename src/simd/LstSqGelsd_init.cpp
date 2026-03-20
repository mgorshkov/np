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

/// Initializes GELSD std::function objects based on runtime CPU detection.

#include <np/internal/CpuDispatch.hpp>
#include <np/internal/cpu/LstSqGelsd.hpp>

namespace np {
    namespace internal {
        namespace cpu {

            // Scalar implementations (always available)
            int lstsq_gelsd_double_scalar(const double *, const double *, double *,
                                          size_t, size_t, double);
            int lstsq_gelsd_float_scalar(const float *, const float *, float *,
                                         size_t, size_t, float);

            // AVX2 implementations
            int lstsq_gelsd_double_avx2(const double *, const double *, double *,
                                        size_t, size_t, double);
            int lstsq_gelsd_float_avx2(const float *, const float *, float *,
                                       size_t, size_t, float);

            // AVX-512 implementations
            int lstsq_gelsd_double_avx512(const double *, const double *, double *,
                                          size_t, size_t, double);
            int lstsq_gelsd_float_avx512(const float *, const float *, float *,
                                         size_t, size_t, float);

            // AMX implementations
            int lstsq_gelsd_double_amx(const double *, const double *, double *,
                                       size_t, size_t, double);
            int lstsq_gelsd_float_amx(const float *, const float *, float *,
                                      size_t, size_t, float);

            // ---- Function pointer definitions ----

            lstsq_gelsd_fn<double> lstsq_gelsd_double = lstsq_gelsd_double_scalar;
            lstsq_gelsd_fn<float> lstsq_gelsd_float = lstsq_gelsd_float_scalar;

            void init_lstsq_gelsd_dispatch() {
                SimdLevel level = max_simd_level();

                // AVX2 level
                if (level >= SimdLevel::AVX2) {
                    lstsq_gelsd_double = lstsq_gelsd_double_avx2;
                    lstsq_gelsd_float = lstsq_gelsd_float_avx2;
                }

                // AVX-512 level (overrides AVX2)
                if (level >= SimdLevel::AVX512) {
                    lstsq_gelsd_double = lstsq_gelsd_double_avx512;
                    lstsq_gelsd_float = lstsq_gelsd_float_avx512;
                }

                // AMX level (overrides AVX-512)
                if (level >= SimdLevel::AMX) {
                    lstsq_gelsd_double = lstsq_gelsd_double_amx;
                    lstsq_gelsd_float = lstsq_gelsd_float_amx;
                }
            }

            // Static initializer
            namespace {
                struct GelsdInit {
                    GelsdInit() {
                        init_lstsq_gelsd_dispatch();
                    }
                };
                GelsdInit g_gelsd_init;
            }

        } // namespace cpu
    } // namespace internal
} // namespace np

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

/// AVX-512 implementations of SIMD operations.

#include <cstddef>
#include <immintrin.h>

// Mark each function with the required target ISA and disable auto-vectorization
// so the compiler only emits AVX512 instructions where explicitly requested.
#define AVX512_TARGET_ATTR __attribute__((target("avx512f,avx512dq,avx512bw,avx512vbmi,avx512vbmi2,avx512vl"), optimize("no-tree-vectorize")))

namespace np {
    namespace internal {

        AVX512_TARGET_ATTR
        void add_pd_avx512(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d vr = _mm512_add_pd(va, vb);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        AVX512_TARGET_ATTR
        void sub_pd_avx512(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d vr = _mm512_sub_pd(va, vb);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        AVX512_TARGET_ATTR
        void mul_pd_avx512(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d vr = _mm512_mul_pd(va, vb);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        AVX512_TARGET_ATTR
        void div_pd_avx512(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d vr = _mm512_div_pd(va, vb);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        AVX512_TARGET_ATTR
        void add_ps_avx512(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                __m512 vr = _mm512_add_ps(va, vb);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        AVX512_TARGET_ATTR
        void sub_ps_avx512(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                __m512 vr = _mm512_sub_ps(va, vb);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        AVX512_TARGET_ATTR
        void mul_ps_avx512(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                __m512 vr = _mm512_mul_ps(va, vb);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        AVX512_TARGET_ATTR
        void div_ps_avx512(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                __m512 vr = _mm512_div_ps(va, vb);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        AVX512_TARGET_ATTR
        std::size_t count_lt_pd_avx512(const double *a, double threshold, std::size_t n) {
            std::size_t count = 0;
            __m512d thresh = _mm512_set1_pd(threshold);
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __mmask8 mask = _mm512_cmp_pd_mask(va, thresh, _CMP_LT_OQ);
                count += static_cast<std::size_t>(__builtin_popcount(mask & 0xFF));
            }
            for (; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        AVX512_TARGET_ATTR
        std::size_t count_lt_ps_avx512(const float *a, float threshold, std::size_t n) {
            std::size_t count = 0;
            __m512 thresh = _mm512_set1_ps(threshold);
            std::size_t i = 0;
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __mmask16 mask = _mm512_cmp_ps_mask(va, thresh, _CMP_LT_OQ);
                count += static_cast<std::size_t>(__builtin_popcount(mask & 0xFFFF));
            }
            for (; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        AVX512_TARGET_ATTR
        void abs_pd_avx512(const double *a, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m512d sign_mask = _mm512_set1_pd(-0.0);
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vr = _mm512_andnot_pd(sign_mask, va);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        AVX512_TARGET_ATTR
        void abs_ps_avx512(const float *a, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vr = _mm512_andnot_ps(sign_mask, va);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        AVX512_TARGET_ATTR
        void where_tukey_pd_avx512(const double *a, double k, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m512d k_vec = _mm512_set1_pd(k);
            const __m512d one_vec = _mm512_set1_pd(1.0);
            const __m512d two_vec = _mm512_set1_pd(2.0);
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                // condition: a[i] <= k
                __mmask8 le_mask = _mm512_cmp_pd_mask(va, k_vec, _CMP_LE_OQ);
                // Tukey bisquare: (k/a)*(2 - k/a) when a > k
                __m512d ratio = _mm512_div_pd(k_vec, va);
                __m512d neg = _mm512_mul_pd(ratio, _mm512_sub_pd(two_vec, ratio));
                // Blend: mask selects 1.0 when condition true, neg when false
                __m512d vr = _mm512_mask_blend_pd(le_mask, neg, one_vec);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0 : (2.0 * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        AVX512_TARGET_ATTR
        void where_tukey_ps_avx512(const float *a, float k, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m512 k_vec = _mm512_set1_ps(k);
            const __m512 one_vec = _mm512_set1_ps(1.0f);
            const __m512 two_vec = _mm512_set1_ps(2.0f);
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                // condition: a[i] <= k
                __mmask16 le_mask = _mm512_cmp_ps_mask(va, k_vec, _CMP_LE_OQ);
                // Tukey bisquare: (k/a)*(2 - k/a) when a > k
                __m512 ratio = _mm512_div_ps(k_vec, va);
                __m512 neg = _mm512_mul_ps(ratio, _mm512_sub_ps(two_vec, ratio));
                // Blend: mask selects 1.0 when condition true, neg when false
                __m512 vr = _mm512_mask_blend_ps(le_mask, neg, one_vec);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0f : (2.0f * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        AVX512_TARGET_ATTR
        void abs_sub_pd_avx512(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m512d sign_mask = _mm512_set1_pd(-0.0);
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vb = _mm512_loadu_pd(b + i);
                __m512d diff = _mm512_sub_pd(va, vb);
                __m512d abs_diff = _mm512_andnot_pd(sign_mask, diff);
                _mm512_storeu_pd(result + i, abs_diff);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        AVX512_TARGET_ATTR
        void abs_sub_ps_avx512(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m512 sign_mask = _mm512_set1_ps(-0.0f);
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                __m512 diff = _mm512_sub_ps(va, vb);
                __m512 abs_diff = _mm512_andnot_ps(sign_mask, diff);
                _mm512_storeu_ps(result + i, abs_diff);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        AVX512_TARGET_ATTR
        double sum_sq_weighted_pd_avx512(const double *a, const double *w, std::size_t n) {
            std::size_t i = 0;
            __m512d sum_vec = _mm512_setzero_pd();
            for (; i + 7 < n; i += 8) {
                __m512d va = _mm512_loadu_pd(a + i);
                __m512d vw = _mm512_loadu_pd(w + i);
                __m512d sq = _mm512_mul_pd(va, va);
                __m512d prod = _mm512_mul_pd(sq, vw);
                sum_vec = _mm512_add_pd(sum_vec, prod);
            }
            double result = _mm512_reduce_add_pd(sum_vec);
            for (; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        AVX512_TARGET_ATTR
        float sum_sq_weighted_ps_avx512(const float *a, const float *w, std::size_t n) {
            std::size_t i = 0;
            __m512 sum_vec = _mm512_setzero_ps();
            for (; i + 15 < n; i += 16) {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vw = _mm512_loadu_ps(w + i);
                __m512 sq = _mm512_mul_ps(va, va);
                __m512 prod = _mm512_mul_ps(sq, vw);
                sum_vec = _mm512_add_ps(sum_vec, prod);
            }
            float result = _mm512_reduce_add_ps(sum_vec);
            for (; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        AVX512_TARGET_ATTR
        void interp_pd_avx512(const double *x, double x0, double y0, double x1, double y1, double inv_dx, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m512d x0_vec = _mm512_set1_pd(x0);
            const __m512d x1_vec = _mm512_set1_pd(x1);
            const __m512d y0_vec = _mm512_set1_pd(y0);
            const __m512d y1_vec = _mm512_set1_pd(y1);
            const __m512d slope_vec = _mm512_set1_pd((y1 - y0) * inv_dx);
            for (; i + 7 < n; i += 8) {
                __m512d elem = _mm512_loadu_pd(x + i);
                __mmask8 le_mask = _mm512_cmp_pd_mask(elem, x0_vec, _CMP_LE_OQ);
                __mmask8 ge_mask = _mm512_cmp_pd_mask(elem, x1_vec, _CMP_GE_OQ);
                __m512d t = _mm512_mul_pd(_mm512_sub_pd(elem, x0_vec), slope_vec);
                __m512d interp = _mm512_add_pd(y0_vec, t);
                // Blend: le -> y0, ge -> y1, else -> interp
                __m512d tmp = _mm512_mask_blend_pd(ge_mask, interp, y1_vec);
                __m512d vr = _mm512_mask_blend_pd(le_mask, tmp, y0_vec);
                _mm512_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                auto element = x[i];
                if (element <= x0) {
                    result[i] = y0;
                } else if (element >= x1) {
                    result[i] = y1;
                } else {
                    result[i] = y0 + (element - x0) * (y1 - y0) * inv_dx;
                }
            }
        }

        AVX512_TARGET_ATTR
        void interp_ps_avx512(const float *x, float x0, float y0, float x1, float y1, float inv_dx, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m512 x0_vec = _mm512_set1_ps(x0);
            const __m512 x1_vec = _mm512_set1_ps(x1);
            const __m512 y0_vec = _mm512_set1_ps(y0);
            const __m512 y1_vec = _mm512_set1_ps(y1);
            const __m512 slope_vec = _mm512_set1_ps((y1 - y0) * inv_dx);
            for (; i + 15 < n; i += 16) {
                __m512 elem = _mm512_loadu_ps(x + i);
                __mmask16 le_mask = _mm512_cmp_ps_mask(elem, x0_vec, _CMP_LE_OQ);
                __mmask16 ge_mask = _mm512_cmp_ps_mask(elem, x1_vec, _CMP_GE_OQ);
                __m512 t = _mm512_mul_ps(_mm512_sub_ps(elem, x0_vec), slope_vec);
                __m512 interp = _mm512_add_ps(y0_vec, t);
                // Blend: le -> y0, ge -> y1, else -> interp
                __m512 tmp = _mm512_mask_blend_ps(ge_mask, interp, y1_vec);
                __m512 vr = _mm512_mask_blend_ps(le_mask, tmp, y0_vec);
                _mm512_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                auto element = x[i];
                if (element <= x0) {
                    result[i] = y0;
                } else if (element >= x1) {
                    result[i] = y1;
                } else {
                    result[i] = y0 + (element - x0) * (y1 - y0) * inv_dx;
                }
            }
        }

    } // namespace internal
} // namespace np

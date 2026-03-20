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

/// AVX2 implementations of SIMD operations.

#include <cstddef>
#include <immintrin.h>

// Mark each function with the required target ISA and disable auto-vectorization
// so the compiler only emits AVX2 (not AVX512 or AMX) instructions in this translation unit.
#define AVX2_TARGET_ATTR __attribute__((target("avx2"), optimize("no-tree-vectorize")))

// Horizontal add helper for AVX2: sums all 4 doubles in a __m256d register
inline double horizontalAdd(__m256d a) {
    __m128d lo = _mm256_castpd256_pd128(a);
    __m128d hi = _mm256_extractf128_pd(a, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    return _mm_cvtsd_f64(sum);
}

// Horizontal add helper for AVX2: sums all 8 floats in a __m256 register
inline float horizontalAdd(__m256 a) {
    __m128 lo = _mm256_castps256_ps128(a);
    __m128 hi = _mm256_extractf128_ps(a, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

namespace np {
    namespace internal {

        AVX2_TARGET_ATTR
        void add_pd_avx2(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vr = _mm256_add_pd(va, vb);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        AVX2_TARGET_ATTR
        void sub_pd_avx2(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vr = _mm256_sub_pd(va, vb);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        AVX2_TARGET_ATTR
        void mul_pd_avx2(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vr = _mm256_mul_pd(va, vb);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        AVX2_TARGET_ATTR
        void div_pd_avx2(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d vr = _mm256_div_pd(va, vb);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        AVX2_TARGET_ATTR
        void add_ps_avx2(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vr = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        AVX2_TARGET_ATTR
        void sub_ps_avx2(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vr = _mm256_sub_ps(va, vb);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        AVX2_TARGET_ATTR
        void mul_ps_avx2(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vr = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        AVX2_TARGET_ATTR
        void div_ps_avx2(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 vr = _mm256_div_ps(va, vb);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        AVX2_TARGET_ATTR
        std::size_t count_lt_pd_avx2(const double *a, double threshold, std::size_t n) {
            std::size_t count = 0;
            __m256d thresh = _mm256_set1_pd(threshold);
            std::size_t i = 0;
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d mask = _mm256_cmp_pd(va, thresh, _CMP_LT_OQ);
                int mask_bits = _mm256_movemask_pd(mask);
                count += static_cast<std::size_t>(__builtin_popcount(mask_bits));
            }
            for (; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        AVX2_TARGET_ATTR
        std::size_t count_lt_ps_avx2(const float *a, float threshold, std::size_t n) {
            std::size_t count = 0;
            __m256 thresh = _mm256_set1_ps(threshold);
            std::size_t i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 mask = _mm256_cmp_ps(va, thresh, _CMP_LT_OQ);
                int mask_bits = _mm256_movemask_ps(mask);
                count += static_cast<std::size_t>(__builtin_popcount(mask_bits));
            }
            for (; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        AVX2_TARGET_ATTR
        void abs_pd_avx2(const double *a, double *result, std::size_t n) {
            std::size_t i = 0;
            // Clear sign bit using AND NOT with sign bit mask
            // -0.0 = 0x8000000000000000 (sign bit only in IEEE 754)
            __m256d sign_mask = _mm256_set1_pd(-0.0);
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vr = _mm256_andnot_pd(sign_mask, va);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        AVX2_TARGET_ATTR
        void abs_ps_avx2(const float *a, float *result, std::size_t n) {
            std::size_t i = 0;
            __m256 sign_mask = _mm256_set1_ps(-0.0f);
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vr = _mm256_andnot_ps(sign_mask, va);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        AVX2_TARGET_ATTR
        void where_tukey_pd_avx2(const double *a, double k, double *result, std::size_t n) {
            std::size_t i = 0;
            __m256d k_vec = _mm256_set1_pd(k);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d two_vec = _mm256_set1_pd(2.0);
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                // condition: a[i] <= k
                __m256d mask = _mm256_cmp_pd(va, k_vec, _CMP_LE_OQ);
                // Tukey bisquare: (k/a)*(2 - k/a) when a > k
                __m256d ratio = _mm256_div_pd(k_vec, va);
                __m256d neg = _mm256_mul_pd(ratio, _mm256_sub_pd(two_vec, ratio));
                // Blend: mask selects positive (1.0) when condition true, negative when false
                __m256d vr = _mm256_blendv_pd(neg, one_vec, mask);
                _mm256_storeu_pd(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0 : (2.0 * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        AVX2_TARGET_ATTR
        void where_tukey_ps_avx2(const float *a, float k, float *result, std::size_t n) {
            std::size_t i = 0;
            __m256 k_vec = _mm256_set1_ps(k);
            __m256 one_vec = _mm256_set1_ps(1.0f);
            __m256 two_vec = _mm256_set1_ps(2.0f);
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 mask = _mm256_cmp_ps(va, k_vec, _CMP_LE_OQ);
                __m256 ratio = _mm256_div_ps(k_vec, va);
                __m256 neg = _mm256_mul_ps(ratio, _mm256_sub_ps(two_vec, ratio));
                __m256 vr = _mm256_blendv_ps(neg, one_vec, mask);
                _mm256_storeu_ps(result + i, vr);
            }
            for (; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0f : (2.0f * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        AVX2_TARGET_ATTR
        void abs_sub_pd_avx2(const double *a, const double *b, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m256d sign_mask = _mm256_set1_pd(-0.0);
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vb = _mm256_loadu_pd(b + i);
                __m256d diff = _mm256_sub_pd(va, vb);
                __m256d abs_diff = _mm256_andnot_pd(sign_mask, diff);
                _mm256_storeu_pd(result + i, abs_diff);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        AVX2_TARGET_ATTR
        void abs_sub_ps_avx2(const float *a, const float *b, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m256 sign_mask = _mm256_set1_ps(-0.0f);
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vb = _mm256_loadu_ps(b + i);
                __m256 diff = _mm256_sub_ps(va, vb);
                __m256 abs_diff = _mm256_andnot_ps(sign_mask, diff);
                _mm256_storeu_ps(result + i, abs_diff);
            }
            for (; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        AVX2_TARGET_ATTR
        double sum_sq_weighted_pd_avx2(const double *a, const double *w, std::size_t n) {
            std::size_t i = 0;
            __m256d sum_vec = _mm256_setzero_pd();
            for (; i + 3 < n; i += 4) {
                __m256d va = _mm256_loadu_pd(a + i);
                __m256d vw = _mm256_loadu_pd(w + i);
                __m256d sq = _mm256_mul_pd(va, va);
                __m256d prod = _mm256_mul_pd(sq, vw);
                sum_vec = _mm256_add_pd(sum_vec, prod);
            }
            double result = horizontalAdd(sum_vec);
            for (; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        AVX2_TARGET_ATTR
        float sum_sq_weighted_ps_avx2(const float *a, const float *w, std::size_t n) {
            std::size_t i = 0;
            __m256 sum_vec = _mm256_setzero_ps();
            for (; i + 7 < n; i += 8) {
                __m256 va = _mm256_loadu_ps(a + i);
                __m256 vw = _mm256_loadu_ps(w + i);
                __m256 sq = _mm256_mul_ps(va, va);
                __m256 prod = _mm256_mul_ps(sq, vw);
                sum_vec = _mm256_add_ps(sum_vec, prod);
            }
            float result = horizontalAdd(sum_vec);
            for (; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        AVX2_TARGET_ATTR
        void interp_pd_avx2(const double *x, double x0, double y0, double x1, double y1, double inv_dx, double *result, std::size_t n) {
            std::size_t i = 0;
            const __m256d x0_vec = _mm256_set1_pd(x0);
            const __m256d x1_vec = _mm256_set1_pd(x1);
            const __m256d y0_vec = _mm256_set1_pd(y0);
            const __m256d y1_vec = _mm256_set1_pd(y1);
            const __m256d slope_vec = _mm256_set1_pd((y1 - y0) * inv_dx);
            for (; i + 3 < n; i += 4) {
                __m256d elem = _mm256_loadu_pd(x + i);
                __m256d le_mask = _mm256_cmp_pd(elem, x0_vec, _CMP_LE_OQ);
                __m256d ge_mask = _mm256_cmp_pd(elem, x1_vec, _CMP_GE_OQ);
                __m256d t = _mm256_mul_pd(_mm256_sub_pd(elem, x0_vec), slope_vec);
                __m256d interp = _mm256_add_pd(y0_vec, t);
                // Blend: le -> y0, ge -> y1, else -> interp
                __m256d tmp = _mm256_blendv_pd(interp, y1_vec, ge_mask);
                __m256d vr = _mm256_blendv_pd(tmp, y0_vec, le_mask);
                _mm256_storeu_pd(result + i, vr);
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

        AVX2_TARGET_ATTR
        void interp_ps_avx2(const float *x, float x0, float y0, float x1, float y1, float inv_dx, float *result, std::size_t n) {
            std::size_t i = 0;
            const __m256 x0_vec = _mm256_set1_ps(x0);
            const __m256 x1_vec = _mm256_set1_ps(x1);
            const __m256 y0_vec = _mm256_set1_ps(y0);
            const __m256 y1_vec = _mm256_set1_ps(y1);
            const __m256 slope_vec = _mm256_set1_ps((y1 - y0) * inv_dx);
            for (; i + 7 < n; i += 8) {
                __m256 elem = _mm256_loadu_ps(x + i);
                __m256 le_mask = _mm256_cmp_ps(elem, x0_vec, _CMP_LE_OQ);
                __m256 ge_mask = _mm256_cmp_ps(elem, x1_vec, _CMP_GE_OQ);
                __m256 t = _mm256_mul_ps(_mm256_sub_ps(elem, x0_vec), slope_vec);
                __m256 interp = _mm256_add_ps(y0_vec, t);
                __m256 tmp = _mm256_blendv_ps(interp, y1_vec, ge_mask);
                __m256 vr = _mm256_blendv_ps(tmp, y0_vec, le_mask);
                _mm256_storeu_ps(result + i, vr);
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

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

/// Scalar (no-SIMD) implementations of SIMD operations.
/// These are compiled without any -march flags and serve as the
/// portable fallback when no SIMD is available.

#include <cstddef>
#include <cmath>

namespace np {
    namespace internal {

        void add_pd_scalar(const double *a, const double *b, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        void sub_pd_scalar(const double *a, const double *b, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        void mul_pd_scalar(const double *a, const double *b, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        void div_pd_scalar(const double *a, const double *b, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        void add_ps_scalar(const float *a, const float *b, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] + b[i];
            }
        }

        void sub_ps_scalar(const float *a, const float *b, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] - b[i];
            }
        }

        void mul_ps_scalar(const float *a, const float *b, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] * b[i];
            }
        }

        void div_ps_scalar(const float *a, const float *b, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = a[i] / b[i];
            }
        }

        std::size_t count_lt_pd_scalar(const double *a, double threshold, std::size_t n) {
            std::size_t count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        std::size_t count_lt_ps_scalar(const float *a, float threshold, std::size_t n) {
            std::size_t count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (a[i] < threshold) ++count;
            }
            return count;
        }

        void abs_pd_scalar(const double *a, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        void abs_ps_scalar(const float *a, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = std::abs(a[i]);
            }
        }

        void where_tukey_pd_scalar(const double *a, double k, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0 : (2.0 * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        void where_tukey_ps_scalar(const float *a, float k, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = (a[i] <= k) ? 1.0f : (2.0f * k / a[i] - k * k / (a[i] * a[i]));
            }
        }

        void abs_sub_pd_scalar(const double *a, const double *b, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        void abs_sub_ps_scalar(const float *a, const float *b, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = std::abs(a[i] - b[i]);
            }
        }

        double sum_sq_weighted_pd_scalar(const double *a, const double *w, std::size_t n) {
            double result = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        float sum_sq_weighted_ps_scalar(const float *a, const float *w, std::size_t n) {
            float result = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                result += a[i] * a[i] * w[i];
            }
            return result;
        }

        void interp_pd_scalar(const double *x, double x0, double y0, double x1, double y1, double inv_dx, double *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
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

        void interp_ps_scalar(const float *x, float x0, float y0, float x1, float y1, float inv_dx, float *result, std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
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

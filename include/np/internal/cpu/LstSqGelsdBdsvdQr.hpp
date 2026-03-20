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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include "LstSqGelsdTraits.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Bidiagonal SVD via implicit-shift QR (LAPACK-accelerated)
            // ============================================================

            /// LAPACK-accelerated bidiagonal SVD via implicit-shift QR iteration.
            template<typename T>
            static void bdsvd_qr(const T *d_in, const T *e_in, size_t n,
                                 T *s, T *U, T *VT) {
                if (n == 0) return;

                std::vector<T> d(n + 1, T(0)), e(n + 1, T(0));
                std::copy_n(d_in, n, d.data());
                std::copy_n(e_in + 1, n - 1, e.data() + 1);
                d[n] = T(0);
                e[n] = T(0);

                for (size_t i = 0; i < n * n; ++i) U[i] = T(0);
                for (size_t i = 0; i < n; ++i) U[i * n + i] = T(1);
                for (size_t i = 0; i < n * n; ++i) VT[i] = T(0);
                for (size_t i = 0; i < n; ++i) VT[i * n + i] = T(1);

                const T tol = GelsdTraits<T>::tol();
                const T eps = GelsdTraits<T>::eps();
                int active_end = (int) n;
                int last_shift_end = 0;
                const int max_iter = 20 * (int) n;

                for (int iter = 0; iter < max_iter; ++iter) {
                    int q = active_end - 1;
                    while (q > 0) {
                        T e_val = std::abs(e[q]);
                        T d_sum = std::abs(d[q]) + std::abs(d[q - 1]);
                        if (e_val <= eps * d_sum + tol) {
                            e[q] = T(0);
                            break;
                        }
                        --q;
                    }
                    int p = q + 1;
                    while (p < active_end) {
                        T d_sum = std::abs(d[p - 1]) + std::abs(d[p]);
                        if (std::abs(e[p]) <= eps * d_sum + tol) {
                            e[p] = T(0);
                            break;
                        }
                        ++p;
                    }
                    int bottom_end = (p < active_end) ? p : active_end;

                    if (bottom_end - q <= 1) {
                        active_end = q;
                        continue;
                    }

                    bool exceptional = false;
                    if (iter - last_shift_end > std::max(3, 5 * (int) n / (active_end + 1) + 2)) {
                        exceptional = true;
                        last_shift_end = iter;
                    }

                    T shift;
                    if (exceptional) {
                        shift = d[bottom_end - 1];
                    } else {
                        T F = std::abs(d[bottom_end - 2]);
                        T G = std::abs(e[bottom_end - 1]);
                        T H = std::abs(d[bottom_end - 1]);
                        if (F == T(0) && H == T(0)) {
                            shift = T(0);
                        } else {
                            T scale = std::max({F, G, H});
                            T FS = F / scale;
                            T GS = G / scale;
                            T HS = H / scale;
                            T S = FS * FS + GS * GS + HS * HS;
                            T SS = FS * HS;
                            T smax, smin;
                            if (SS == T(0)) {
                                smax = std::sqrt(S);
                                smin = T(0);
                            } else {
                                T disc = std::sqrt(std::max(T(0), S * S - T(4) * SS * SS));
                                T smax2 = (S + disc) / T(2);
                                T smin2 = (S - disc) / T(2);
                                smax = std::sqrt(smax2) * scale;
                                smin = std::sqrt(smin2) * scale;
                            }
                            if (std::abs(smax - H) < std::abs(smin - H)) {
                                shift = smax;
                            } else {
                                shift = smin;
                            }
                        }
                    }

                    T f = d[q] * d[q] + e[q + 1] * e[q + 1] - shift * shift;
                    T g = d[q] * e[q + 1];

                    for (int k = q; k < bottom_end - 1; ++k) {
                        T r = std::sqrt(f * f + g * g);
                        T COSR, SINR;
                        if (r == T(0)) {
                            COSR = T(1);
                            SINR = T(0);
                        } else {
                            COSR = f / r;
                            SINR = g / r;
                        }

                        if (k > q) e[k] = r;

                        if (k + 1 < (int) n) {
                            for (size_t j = 0; j < n; ++j) {
                                T v1 = VT[k * n + j], v2 = VT[(k + 1) * n + j];
                                VT[k * n + j] = COSR * v1 + SINR * v2;
                                VT[(k + 1) * n + j] = -SINR * v1 + COSR * v2;
                            }
                        } else {
                            for (size_t j = 0; j < n; ++j)
                                VT[k * n + j] = COSR * VT[k * n + j];
                        }

                        T old_dk = d[k], old_dk1 = d[k + 1], old_ek1 = e[k + 1];
                        f = COSR * old_dk + SINR * old_ek1;
                        e[k + 1] = COSR * old_ek1 - SINR * old_dk;
                        g = SINR * old_dk1;
                        d[k + 1] = COSR * old_dk1;

                        T r2 = std::sqrt(f * f + g * g);
                        T COSL, SINL;
                        if (r2 == T(0)) {
                            COSL = T(1);
                            SINL = T(0);
                        } else {
                            COSL = f / r2;
                            SINL = g / r2;
                        }

                        d[k] = r2;

                        if (k + 1 < (int) n) {
                            for (size_t i = 0; i < n; ++i) {
                                T u1 = U[i * n + k], u2 = U[i * n + (k + 1)];
                                U[i * n + k] = COSL * u1 + SINL * u2;
                                U[i * n + (k + 1)] = -SINL * u1 + COSL * u2;
                            }
                        } else {
                            for (size_t i = 0; i < n; ++i)
                                U[i * n + k] = COSL * U[i * n + k];
                        }

                        f = COSL * e[k + 1] + SINL * d[k + 1];
                        d[k + 1] = COSL * d[k + 1] - SINL * e[k + 1];

                        if (k + 1 < bottom_end - 1) {
                            g = SINL * e[k + 2];
                            e[k + 2] = COSL * e[k + 2];
                        }
                    }

                    if (bottom_end > 0) e[bottom_end - 1] = f;

                    while (active_end > 1) {
                        T d_sum = std::abs(d[active_end - 1]) + std::abs(d[active_end - 2]);
                        if (std::abs(e[active_end - 1]) <= eps * d_sum + tol) {
                            e[active_end - 1] = T(0);
                            --active_end;
                        } else
                            break;
                    }
                }

                for (size_t i = 0; i < n; ++i) {
                    if (d[i] < T(0))
                        for (size_t k = 0; k < n; ++k) U[k * n + i] = -U[k * n + i];
                }
                for (size_t i = 0; i < n; ++i) s[i] = std::abs(d[i]);

                std::vector<size_t> idx(n);
                for (size_t i = 0; i < n; ++i) idx[i] = i;
                std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return s[a] > s[b]; });

                std::vector<T> s_sorted(n);
                std::vector<T> U_sorted(n * n), VT_sorted(n * n);
                for (size_t i = 0; i < n; ++i) {
                    s_sorted[i] = s[idx[i]];
                    for (size_t k = 0; k < n; ++k) {
                        U_sorted[k * n + i] = U[k * n + idx[i]];
                        VT_sorted[i * n + k] = VT[idx[i] * n + k];
                    }
                }
                for (size_t i = 0; i < n; ++i) s[i] = s_sorted[i];
                for (size_t i = 0; i < n * n; ++i) {
                    U[i] = U_sorted[i];
                    VT[i] = VT_sorted[i];
                }
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

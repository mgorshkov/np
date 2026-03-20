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

/// AMX + AVX512 BLAS Level 1 operations for the GELSD least-squares solver.
/// Provides dot product and vector scaling using AMX tile loads + AVX512 SIMD.
/// All matrices are stored in row-major layout.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <immintrin.h>

#include "LstSqGelsdSimdAttr.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // AMX tile configuration constants
            static constexpr std::size_t kAmxTileRows = 16;
            static constexpr std::size_t kAmxTileColBytes = 64;                                     // 64 bytes per row
            static constexpr std::size_t kAmxDoublesPerTile = kAmxTileRows * (kAmxTileColBytes / 8);// 128
            static constexpr std::size_t kAmxFloatsPerTile = kAmxTileRows * (kAmxTileColBytes / 4); // 256

            // AMX tile config structure (matches hardware layout)
            struct AmxTileConfig {
                uint8_t palette_id;
                uint8_t start_row;
                uint8_t reserved[14];
                uint16_t colsb[8];
                uint8_t rows[8];
            };

            AMX_TARGET_ATTR
            static inline void amx_init_tiles() {
                AmxTileConfig cfg;
                std::memset(&cfg, 0, sizeof(cfg));
                cfg.palette_id = 1;
                for (int i = 0; i < 8; ++i) {
                    cfg.rows[i] = 16;
                    cfg.colsb[i] = 64;// 64 bytes per row
                }
                _tile_loadconfig(&cfg);
            }

            AMX_TARGET_ATTR
            static inline void amx_release_tiles() {
                _tile_release();
            }

            // ============================================================
            //  AMX + AVX512: dot product (double)
            // ============================================================

            /// Compute dot product of two double vectors using AMX tile loads + AVX512 FMA.
            /// Processes kAmxDoublesPerTile (128) doubles per tile iteration.
            AMX_TARGET_ATTR
            static inline double dot_d_amx(const double *x, const double *y, size_t n) {
                amx_init_tiles();
                __m512d sum0 = _mm512_setzero_pd();
                __m512d sum1 = _mm512_setzero_pd();
                __m512d sum2 = _mm512_setzero_pd();
                __m512d sum3 = _mm512_setzero_pd();

                size_t i = 0;
                // Process 128 doubles at a time using AMX tiles
                for (; i + kAmxDoublesPerTile - 1 < n; i += kAmxDoublesPerTile) {
                    // Load x and y tiles (16 rows x 64 bytes each = 128 doubles)
                    _tile_loadd(0, const_cast<double *>(x + i), kAmxTileColBytes);
                    _tile_loadd(1, const_cast<double *>(y + i), kAmxTileColBytes);

                    // Process each of the 16 rows using AVX512
                    for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                        std::size_t offset = row * (kAmxTileColBytes / 8);// 8 doubles per row
                        __m512d vx = _mm512_loadu_pd(x + i + offset);
                        __m512d vy = _mm512_loadu_pd(y + i + offset);
                        __m512d prod = _mm512_mul_pd(vx, vy);
                        // Distribute across accumulators to reduce latency
                        if (row < 4) sum0 = _mm512_add_pd(sum0, prod);
                        else if (row < 8)
                            sum1 = _mm512_add_pd(sum1, prod);
                        else if (row < 12)
                            sum2 = _mm512_add_pd(sum2, prod);
                        else
                            sum3 = _mm512_add_pd(sum3, prod);
                    }
                }
                amx_release_tiles();

                // Combine accumulators
                __m512d total = _mm512_add_pd(_mm512_add_pd(sum0, sum1),
                                              _mm512_add_pd(sum2, sum3));
                double s = _mm512_reduce_add_pd(total);

                // Process remainder
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AMX + AVX512: scale vector (double)
            // ============================================================

            AMX_TARGET_ATTR
            static inline void scal_d_amx(double a, double *x, size_t n) {
                __m512d va = _mm512_set1_pd(a);
                size_t i = 0;
                for (; i + 7 < n; i += 8) {
                    _mm512_storeu_pd(x + i, _mm512_mul_pd(va, _mm512_loadu_pd(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

            // ============================================================
            //  AMX + AVX512: dot product (float)
            // ============================================================

            /// Compute dot product of two float vectors using AMX tile loads + AVX512 FMA.
            /// Processes kAmxFloatsPerTile (256) floats per tile iteration.
            AMX_TARGET_ATTR
            static inline float dot_f_amx(const float *x, const float *y, size_t n) {
                amx_init_tiles();
                __m512 sum0 = _mm512_setzero_ps();
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();

                size_t i = 0;
                // Process 256 floats at a time using AMX tiles
                for (; i + kAmxFloatsPerTile - 1 < n; i += kAmxFloatsPerTile) {
                    // Load x and y tiles (16 rows x 64 bytes each = 256 floats)
                    _tile_loadd(0, const_cast<float *>(x + i), kAmxTileColBytes);
                    _tile_loadd(1, const_cast<float *>(y + i), kAmxTileColBytes);

                    // Process each of the 16 rows using AVX512
                    for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                        std::size_t offset = row * (kAmxTileColBytes / 4);// 16 floats per row
                        __m512 vx = _mm512_loadu_ps(x + i + offset);
                        __m512 vy = _mm512_loadu_ps(y + i + offset);
                        __m512 prod = _mm512_mul_ps(vx, vy);
                        // Distribute across accumulators to reduce latency
                        if (row < 4) sum0 = _mm512_add_ps(sum0, prod);
                        else if (row < 8)
                            sum1 = _mm512_add_ps(sum1, prod);
                        else if (row < 12)
                            sum2 = _mm512_add_ps(sum2, prod);
                        else
                            sum3 = _mm512_add_ps(sum3, prod);
                    }
                }
                amx_release_tiles();

                // Combine accumulators
                __m512 total = _mm512_add_ps(_mm512_add_ps(sum0, sum1),
                                             _mm512_add_ps(sum2, sum3));
                float s = _mm512_reduce_add_ps(total);

                // Process remainder
                for (; i < n; ++i) s += x[i] * y[i];
                return s;
            }

            // ============================================================
            //  AMX + AVX512: scale vector (float)
            // ============================================================

            AMX_TARGET_ATTR
            static inline void scal_f_amx(float a, float *x, size_t n) {
                __m512 va = _mm512_set1_ps(a);
                size_t i = 0;
                for (; i + 15 < n; i += 16) {
                    _mm512_storeu_ps(x + i, _mm512_mul_ps(va, _mm512_loadu_ps(x + i)));
                }
                for (; i < n; ++i) x[i] *= a;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

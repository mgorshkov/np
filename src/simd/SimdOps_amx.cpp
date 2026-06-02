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

/// AMX (Intel Advanced Matrix Extensions) implementations of SIMD operations.
///
/// AMX doesn't have element-wise SIMD operations like AVX.
/// The strategy is to use AMX tiles as "wide load/store" units:
/// - Load 128 doubles (or 256 floats) at once into a tile
/// - Process each row using AVX512 intrinsics
/// - Store results back
/// This gives 16x wider loads than AVX512 (1024 bytes vs 64 bytes for zmm).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

// Mark each function with the required target ISA and disable auto-vectorization
// so the compiler only emits AMX/AVX512 instructions where explicitly requested.
#define AMX_TARGET_ATTR __attribute__((target("amx-tile,amx-int8,amx-bf16,avx512f"), optimize("no-tree-vectorize")))

namespace np {
    namespace internal {

        // Forward declarations of AVX512 interp functions used for remainder handling
        void interp_pd_avx512(const double *, double, double, double, double, double, double *, std::size_t);
        void interp_ps_avx512(const float *, float, float, float, float, float, float *, std::size_t);

        // AMX tile configuration constants
        static constexpr std::size_t kAmxTileRows = 16;
        static constexpr std::size_t kAmxTileColBytes = 64; // 64 bytes per row
        static constexpr std::size_t kAmxDoublesPerTile = kAmxTileRows * (kAmxTileColBytes / 8); // 128
        static constexpr std::size_t kAmxFloatsPerTile = kAmxTileRows * (kAmxTileColBytes / 4);  // 256

        // AMX tile config structure (matches hardware layout)
        struct AmxTileConfig {
            uint8_t palette_id;
            uint8_t start_row;
            uint8_t reserved[14];
            uint16_t colsb[8];
            uint8_t rows[8];
        };

        AMX_TARGET_ATTR
        static void amx_init_tiles() {
            AmxTileConfig cfg;
            std::memset(&cfg, 0, sizeof(cfg));
            cfg.palette_id = 1;
            for (int i = 0; i < 8; ++i) {
                cfg.rows[i] = 16;
                cfg.colsb[i] = 64; // 64 bytes per row
            }
            _tile_loadconfig(&cfg);
        }

        AMX_TARGET_ATTR
        static void amx_release_tiles() {
            _tile_release();
        }

        // ---- AMX + AVX512: process tiles of double data ----

        AMX_TARGET_ATTR
        void amx_process_pd_impl(const double *a, const double *b, double *result, std::size_t n,
                                 void (*op)(const double *, const double *, double *, std::size_t)) {
            amx_init_tiles();
            std::size_t i = 0;
            for (; i + kAmxDoublesPerTile - 1 < n; i += kAmxDoublesPerTile) {
                // Load left and right tiles
                _tile_loadd(0, a + i, kAmxTileColBytes);
                _tile_loadd(1, b + i, kAmxTileColBytes);
                // Process each row of the tiles using AVX512
                for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                    std::size_t offset = row * (kAmxTileColBytes / 8);
                    __m512d va = _mm512_loadu_pd(a + i + offset);
                    __m512d vb = _mm512_loadu_pd(b + i + offset);
                    __m512d vr = _mm512_add_pd(va, vb);
                    _mm512_storeu_pd(result + i + offset, vr);
                }
            }
            amx_release_tiles();
            // Process remainder
            if (i < n) {
                op(a + i, b + i, result + i, n - i);
            }
        }

        // ---- AMX + AVX512: process tiles of float data ----

        AMX_TARGET_ATTR
        void amx_process_ps_impl(const float *a, const float *b, float *result, std::size_t n,
                                 void (*op)(const float *, const float *, float *, std::size_t)) {
            amx_init_tiles();
            std::size_t i = 0;
            for (; i + kAmxFloatsPerTile - 1 < n; i += kAmxFloatsPerTile) {
                // Load left and right tiles
                _tile_loadd(0, a + i, kAmxTileColBytes);
                _tile_loadd(1, b + i, kAmxTileColBytes);
                // Process each row of the tiles using AVX512
                for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                    std::size_t offset = row * (kAmxTileColBytes / 4);
                    __m512 va = _mm512_loadu_ps(a + i + offset);
                    __m512 vb = _mm512_loadu_ps(b + i + offset);
                    __m512 vr = _mm512_add_ps(va, vb);
                    _mm512_storeu_ps(result + i + offset, vr);
                }
            }
            amx_release_tiles();
            // Process remainder
            if (i < n) {
                op(a + i, b + i, result + i, n - i);
            }
        }

        // ---- AMX + AVX512: interp for double data ----

        AMX_TARGET_ATTR
        void interp_pd_amx(const double *x, double x0, double y0, double x1, double y1, double inv_dx, double *result, std::size_t n) {
            amx_init_tiles();
            const __m512d x0_vec = _mm512_set1_pd(x0);
            const __m512d x1_vec = _mm512_set1_pd(x1);
            const __m512d y0_vec = _mm512_set1_pd(y0);
            const __m512d y1_vec = _mm512_set1_pd(y1);
            const __m512d slope_vec = _mm512_set1_pd((y1 - y0) * inv_dx);
            std::size_t i = 0;
            for (; i + kAmxDoublesPerTile - 1 < n; i += kAmxDoublesPerTile) {
                _tile_loadd(0, x + i, kAmxTileColBytes);
                // Process each row of the tile using AVX512 interp logic
                for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                    std::size_t offset = row * (kAmxTileColBytes / 8);
                    __m512d elem = _mm512_loadu_pd(x + i + offset);
                    __mmask8 le_mask = _mm512_cmp_pd_mask(elem, x0_vec, _CMP_LE_OQ);
                    __mmask8 ge_mask = _mm512_cmp_pd_mask(elem, x1_vec, _CMP_GE_OQ);
                    __m512d t = _mm512_mul_pd(_mm512_sub_pd(elem, x0_vec), slope_vec);
                    __m512d interp = _mm512_add_pd(y0_vec, t);
                    __m512d tmp = _mm512_mask_blend_pd(ge_mask, interp, y1_vec);
                    __m512d vr = _mm512_mask_blend_pd(le_mask, tmp, y0_vec);
                    _mm512_storeu_pd(result + i + offset, vr);
                }
            }
            amx_release_tiles();
            // Process remainder using AVX512 interp
            if (i < n) {
                interp_pd_avx512(x + i, x0, y0, x1, y1, inv_dx, result + i, n - i);
            }
        }

        // ---- AMX + AVX512: interp for float data ----

        AMX_TARGET_ATTR
        void interp_ps_amx(const float *x, float x0, float y0, float x1, float y1, float inv_dx, float *result, std::size_t n) {
            amx_init_tiles();
            const __m512 x0_vec = _mm512_set1_ps(x0);
            const __m512 x1_vec = _mm512_set1_ps(x1);
            const __m512 y0_vec = _mm512_set1_ps(y0);
            const __m512 y1_vec = _mm512_set1_ps(y1);
            const __m512 slope_vec = _mm512_set1_ps((y1 - y0) * inv_dx);
            std::size_t i = 0;
            for (; i + kAmxFloatsPerTile - 1 < n; i += kAmxFloatsPerTile) {
                _tile_loadd(0, x + i, kAmxTileColBytes);
                // Process each row of the tile using AVX512 interp logic
                for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                    std::size_t offset = row * (kAmxTileColBytes / 4);
                    __m512 elem = _mm512_loadu_ps(x + i + offset);
                    __mmask16 le_mask = _mm512_cmp_ps_mask(elem, x0_vec, _CMP_LE_OQ);
                    __mmask16 ge_mask = _mm512_cmp_ps_mask(elem, x1_vec, _CMP_GE_OQ);
                    __m512 t = _mm512_mul_ps(_mm512_sub_ps(elem, x0_vec), slope_vec);
                    __m512 interp = _mm512_add_ps(y0_vec, t);
                    __m512 tmp = _mm512_mask_blend_ps(ge_mask, interp, y1_vec);
                    __m512 vr = _mm512_mask_blend_ps(le_mask, tmp, y0_vec);
                    _mm512_storeu_ps(result + i + offset, vr);
                }
            }
            amx_release_tiles();
            // Process remainder using AVX512 interp
            if (i < n) {
                interp_ps_avx512(x + i, x0, y0, x1, y1, inv_dx, result + i, n - i);
            }
        }

        // ---- Matrix-vector dot product (1D · 2D) with AMX tiles ----

        /// AMX + AVX512 implementation of y = x * W^T where W is (rows x cols).
        /// Uses AMX tiles to load 16 rows of W at once (256 floats per tile row),
        /// then processes each column using AVX512 FMA for the inner dot product.
        /// The tile acts as a wide load unit (16x wider than AVX512) to maximize
        /// memory bandwidth utilization.
        AMX_TARGET_ATTR
        void dot_1d_2d_ps_amx(const float *x, const float *W, std::size_t rows, std::size_t cols, float *result) {
            amx_init_tiles();
            // Zero out result first
            std::memset(result, 0, cols * sizeof(float));

            std::size_t i = 0;
            // Process rows in tiles of 16
            for (; i + kAmxTileRows - 1 < rows; i += kAmxTileRows) {
                // Load 16 rows of W into tile 0 (16 rows x 64 bytes per row = 256 floats per row stride)
                // Each tile row stores cols floats, but only 64 bytes (16 floats) are contiguous
                // The tile stride is cols * sizeof(float) bytes per row
                _tile_loadd(0, W + i * cols, cols * sizeof(float));

                // Process columns in batches of 16 using AVX512
                // For each column batch, we compute the dot product of x[i..i+15]
                // with W[(i+row)*cols + j..j+15] for each row in the tile
                std::size_t j = 0;
                for (; j + 15 < cols; j += 16) {
                    __m512 sum_vec = _mm512_setzero_ps();
                    // For each row in the tile, load 16 floats and FMA with x[row]
                    for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                        __m512 x_brcst = _mm512_set1_ps(x[i + row]);
                        __m512 w_vec = _mm512_loadu_ps(W + (i + row) * cols + j);
                        sum_vec = _mm512_fmadd_ps(x_brcst, w_vec, sum_vec);
                    }
                    // Accumulate into result
                    __m512 r_vec = _mm512_loadu_ps(result + j);
                    r_vec = _mm512_add_ps(r_vec, sum_vec);
                    _mm512_storeu_ps(result + j, r_vec);
                }
                // Handle remaining columns (less than 16)
                for (; j < cols; ++j) {
                    float sum = 0.0f;
                    for (std::size_t row = 0; row < kAmxTileRows; ++row) {
                        sum += x[i + row] * W[(i + row) * cols + j];
                    }
                    result[j] += sum;
                }
            }

            amx_release_tiles();

            // Process remaining rows (less than a full tile) using AVX512
            if (i < rows) {
                std::size_t remaining = rows - i;
                // Process columns in batches of 16
                std::size_t j = 0;
                for (; j + 15 < cols; j += 16) {
                    __m512 sum_vec = _mm512_setzero_ps();
                    for (std::size_t r = 0; r < remaining; ++r) {
                        __m512 x_brcst = _mm512_set1_ps(x[i + r]);
                        __m512 w_vec = _mm512_loadu_ps(W + (i + r) * cols + j);
                        sum_vec = _mm512_fmadd_ps(x_brcst, w_vec, sum_vec);
                    }
                    __m512 r_vec = _mm512_loadu_ps(result + j);
                    r_vec = _mm512_add_ps(r_vec, sum_vec);
                    _mm512_storeu_ps(result + j, r_vec);
                }
                // Handle remaining columns
                for (; j < cols; ++j) {
                    float sum = 0.0f;
                    for (std::size_t r = 0; r < remaining; ++r) {
                        sum += x[i + r] * W[(i + r) * cols + j];
                    }
                    result[j] += sum;
                }
            }
        }

    } // namespace internal
} // namespace np

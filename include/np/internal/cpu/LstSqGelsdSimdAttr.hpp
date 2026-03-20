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

/// Common SIMD target attribute macros for the GELSD solver.
///
/// Each macro marks a function with the required target ISA and disables
/// auto-vectorization so the compiler only emits the specified SIMD instructions.
///
/// These are used by the LstSqGelsd*_avx2.hpp, LstSqGelsd*_avx512.hpp,
/// and LstSqGelsd*_amx.hpp headers.

#pragma once

/// AVX2 target attribute: enables AVX2 + FMA instructions.
#define AVX2_TARGET_ATTR __attribute__((target("avx2,fma"), optimize("no-tree-vectorize")))

/// AVX-512 target attribute: enables AVX-512F, AVX-512DQ, AVX-512BW, AVX-512VL.
#define AVX512_TARGET_ATTR __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl"), optimize("no-tree-vectorize")))

/// AMX target attribute: enables AMX tile, AMX INT8, AMX BF16, plus required AVX-512 features.
#ifndef AMX_TARGET_ATTR
#define AMX_TARGET_ATTR __attribute__((target("amx-tile,amx-int8,amx-bf16,avx512f,avx512dq,avx512bw,avx512vl"), optimize("no-tree-vectorize")))
#endif

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

#include <cstdint>
#include <cstring>

namespace np {
    namespace internal {

        /// SIMD instruction set levels in ascending order.
        /// The runtime-detected maximum level is capped by the compile-time
        /// ENABLE_AVX/ENABLE_AVX2/ENABLE_AVX512/ENABLE_AMX macros.
        enum class SimdLevel : int {
            SCALAR = 0,///< No SIMD (scalar fallback)
            SSE2 = 1,  ///< SSE2
            SSE3 = 2,  ///< SSE3
            AVX = 3,   ///< AVX (128-bit)
            AVX2 = 4,  ///< AVX2 (256-bit)
            AVX512 = 5,///< AVX-512
            AMX = 6    ///< AMX (Intel Advanced Matrix Extensions)
        };

        /// Check if SSE2 is supported at runtime using CPUID.
        inline bool cpu_supports_sse2() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(1), "c"(0));
            // EDX bit 26: SSE2
            return (edx >> 26) & 1;
#else
            return false;
#endif
        }

        /// Check if SSE3 is supported at runtime using CPUID.
        inline bool cpu_supports_sse3() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(1), "c"(0));
            // ECX bit 0: SSE3
            return ecx & 1;
#else
            return false;
#endif
        }

        /// Check if AVX is supported at runtime using CPUID.
        inline bool cpu_supports_avx() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(1), "c"(0));
            // ECX bit 28: AVX
            bool has_avx = (ecx >> 28) & 1;
            // Also need OSXSAVE (ECX bit 27) and XSAVE enabled for YMM state
            bool has_osxsave = (ecx >> 27) & 1;
            if (!has_avx || !has_osxsave) return false;
            // Check that the OS saves YMM state properly
            uint64_t xcr0 = 0;
#if defined(__GNUC__) || defined(__clang__)
            __asm__ volatile("xgetbv"
                             : "=a"(eax), "=d"(edx)
                             : "c"(0));
            xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
#endif
            // XCR0 bit 1: SSE state, bit 2: AVX state (YMM)
            return (xcr0 & 0x6) == 0x6;
#else
            return false;
#endif
        }

        /// Check if AVX2 is supported at runtime using CPUID.
        inline bool cpu_supports_avx2() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            if (!cpu_supports_avx()) return false;
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));
            // EBX bit 5: AVX2
            return (ebx >> 5) & 1;
#else
            return false;
#endif
        }

        /// Check if AVX-512F (foundation) is supported at runtime using CPUID.
        inline bool cpu_supports_avx512() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            if (!cpu_supports_avx()) return false;
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));
            // EBX bit 16: AVX512F
            bool has_avx512f = (ebx >> 16) & 1;
            if (!has_avx512f) return false;
            // Check that the OS saves ZMM state properly
            uint64_t xcr0 = 0;
#if defined(__GNUC__) || defined(__clang__)
            __asm__ volatile("xgetbv"
                             : "=a"(eax), "=d"(edx)
                             : "c"(0));
            xcr0 = (static_cast<uint64_t>(edx) << 32) | eax;
#endif
            // XCR0 bit 1: SSE, bit 2: AVX (YMM), bit 5: OPMASK, bit 6: ZMM_Hi256
            return (xcr0 & 0xE6) == 0xE6;
#else
            return false;
#endif
        }

        /// Check if AMX is supported at runtime using CPUID.
        inline bool cpu_supports_amx() {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(7), "c"(0));
            // EDX bit 24: AMX-TILE
            return (edx >> 24) & 1;
#else
            return false;
#endif
        }

        /// Returns the maximum SIMD level supported by the current CPU,
        /// capped by the compile-time ENABLE_* macros.
        ///
        /// Example: if ENABLE_AVX512 is defined but the CPU only supports AVX2,
        /// this returns SimdLevel::AVX2. If ENABLE_AVX2 is defined and the CPU
        /// supports AVX512, this returns SimdLevel::AVX2 (capped by the macro).
        inline SimdLevel max_simd_level() {
            // Start from the highest compile-time allowed level and work down
            // to find the first one supported by the CPU.

#ifdef ENABLE_AMX
            if (cpu_supports_amx()) {
                return SimdLevel::AMX;
            }
#endif

#ifdef ENABLE_AVX512
            if (cpu_supports_avx512()) {
                return SimdLevel::AVX512;
            }
#endif

#ifdef ENABLE_AVX2
            if (cpu_supports_avx2()) {
                return SimdLevel::AVX2;
            }
#endif

#ifdef ENABLE_AVX
            if (cpu_supports_avx()) {
                return SimdLevel::AVX;
            }
#endif

#ifdef ENABLE_SSE3
            if (cpu_supports_sse3()) {
                return SimdLevel::SSE3;
            }
#endif

#ifdef ENABLE_SSE2
            if (cpu_supports_sse2()) {
                return SimdLevel::SSE2;
            }
#endif

            return SimdLevel::SCALAR;
        }

        /// Convenience: returns true if the current SIMD level is at least the given level.
        inline bool simd_at_least(SimdLevel level) {
            static SimdLevel cached = max_simd_level();
            return cached >= level;
        }

    }// namespace internal
}// namespace np

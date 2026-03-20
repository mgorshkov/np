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

/*
 * Benchmark GEBRD (bidiagonal reduction) comparing scalar vs AVX2 implementations.
 * This test directly calls gebrd_d_avx2 (AVX2-accelerated, blocked compact WY)
 * and compares its performance against the scalar gebrd<double>().
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

#include <np/internal/CpuDispatch.hpp>
#include <np/internal/cpu/LstSqGelsdGebrd.hpp>
#include <np/internal/cpu/LstSqGelsdGebrd_avx2.hpp>
#include <np/internal/cpu/LstSqGelsdGebrd_avx512.hpp>
#include <np/internal/cpu/LstSqGelsdTraits.hpp>

using namespace np;
using namespace np::internal;
using namespace np::internal::cpu;

class GebrdTimer {
public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count();
    }
    double ms() const { return static_cast<double>(m_elapsed); }

private:
    std::chrono::high_resolution_clock::time_point m_start, m_end;
    double m_elapsed = 0;
};

/// Benchmark GEBRD for a given matrix size, comparing scalar vs AVX2.
/// Uses the same random data for both runs.
static void benchmark_gebrd_compare(size_t m, size_t n) {
    size_t k = std::min(m, n);
    std::cout << "\n---------- GEBRD Comparison: " << m << "x" << n
              << " (k=" << k << ", double) ----------" << std::endl;

    random::seed(42);
    auto A_np = random::rand<double>(Shape({m, n}));
    std::vector<double> A_ref(A_np.data(), A_np.data() + m * n);

    // ---- Scalar GEBRD ----
    {
        std::vector<double> A_work = A_ref;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);

        GebrdTimer timer;
        timer.start();
        gebrd(A_work.data(), m, n, d.data(), e.data(), tauq.data(), taup.data());
        timer.stop();
        std::cout << "  Scalar gebrd:  " << std::setw(10) << timer.ms() << " ms" << std::endl;
    }

    // ---- AVX2 GEBRD (blocked compact WY) ----
#ifdef ENABLE_AVX2
    if (simd_at_least(SimdLevel::AVX2)) {
        std::vector<double> A_work = A_ref;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);

        GebrdTimer timer;
        timer.start();
        gebrd_d_avx2(A_work.data(), m, n, d.data(), e.data(), tauq.data(), taup.data());
        timer.stop();
        std::cout << "  AVX2 gebrd:    " << std::setw(10) << timer.ms() << " ms" << std::endl;
    } else {
        std::cout << "  AVX2 gebrd:    NOT AVAILABLE" << std::endl;
    }
#else
    std::cout << "  AVX2 gebrd:    NOT BUILT (ENABLE_AVX2=OFF)" << std::endl;
#endif

    // ---- AVX512 GEBRD (blocked compact WY) ----
#ifdef ENABLE_AVX512
    if (simd_at_least(SimdLevel::AVX512)) {
        std::vector<double> A_work = A_ref;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);

        GebrdTimer timer;
        timer.start();
        gebrd_d_avx512(A_work.data(), m, n, d.data(), e.data(), tauq.data(), taup.data());
        timer.stop();
        std::cout << "  AVX512 gebrd:  " << std::setw(10) << timer.ms() << " ms" << std::endl;
    } else {
        std::cout << "  AVX512 gebrd:  NOT AVAILABLE" << std::endl;
    }
#else
    std::cout << "  AVX512 gebrd:  NOT BUILT (ENABLE_AVX512=OFF)" << std::endl;
#endif
}

// ============================================================
// Test cases with various matrix sizes
// ============================================================

TEST(LinalgLstSqGelsdGebrdAvx2BenchmarkTest, benchmarkSmall) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gebrd_compare(1000, 200);
}

TEST(LinalgLstSqGelsdGebrdAvx2BenchmarkTest, benchmarkMedium) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gebrd_compare(5000, 500);
}

TEST(LinalgLstSqGelsdGebrdAvx2BenchmarkTest, benchmarkLarge) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gebrd_compare(10000, 500);
}

TEST(LinalgLstSqGelsdGebrdAvx2BenchmarkTest, benchmarkTall) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gebrd_compare(20000, 500);
}

TEST(LinalgLstSqGelsdGebrdAvx2BenchmarkTest, benchmarkSquare) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gebrd_compare(1000, 1000);
}

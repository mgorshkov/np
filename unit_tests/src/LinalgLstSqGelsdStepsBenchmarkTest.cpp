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
 * Benchmark each step of the GELSD pipeline individually.
 * Minimal version with progress output to track where time is spent.
 */

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

#include <np/internal/cpu/LstSqGelsdBackTransform.hpp>
#include <np/internal/cpu/LstSqGelsdBdsvdQr.hpp>
#include <np/internal/cpu/LstSqGelsdBlas.hpp>
#include <np/internal/cpu/LstSqGelsdDc.hpp>
#include <np/internal/cpu/LstSqGelsdGebrd.hpp>
#include <np/internal/cpu/LstSqGelsdHouseholder.hpp>
#include <np/internal/cpu/LstSqGelsdSolver.hpp>
#include <np/internal/cpu/LstSqGelsdTraits.hpp>

using namespace np;
using namespace np::internal::cpu;

class Timer {
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

template<typename T>
void benchmark_gelsd_steps(size_t m, size_t n) {
    size_t k = std::min(m, n);
    std::cout << "\n========== GELSD Step-by-Step: " << m << "x" << n
              << " (k=" << k << ", " << (sizeof(T) == sizeof(double) ? "double" : "float") << ")"
              << " ==========" << std::endl;

    random::seed(42);
    auto A_np = random::rand<T>(Shape({m, n}));
    auto b_np = random::rand<T>(Shape({m}));

    std::vector<T> A(A_np.data(), A_np.data() + m * n);
    std::vector<T> b(b_np.data(), b_np.data() + m);
    std::vector<T> x(n);

    // ============================================================
    // Step 1: GEBRD - Bidiagonal reduction
    // ============================================================
    std::vector<T> A_work = A;
    std::vector<T> d(k), e(k, T(0));
    std::vector<T> tauq(k), taup(k);

    std::cout << "  Starting GEBRD..." << std::endl;
    Timer timer;
    timer.start();
    gebrd(A_work.data(), m, n, d.data(), e.data(), tauq.data(), taup.data());
    timer.stop();
    double t_gebrd = timer.ms();
    std::cout << "  Step 1 - GEBRD: " << t_gebrd << " ms" << std::endl;

    // ============================================================
    // Step 2a: bdsvd_qr for various sizes
    // ============================================================
    std::cout << "  Step 2a - bdsvd_qr:" << std::endl;
    for (size_t test_n: {32, 64, 128, 256}) {
        if (test_n > k) break;
        std::vector<T> d_test(test_n), e_test(test_n, T(0));
        std::vector<T> s_test(test_n), U_test(test_n * test_n), VT_test(test_n * test_n);
        for (size_t i = 0; i < test_n; ++i) d_test[i] = T(1.0) + T(i) / T(test_n);
        for (size_t i = 1; i < test_n; ++i) e_test[i] = T(0.1) / T(test_n);

        Timer t2;
        t2.start();
        bdsvd_qr(d_test.data(), e_test.data(), test_n,
                 s_test.data(), U_test.data(), VT_test.data());
        t2.stop();
        std::cout << "    n=" << test_n << ": " << t2.ms() << " ms" << std::endl;
    }

    // ============================================================
    // Step 2b: bdsvd_dc (divide-and-conquer bidiagonal SVD)
    // ============================================================
    {
        std::cout << "  Starting bdsvd_dc..." << std::endl;
        std::vector<T> d_copy = d, e_copy = e;
        std::vector<T> s_dc(k), U_dc(k * k), VT_dc(k * k);

        Timer t_dc;
        t_dc.start();
        bdsvd_dc(d_copy.data(), e_copy.data(), k,
                 s_dc.data(), U_dc.data(), VT_dc.data());
        t_dc.stop();
        double t_dc_ms = t_dc.ms();
        std::cout << "  Step 2b - bdsvd_dc: " << t_dc_ms << " ms" << std::endl;
    }

    // ============================================================
    // Step 2c: Secular equation merge timing at each level
    // ============================================================
    std::cout << "  Step 2c - apply_svd_merge:" << std::endl;
    for (size_t merge_n: {32, 64, 128, 256}) {
        if (merge_n > k) break;
        size_t NL = merge_n / 2;
        size_t NR = merge_n - NL;

        std::vector<T> sL(NL), sR(NR);
        std::vector<T> UL(NL * NL), VTL(NL * NL);
        std::vector<T> UR(NR * NR), VTR(NR * NR);
        std::vector<T> z_coup(NL), w_coup(NR);

        for (size_t i = 0; i < NL; ++i) {
            sL[i] = T(NL - i) / T(NL);
            for (size_t j = 0; j < NL; ++j)
                UL[i * NL + j] = (i == j) ? T(1) : T(0);
        }
        for (size_t i = 0; i < NR; ++i) {
            sR[i] = T(NR - i) / T(NR);
            for (size_t j = 0; j < NR; ++j)
                UR[i * NR + j] = (i == j) ? T(1) : T(0);
        }
        for (size_t i = 0; i < NL; ++i) z_coup[i] = T(1) / T(NL);
        for (size_t i = 0; i < NR; ++i) w_coup[i] = T(1) / T(NR);
        T rho = T(0.1);

        std::vector<T> s_merged(merge_n);
        std::vector<T> U_merged(merge_n * merge_n);
        std::vector<T> VT_merged(merge_n * merge_n);

        merge_sorted_svd(sL.data(), sR.data(), UL.data(), VTL.data(),
                         UR.data(), VTR.data(), NL, NR,
                         s_merged.data(), U_merged.data(), VT_merged.data());

        Timer t_merge;
        t_merge.start();
        apply_svd_merge(sL.data(), sR.data(), z_coup.data(), w_coup.data(),
                        rho, NL, NR,
                        U_merged.data(), VT_merged.data(), s_merged.data());
        t_merge.stop();
        std::cout << "    N=" << merge_n << " (NL=" << NL << ", NR=" << NR << "): "
                  << t_merge.ms() << " ms" << std::endl;
    }

    // ============================================================
    // Step 3: Back-transform
    // ============================================================
    {
        std::vector<T> U_full(m * k, T(0));
        std::vector<T> VT_full(k * n, T(0));
        for (size_t i = 0; i < k; ++i) {
            U_full[i * k + i] = T(1);
            VT_full[i * n + i] = T(1);
        }

        Timer t_bt;
        t_bt.start();
        multiply_left_q(A_work.data(), m, n, tauq.data(), k, U_full.data(), k);
        multiply_right_pt(A_work.data(), m, n, taup.data(), k, VT_full.data(), k);
        t_bt.stop();
        std::cout << "  Step 3 - Back-transform: " << t_bt.ms() << " ms" << std::endl;
    }

    // ============================================================
    // Step 4: Solve
    // ============================================================
    {
        std::vector<T> d_copy = d, e_copy = e;
        std::vector<T> s_svd(k), U_svd(k * k), VT_svd(k * k);
        bdsvd_dc(d_copy.data(), e_copy.data(), k,
                 s_svd.data(), U_svd.data(), VT_svd.data());

        std::vector<T> U_full(m * k, T(0));
        std::vector<T> VT_full(k * n, T(0));
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                U_full[i * k + j] = U_svd[i * k + j];
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                VT_full[i * n + j] = VT_svd[i * k + j];

        multiply_left_q(A_work.data(), m, n, tauq.data(), k, U_full.data(), k);
        multiply_right_pt(A_work.data(), m, n, taup.data(), k, VT_full.data(), k);

        Timer t_solve;
        t_solve.start();
        std::vector<T> c(k, T(0));
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < m; ++j)
                c[i] += U_full[j * k + i] * b[j];
        T smax = (k > 0) ? s_svd[0] : T(0);
        T rcond_abs = std::numeric_limits<T>::epsilon() * smax;
        int rank = 0;
        for (size_t i = 0; i < k; ++i)
            if (s_svd[i] > rcond_abs) ++rank;
        for (size_t i = 0; i < k; ++i)
            c[i] = ((int) i < rank) ? (c[i] / s_svd[i]) : T(0);
        for (size_t i = 0; i < n; ++i) {
            x[i] = T(0);
            for (size_t j = 0; j < k; ++j)
                x[i] += VT_full[j * n + i] * c[j];
        }
        t_solve.stop();
        std::cout << "  Step 4 - Solve: " << t_solve.ms() << " ms" << std::endl;
    }

    // ============================================================
    // Full pipeline
    // ============================================================
    {
        std::cout << "  Starting full pipeline..." << std::endl;
        Timer t_full;
        t_full.start();
        A_work = A;
        std::vector<T> b_copy = b;
        std::vector<T> x_out(n);
        lstsq_gelsd_scalar(A_work.data(), b_copy.data(), x_out.data(), m, n, T(-1));
        t_full.stop();
        std::cout << "  Full lstsq_gelsd: " << t_full.ms() << " ms" << std::endl;
    }
}

TEST(LinalgLstSqGelsdStepsBenchmarkTest, benchmarkStepsDouble) {
    std::cout << std::fixed << std::setprecision(1);
    benchmark_gelsd_steps<double>(10000, 500);
}

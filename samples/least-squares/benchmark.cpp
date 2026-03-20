/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov

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

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

using namespace np;
using namespace np::linalg;

struct BenchmarkResult {
    std::string solver;
    size_t rows;
    size_t cols;
    double error;
    long long time_us;
};

// Measure one solver
template<typename SolverFunc>
BenchmarkResult run_solver(const std::string &name, SolverFunc solver,
                           const Array<float_> &A, const Array<float_> &b,
                           const Array<float_> &x_true) {
    auto start = std::chrono::high_resolution_clock::now();
    auto x = solver(A, b);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < x_true.size(); i++) {
        double diff = x.get(i) - x_true.get(i);
        error += diff * diff;
    }
    error = std::sqrt(error);

    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return {name, A.shape()[0], A.shape()[1], error, time_us};
}

int main() {
    // Matrix sizes to test
    std::vector<std::pair<size_t, size_t>> sizes = {
            {100, 10},
            {1000, 50},
            {10000, 100},
            {50000, 10},
            {100000, 2},
            {10000, 500},
    };

    std::vector<BenchmarkResult> results;
    random::seed(42);

    for (const auto &[rows, cols]: sizes) {
        std::cout << "=== Testing " << rows << " x " << cols << " ===" << std::endl;

        // Generate random problem
        Shape shapeA({rows, cols});
        auto A = random::rand<float_>(shapeA);
        Shape shapeX({cols});
        auto x_true = random::rand<float_>(shapeX);
        auto b = A.dot(x_true);// no noise

        // CPU Cholesky (fastest for tiny problems)
        auto res_chol = run_solver("Cholesky", [](const auto &A, const auto &b) { return lstsq_cholesky(A, b); }, A, b, x_true);
        results.push_back(res_chol);
        std::cout << "  Cholesky: error=" << res_chol.error << ", time=" << res_chol.time_us << " us" << std::endl;

        // CPU GELSD (SIMD-optimized divide-and-conquer SVD, same algorithm as numpy.linalg.lstsq)
        auto res_gelsd = run_solver("GELSD", [](const auto &A, const auto &b) { return lstsq_gelsd(A, b); }, A, b, x_true);
        results.push_back(res_gelsd);
        std::cout << "  GELSD:    error=" << res_gelsd.error << ", time=" << res_gelsd.time_us << " us" << std::endl;

#ifdef USE_CUDA
        // CUDA Tikhonov
        auto res_tik = run_solver("Tikhonov", [](const auto &A, const auto &b) { return lstsq_tikhonov(A, b, 1e-6); }, A, b, x_true);
        results.push_back(res_tik);
        std::cout << "  Tikhonov: error=" << res_tik.error << ", time=" << res_tik.time_us << " us" << std::endl;

        // CUDA MRRR
        auto res_mrrr = run_solver("MRRR", [](const auto &A, const auto &b) { return lstsq_mrrr(A, b); }, A, b, x_true);
        results.push_back(res_mrrr);
        std::cout << "  MRRR:     error=" << res_mrrr.error << ", time=" << res_mrrr.time_us << " us" << std::endl;

        // CUDA QR
        auto res_qr = run_solver("QR", [](const auto &A, const auto &b) { return lstsq_qr(A, b); }, A, b, x_true);
        results.push_back(res_qr);
        std::cout << "  QR:       error=" << res_qr.error << ", time=" << res_qr.time_us << " us" << std::endl;
#endif
    }

    // Print summary table
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << std::left << std::setw(12) << "Solver"
              << std::setw(12) << "Rows"
              << std::setw(12) << "Cols"
              << std::setw(16) << "Error"
              << std::setw(12) << "Time (us)"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto &r: results) {
        std::cout << std::left << std::setw(12) << r.solver
                  << std::setw(12) << r.rows
                  << std::setw(12) << r.cols
                  << std::setw(16) << std::scientific << std::setprecision(3) << r.error
                  << std::setw(12) << std::fixed << r.time_us
                  << std::endl;
    }

    return 0;
}

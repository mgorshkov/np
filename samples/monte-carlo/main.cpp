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
#include <iostream>
#include <np/Array.hpp>

int main(int argc, char **argv) {
    // PI number calculation with Monte-Carlo method
    using namespace np;
    static constexpr Size defaultSize = 100'000'000;

    Size size = defaultSize;
    if (argc >= 2) {
        size = std::atoll(argv[1]);
        if (size <= 0) {
            size = defaultSize;
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    auto rx = random::rand(size);
    auto ry = random::rand(size);

    // Compute distance squared: rx*rx + ry*ry
    // This creates a lazy expression tree (BinaryExpression<AddOp, ...>)
    // No evaluation happens until sum() is called
    auto dist = rx * rx + ry * ry;

    // np::sum with condition string "dist<1" evaluates the expression tree
    // in a single fused AVX2 pass with no intermediate array allocations:
    //   - Loads rx[i], ry[i] directly from source arrays
    //   - Computes rx[i]^2 + ry[i]^2 in registers
    //   - Compares to 1.0 and accumulates count
    //   - No store instructions, no temporary arrays
    //   - Returns the count directly as float_ (no IndexParent wrapper)
    auto inside = sum("dist<1", dist);

    auto t_end = std::chrono::high_resolution_clock::now();

    double pi_est = 4 * static_cast<double>(inside) / size;

    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

    std::cout << "PI=" << pi_est << std::endl;
    std::cerr << "total=" << total_us << " us" << std::endl;

    return 0;
}

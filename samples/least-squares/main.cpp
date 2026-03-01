/*
MIT License

Copyright (c) 2023 Mikhail Gorshkov

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

#include <iostream>
#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

int main(int, char **) {
    // LSTSQ calculation with Tikhonov Regularized EVD method
    using namespace np;
    using namespace np::linalg;

    static const constexpr Size rows = 1000;
    static const constexpr Size cols = 100;

    // Generate random matrix A and true solution x_true
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    // Add noise
    auto noise = random::rand(Shape{rows}, -0.01, 0.01); // 1 % noise
    // Compute b = A * x_true + noise
    auto b = A.dot(x_true) + noise;

    // Solve using Tikhonov Regularized EVD method
    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq(A, b);
    auto end = std::chrono::high_resolution_clock::now();

    double error = 0.0;
    for (size_t i = 0; i < cols; ++i) {
        error += (x.get(i) - x_true.get(i)) * (x.get(i) - x_true.get(i));
    }

    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time:  " << time.count() << " ms\n";
    std::cout << "||x - x_true||:  " << sqrtf(error) << "\n";

    return 0;
}

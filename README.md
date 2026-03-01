[![Build status](https://ci.appveyor.com/api/projects/status/g17ss46hmwi71tgu/branch/main?svg=true)](https://ci.appveyor.com/project/mgorshkov/np/branch/main)

# About
⚡ NumPy-style arrays in C++ | CUDA GPU + AVX512 CPU | Tikhonov Regularized EVD, LSQR, MRRR, SVD, eigenvalue solvers

# Description
**High-performance N-dimensional arrays** with CPU/GPU acceleration and built-in ML algorithms (Tikhonov Regularized EVD, LSQR, MRRR, SVD, eigenvalue solvers)

# Requirements
Any C++20-compatible compiler:
* gcc 10 or higher
* clang 6 or higher
* Visual Studio 2019 or higher
* CUDA development environment (NVIDIA CUDA Toolkit, and compatible NVIDIA drivers installed)

# Repo
```
git clone https://github.com/mgorshkov/np.git
```

# Build unit tests and sample
## Linux/MacOS
```
mkdir build && cd build
cmake ..
cmake --build .
```
## Windows
```
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

# Build docs
```
cmake --build . --target doc
```

Open np/build/doc/html/index.html in your browser.

# Install
```
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=~/np_install
cmake --build . --target install
```

# Usage example (samples/monte-carlo)
```
#include <iostream>
#include <np/Creators.hpp>

int main(int, char **) {
    // PI number calculation with Monte-Carlo method
    using namespace np;
    Size size = 10000000;
    auto rx = random::rand(size);
    auto ry = random::rand(size);
    auto dist = rx * rx + ry * ry;
    auto inside = (dist["dist<1"]).size();
    std::cout << "PI=" << 4 * static_cast<double>(inside) / size;
    return 0;
}
```
# How to build the sample

1. Clone the repo
```
git clone https://github.com/mgorshkov/np.git
```
2. cd samples/monte-carlo
```
cd samples/monte-carlo
```
3. Make build dir
```
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
## Linux/MacOS
```
cmake --build .
```
## Windows
```
cmake --build . --config Release
```
6. Run the app
```
$./monte_carlo
PI=3.14158
```

# Usage example (samples/least-squares)
```
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
    auto b = A * x_true + noise;

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
```
# How to build the sample

1. Clone the repo
```
git clone https://github.com/mgorshkov/np.git
```
2. cd samples/least-squares
```
cd samples/least-squares
```
3. Make build dir
```
mkdir -p build-release && cd build-release
```
4. Configure cmake
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
5. Build
## Linux/MacOS
```
cmake --build .
```
## Windows
```
cmake --build . --config Release
```
6. Run the app
```
$./least-squares
Time:  31 ms
||x - x_true||:  5.40635
```

# Links
* Methods from pandas library on top of NP library: https://github.com/mgorshkov/pd
* Scientific methods on top of NP library: https://github.com/mgorshkov/scipy
* ML Methods from scikit-learn library: https://github.com/mgorshkov/sklearn

# Plans
* MRRR LSQR algorithm implementation

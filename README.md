[![Build status](https://ci.appveyor.com/api/projects/status/g17ss46hmwi71tgu/branch/main?svg=true)](https://ci.appveyor.com/project/mgorshkov/np/branch/main)

![np logo](doc/logo.svg)

# About
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU | Tikhonov Regularized EVD, LSQR, MRRR, SVD, QR, eigenvalue solvers

# Description
**High-performance N-dimensional arrays** with CPU/GPU/Multithreading acceleration and built-in ML algorithms (Tikhonov Regularized EVD, LSQR, MRRR, SVD, QR, eigenvalue solvers)

# Requirements
C++20-compatible compiler:
* gcc 13 or higher
* clang 14 or higher
* Visual Studio 2019 or higher
* CUDA development environment (NVIDIA CUDA Toolkit, and compatible NVIDIA drivers installed) to use CUDA optimizations (nvcc 12 or higher)

# Repo
```
git clone https://github.com/mgorshkov/np.git
```

# Build library and unit tests
```
./scripts/build.sh
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
    auto inside = sum("dist<1", dist);
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
mkdir -p build && cd build
```
4. Configure cmake
```
cmake ..
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

# Python vs C++ Monte Carlo π Estimation (AVX2 CPU)

```bash
$ python compare_python_monte_carlo.py
```

## Results by size

=== Testing size=100000 ===
  Python numpy: pi=3.1376000000, time=2890 us, mem=2.3 MiB (est. 2.4 MiB)
  C++: pi=3.1438800000, time=723 us, mem≈1.5 MiB (theoretical)

=== Testing size=1000000 ===
  Python numpy: pi=3.1418640000, time=19819 us, mem=22.9 MiB (est. 23.8 MiB)
  C++: pi=3.1415800000, time=3489 us, mem≈15.3 MiB (theoretical)

=== Testing size=10000000 ===
  Python numpy: pi=3.1415772000, time=176955 us, mem=228.9 MiB (est. 238.4 MiB)
  C++: pi=3.1410600000, time=29943 us, mem≈152.6 MiB (theoretical)

=== Testing size=100000000 ===
  Python numpy: pi=3.1417230400, time=1749960 us, mem=2288.8 MiB (est. 2384.2 MiB)
  C++: pi=3.1414900000, time=289985 us, mem≈1525.9 MiB (theoretical)

## Summary table

| Size       | Py time (us) | Py mem (MiB) | C++ time (us) | C++ mem (MiB) | Speedup | Mem ratio |
|-----------|--------------|--------------|---------------|---------------|---------|-----------|
| 100000    | 2890         | 2.3          | 723           | 1.5           | 4.00x   | 1.5x      |
| 1000000   | 19819        | 22.9         | 3489          | 15.3          | 5.68x   | 1.5x      |
| 10000000  | 176955       | 228.9        | 29943         | 152.6         | 5.91x   | 1.5x      |
| 100000000 | 1749960      | 2288.8       | 289985        | 1525.9        | 6.03x   | 1.5x      |


**Average speedup is 5.4 times**

# Usage example (samples/least-squares)
```
#include <iostream>
#include <np/Array.hpp>
#include <np/linalg/LstSq.hpp>

int main(int, char **) {
    // LSTSQ calculation with MRRR method
    using namespace np;
    using namespace np::linalg;

    static const constexpr Size rows = 10000;
    static const constexpr Size cols = 1000;

    // Generate random matrix A and true solution x_true
    Shape shapeA({rows, cols});
    auto A = random::rand(shapeA);

    Shape shapeX({cols});
    auto x_true = random::rand(shapeX);

    // Add noise
    auto noise = random::rand(Shape{rows}, -0.01, 0.01); // 1 % noise
    // Compute b = A * x_true + noise
    auto b = A * x_true + noise;

    // Solve using MRRR method
    auto start = std::chrono::high_resolution_clock::now();
    auto x = lstsq_mrrr(A, b);
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
cmake ..
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
Time:  628 ms
||x - x_true||:  0.0066314
```

# Python vs C++ Least‑Squares (lstsq) Comparison

```bash
$ python compare_python_lstsq.py
```

## Results by problem size

=== Testing 100 x 10 ===
  numpy.lstsq: error=3.725290e-08, time=274 us, mem=0.0 MiB (est. 0.1 MiB)
  C++: error=1.487000e-09, time=2680 us, mem≈0.0 MiB (theoretical)

=== Testing 1000 x 50 ===
  numpy.lstsq: error=1.487115e-07, time=2001 us, mem=0.2 MiB (est. 4.0 MiB)
  C++: error=2.183000e-10, time=2866 us, mem≈0.4 MiB (theoretical)

=== Testing 10000 x 100 ===
  numpy.lstsq: error=5.107861e-08, time=519852 us, mem=3.9 MiB (est. 80.1 MiB)
  C++: error=3.652000e-11, time=6795 us, mem≈7.9 MiB (theoretical)

=== Testing 50000 x 10 ===
  numpy.lstsq: error=0.000000e+00, time=11025 us, mem=2.1 MiB (est. 40.2 MiB)
  C++: error=1.747000e-12, time=3692 us, mem≈4.2 MiB (theoretical)

=== Testing 100000 x 2 ===
  numpy.lstsq: error=0.000000e+00, time=2469 us, mem=1.1 MiB (est. 16.4 MiB)
  C++: error=1.064000e-13, time=3129 us, mem≈2.3 MiB (theoretical)

=== Testing 10000 x 500 ===
  numpy.lstsq: error=7.147771e-07, time=3330782 us, mem=19.1 MiB (est. 400.6 MiB)
  C++: error=8.301000e-11, time=205797 us, mem≈42.0 MiB (theoretical)

## Summary table

| Solver          | Rows   | Cols   | Error            | Time (us) | Mem (MiB) | vs numpy  |
|-----------------|--------|--------|------------------|-----------|-----------|-----------|
| numpy.lstsq     | 100    | 10     | 3.725290e-08     | 274       | 0.0       |           |
| numpy.lstsq     | 1000   | 50     | 1.487115e-07     | 2001      | 0.2       |           |
| numpy.lstsq     | 10000  | 100    | 5.107861e-08     | 519852    | 3.9       |           |
| numpy.lstsq     | 50000  | 10     | 0.000000e+00     | 11025     | 2.1       |           |
| numpy.lstsq     | 100000 | 2      | 0.000000e+00     | 2469      | 1.1       |           |
| numpy.lstsq     | 10000  | 500    | 7.147771e-07     | 3330782   | 19.1      |           |
| C++ Cholesky    | 100    | 10     | 1.487000e-09     | 2680      | 0.0       | -878.1%   |
| C++ Cholesky    | 1000   | 50     | 2.183000e-10     | 2866      | 0.4       | -43.2%    |
| C++ Cholesky    | 10000  | 100    | 3.652000e-11     | 6795      | 7.9       | +98.7%    |
| C++ Cholesky    | 50000  | 10     | 1.747000e-12     | 3692      | 4.2       | +66.5%    |
| C++ Cholesky    | 100000 | 2      | 1.064000e-13     | 3129      | 2.3       | -26.7%    |
| C++ Cholesky    | 10000  | 500    | 8.301000e-11     | 205797    | 42.0      | +93.8%    |

# Links
* ⚡ Data manipulation and analysis library in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/pd
* ⚡ SciPy methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/scipy
* ⚡ ML methods in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU: https://github.com/mgorshkov/sklearn

# Plans
* Other numpy functions implementation


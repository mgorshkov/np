# About
C++ numpy-like template-based array implementation.

# Description
Implements two flavours of N-dimensional array in a minimalistic way.

## Static array
std::array-based implementation, in which the element type and all the dimensions are fixed and determined at compile time.
This implies stack array storage.

## Dynamic array
std::vector-based implementation in which only the element type is known at compile time.
This implies heap array storage.

# Latest artifact
https://mgorshkov.jfrog.io/artifactory/default-generic-local/np/np-0.0.3.tgz

# Requirements
Any C++17-compatible compiler:
* gcc 8 or higher
* clang 6 or higher
* Visual Studio 2017 or higher

# Repo
```
git clone https://github.com/mgorshkov/np.git
```

# Build unit tests and sample
```
mkdir build && cd build
cmake ..
cmake --build .
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
```
cmake --build .
```
6. Run the app
```
$./monte_carlo
PI=3.14158
```

# Links
ML Methods on top of NP library: https://github.com/mgorshkov/ml
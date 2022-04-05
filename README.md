# About
C++ numpy-like template-based array implementation

# Description
Implements two flavours of N-dimensional array in a minimalistic way

## Static array

std::array-based implementation, in which the element type and all the dimensions are fixed and determined at compile time.
This implies stack array storage.

## Dynamic array
std::vector-based implementation in which only the element type is known at compile time.
This implies heap array storage

# Repo
```
git clone https://github.com/mgorshkov/np.git
```

# Docs
```
cd doc
mkdir -p build && cd build
cmake ..
make doc
```

Open np/doc/build/html/index.html in your browser.

# Usage example
```
#include "include/np.hpp"

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
1. Clone the repo
2. Include the repo into user's project as CMake subproject
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
cmake --build.
```
6. Run the app
```
$./monte_carlo
PI=3.14158
```

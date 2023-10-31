/*
C++ numpy-like template-based array implementation

Copyright (c) 2023 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/Array.hpp>
#include <np/linalg/Inv.hpp>

#include <ArrayTest.hpp>

using namespace np;

class LinalgTest : public ArrayTest {
protected:
};

TEST_F(LinalgTest, testInv2x2) {
    float_ c_array[2][2] = {{1.0, 2.0}, {0.0, 1.0}};
    Array<float_> array{c_array};
    float_ c_array_inv_sample[2][2] = {{1.0, -2.0}, {0.0, 1.0}};
    Array<float_> array_inv_sample{c_array_inv_sample};
    auto result = linalg::inv(array);
    compare(array_inv_sample, result);
}

TEST_F(LinalgTest, testInv3x3) {
    float_ c_array[3][3] = {{1.0, 2.0, 3.0}, {0.0, 1.0, 4.0}, {5.0, 6.0, 1.0}};
    Array<float_> array{c_array};
    float_ c_array_inv_sample[3][3] = {{-11.5, 8.0, 2.5}, {10.0, -7.0, -2.0}, {-2.5, 2.0, 0.5}};
    Array<float_> array_inv_sample{c_array_inv_sample};
    auto result = linalg::inv(array);
    compare(array_inv_sample, result);
}

TEST_F(LinalgTest, testInv3x3Singular) {
    float_ c_array[3][3] = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    Array<float_> array{c_array};
    EXPECT_THROW(auto result = linalg::inv(array), std::runtime_error);
}

TEST_F(LinalgTest, testInv4x4) {
    float_ c_array[4][4] = {{1.0, 2.0, 3.0, 4.0}, {0.0, 1.0, 4.0, 5.0}, {5.0, 6.0, 1.0, 6.0}, {7.0, 6.0, 5.0, 4.0}};
    Array<float_> array{c_array};
    float_ c_array_inv_sample[4][4] = {{-2.21875, 1.25, 0.25, 0.28125}, {2.78125, -1.75, -0.25, -0.21875}, {0.59375, -0.25, -0.25, 0.09375}, {-1.03125, 0.75, 0.25, -0.03125}};
    Array<float_> array_inv_sample{c_array_inv_sample};
    auto result = linalg::inv(array);
    compare(array_inv_sample, result);
}

TEST_F(LinalgTest, testInv5x5) {
    float_ c_array[5][5] = {{1.0, 2.0, 3.0, 4.0, 5.0}, {0.0, 1.0, 4.0, 5.0, 6.0}, {5.0, 6.0, 1.0, 6.0, 2.0}, {7.0, 6.0, 5.0, 4.0, 2.0}, {1.0, 8.0, 3.0, 6.0, 2.0}};
    Array<float_> array{c_array};
    float_ c_array_inv_sample[5][5] = {{0.057554, -0.0611511, 0.104317, 0.0809353, -0.145683},
                                       {0.561151, -0.471223, -0.107914, -0.0233813, 0.142086},
                                       {-0.446043, 0.348921, -0.183453, 0.185252, 0.0665468},
                                       {-0.834532, 0.636691, 0.23741, -0.0485612, -0.0125899},
                                       {0.899281, -0.517986, -0.057554, -0.0791367, -0.057554}};
    Array<float_> array_inv_sample{c_array_inv_sample};
    auto result = linalg::inv(array);
    compare(array_inv_sample, result);
}

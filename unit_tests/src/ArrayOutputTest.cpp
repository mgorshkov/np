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

#include <sstream>

#include <np/Array.hpp>

#include <ArrayTest.hpp>

using namespace np;

class ArrayOutputTest : public ArrayTest {
};

TEST_F(ArrayOutputTest, dynamicEmptyArrayTest) {
    /*
     >> np.array([])
     */
    Array<intc> array{};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[]");
}

TEST_F(ArrayOutputTest, dynamic1DArrayTest) {
    /*
     >> np.array([1, 2, 3])
     */
    Array<intc> array{1, 2, 3};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[1 2 3]");
}

TEST_F(ArrayOutputTest, dynamic2DArrayTest) {
    /*
     >> np.array([[1, 2], [3, 4], [5, 6])
     */
    intc array_c[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<intc> array{array_c};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[[1 2]\n [3 4]\n [5 6]]");
}

TEST_F(ArrayOutputTest, staticEmptyArrayTest) {
    /*
     >> np.array([])
     */
    Array<intc, 0> array{};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[]");
}

TEST_F(ArrayOutputTest, static1DArrayTest) {
    /*
     >> np.array([1, 2, 3])
     */
    Array<intc, 3> array{1, 2, 3};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[1 2 3]");
}

TEST_F(ArrayOutputTest, static2DArrayTest) {
    /*
     >> np.array([[1, 2], [3, 4], [5, 6])
     */
    intc array_c[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<intc, 6> array{array_c};
    std::stringstream stream;
    stream << array;
    EXPECT_EQ(stream.str(), "[[1 2]\n [3 4]\n [5 6]]");
}

TEST_F(ArrayOutputTest, identityArrayTest) {
    std::stringstream stream;
    stream << eye(4);
    EXPECT_EQ(stream.str(), "[[1 0 0 0]\n [0 1 0 0]\n [0 0 1 0]\n [0 0 0 1]]");
}

TEST_F(ArrayOutputTest, diagonalArrayTest) {
    intc array_c[4][2] = {{1, 2}, {3, 4}, {5, 6}};
    Array<intc, 8> array{array_c};
    std::stringstream stream;
    stream << diag(array);
    EXPECT_EQ(stream.str(), "[1 4]");
}

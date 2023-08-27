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

/* Sample data generation for ML methods
Python equivalent perf.py:

import numpy as np

def generate_data(rank, num_points, noise_level):
    np.random.seed(42)
    x = np.linspace(-10, 10, num_points)
    y = np.linspace(-10, 10, num_points)
    if rank == 1:
        z = 3 * x + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 2:
        z = 2 * x + 3 * y + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    elif rank == 3:
        z = 2 * x**2 + 3 * y**2 + 5 + noise_level * np.random.randn(num_points)
        data = np.column_stack((x, y, z))
    return data

num_points = 100*1000
n_runs = 50
ranks = [1, 2, 3]
noise_levels = [0, 1, 10, 50]
for rank in ranks:
    for noise_level in noise_levels:
        data = generate_data(rank, num_points, noise_level)
        print(data)

 $ time python3 perf.py
[[-10.         -10.         -25.        ]
 [ -9.9998      -9.9998     -24.99939999]
 [ -9.9996      -9.9996     -24.99879999]
 ...
 [  9.9996       9.9996      34.99879999]
 [  9.9998       9.9998      34.99939999]
 [ 10.          10.          35.        ]]
[[-10.         -10.         -24.50328585]
 [ -9.9998      -9.9998     -25.1376643 ]
 [ -9.9996      -9.9996     -24.35111145]
 ...
 [  9.9996       9.9996      35.40798507]
 [  9.9998       9.9998      34.78830832]
 [ 10.          10.          35.12006294]]
[[-10.         -10.         -20.03285847]
 [ -9.9998      -9.9998     -26.38204301]
 [ -9.9996      -9.9996     -18.52191461]
 ...
 [  9.9996       9.9996      39.09065077]
 [  9.9998       9.9998      32.88848329]
 [ 10.          10.          36.20062941]]
[[-10.         -10.          -0.16429235]
 [ -9.9998      -9.9998     -31.91261505]
 [ -9.9996      -9.9996       7.38562692]
 ...
 [  9.9996       9.9996      55.4580539 ]
 [  9.9998       9.9998      24.44481646]
 [ 10.          10.          41.00314704]]
[[-10.         -10.         -45.        ]
 [ -9.9998      -9.9998     -44.99899999]
 [ -9.9996      -9.9996     -44.99799998]
 ...
 [  9.9996       9.9996      54.99799998]
 [  9.9998       9.9998      54.99899999]
 [ 10.          10.          55.        ]]
[[-10.         -10.         -44.50328585]
 [ -9.9998      -9.9998     -45.13726429]
 [ -9.9996      -9.9996     -44.35031144]
 ...
 [  9.9996       9.9996      55.40718506]
 [  9.9998       9.9998      54.78790832]
 [ 10.          10.          55.12006294]]
[[-10.         -10.         -40.03285847]
 [ -9.9998      -9.9998     -46.381643  ]
 [ -9.9996      -9.9996     -38.5211146 ]
 ...
 [  9.9996       9.9996      59.08985076]
 [  9.9998       9.9998      52.88808328]
 [ 10.          10.          56.20062941]]
[[-10.         -10.         -20.16429235]
 [ -9.9998      -9.9998     -51.91221505]
 [ -9.9996      -9.9996     -12.61357307]
 ...
 [  9.9996       9.9996      75.4572539 ]
 [  9.9998       9.9998      44.44441645]
 [ 10.          10.          61.00314704]]
[[-10.        -10.        505.       ]
 [ -9.9998     -9.9998    504.98     ]
 [ -9.9996     -9.9996    504.9600004]
 ...
 [  9.9996      9.9996    504.9600004]
 [  9.9998      9.9998    504.98     ]
 [ 10.         10.        505.       ]]
[[-10.         -10.         505.49671415]
 [ -9.9998      -9.9998     504.8417357 ]
 [ -9.9996      -9.9996     505.60768894]
 ...
 [  9.9996       9.9996     505.36918548]
 [  9.9998       9.9998     504.76890833]
 [ 10.          10.         505.12006294]]
[[-10.         -10.         509.96714153]
 [ -9.9998      -9.9998     503.59735699]
 [ -9.9996      -9.9996     511.43688578]
 ...
 [  9.9996       9.9996     509.05185118]
 [  9.9998       9.9998     502.86908329]
 [ 10.          10.         506.20062941]]
[[-10.         -10.         529.83570765]
 [ -9.9998      -9.9998     498.06678494]
 [ -9.9996      -9.9996     537.34442731]
 ...
 [  9.9996       9.9996     525.41925432]
 [  9.9998       9.9998     494.42541646]
 [ 10.          10.         511.00314704]]

real	0m1.598s
user	0m0.378s
sys	0m0.111s
 */

using namespace np;

auto generate_data(auto rank, auto num_points, auto noise_level) {
    random::seed(42);
    auto x = linspace(-10.0, 10.0, num_points);
    auto y = linspace(-10.0, 10.0, num_points);
    if (rank == 1) {
        auto z = 3 * x + 5 + noise_level * random::randn(num_points);
        return column_stack(x, y, z);
    }
    if (rank == 2) {
        auto z = 2 * x + 3 * y + 5 + noise_level * random::randn(num_points);
        return column_stack(x, y, z);
    }
    auto z = 2 * x * x + 3 * y * y + 5 + noise_level * random::randn(num_points);
    return column_stack(x, y, z);
}

int main(int, char **) {
    int num_points = 100 * 1000;
    auto ranks = {1, 2, 3};
    auto noise_levels = {0, 1, 10, 50};
    for (auto rank: ranks) {
        for (auto noise_level: noise_levels) {
            auto data = generate_data(rank, num_points, noise_level);
            std::cout << "data=" << data << std::endl;
        }
    }

    return 0;
}

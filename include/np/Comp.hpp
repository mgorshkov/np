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

#pragma once

#include <np/Constants.hpp>
#include <np/DType.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/dynamic/NDArrayDynamicCompImpl.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>
#include <np/ndarray/static/NDArrayStaticCompImpl.hpp>

#include <np/ndarray/internal/Tools.hpp>

#include <np/internal/Tools.hpp>

namespace np {

    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    inline static bool array_equal(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b) {
        if (a.shape() != b.shape())
            return false;
        for (Size index = 0; index < a.size(); ++index) {
            if (!internal::element_equal(a.get(index), b.get(index)))
                return false;
        }
        return true;
    }

    template<typename DType1, typename Derived1, typename Storage1, typename DType2, typename Derived2, typename Storage2>
    inline static auto isclose(const ndarray::internal::NDArrayBase<DType1, Derived1, Storage1> &a, const ndarray::internal::NDArrayBase<DType2, Derived2, Storage2> &b,
                               np::float_ rtol = 1e-05, np::float_ atol = 1e-08, bool equal_nan = false) {
        if (a.shape() != b.shape())
            throw std::runtime_error("Arrays are different");
        ndarray::array_dynamic::NDArrayDynamic<bool_> result{a.shape()};
        for (Size index = 0; index < a.size(); ++index) {
            result.set(index, internal::element_equal(a.get(index), b.get(index), rtol, atol, equal_nan));
        }
        return result;
    }

}// namespace np

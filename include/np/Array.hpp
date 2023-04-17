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

#ifdef OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() {
    return 1;
}

inline int omp_get_thread_num() {
    return 0;
}
#endif

#include <iostream>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

// Multidimensional, homogeneous array of fixed-size items.
namespace np {
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    using Array = typename std::conditional<
            SizeT == SIZE_DEFAULT,
            ndarray::array_dynamic::NDArrayDynamic<DType>,
            ndarray::array_static::NDArrayStatic<DType, SizeT>>::type;

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, typename... Args>
    auto createArray(Args &&...args) {
        return Array<DType, SizeT>{std::forward<Args>(args)...};
    }

    template<Size SizeT = SIZE_DEFAULT, typename... Args>
    auto createIntArray(Args &&...args) {
        return createArray<int_, SizeT>(std::forward<Args>(args)...);
    }

    template<Size SizeT = SIZE_DEFAULT, typename... Args>
    auto createFloatArray(Args &&...args) {
        return createArray<float_, SizeT>(std::forward<Args>(args)...);
    }

    template<Size SizeT = SIZE_DEFAULT, typename... Args>
    auto createStringArray(Args &&...args) {
        return createArray<string_, SizeT>(std::forward<Args>(args)...);
    }

    template<Size SizeT = SIZE_DEFAULT, typename... Args>
    auto createUnicodeArray(Args &&...args) {
        return createArray<unicode_, SizeT>(std::forward<Args>(args)...);
    }

    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    using ArrayIndexKeyType = typename std::conditional<
            SizeT == SIZE_DEFAULT,
            ndarray::array_dynamic::NDArrayDynamicIndexKeyType<DType>,
            ndarray::array_static::NDArrayStaticIndexKeyType<DType, SizeT>>::type;

    template<typename DType = DTypeDefault, typename ValueType = DType, Size SizeT = SIZE_DEFAULT, typename Hasher = ndarray::internal::NDArrayBaseHasher, typename EqualTo = ndarray::internal::NDArrayBaseEqualTo>
    using ArrayIndexMap = typename std::conditional<
            SizeT == SIZE_DEFAULT,
            ndarray::array_dynamic::NDArrayDynamicIndexMap<DType, ValueType, Hasher, EqualTo>,
            ndarray::array_static::NDArrayStaticIndexMap<DType, SizeT, ValueType, Hasher, EqualTo>>::type;

    template<typename DType = DTypeDefault, typename ValueType = DType, Size SizeT = SIZE_DEFAULT, typename Hasher = ndarray::internal::NDArrayBaseHasher, typename EqualTo = ndarray::internal::NDArrayBaseEqualTo>
    using ArrayIndexConstMap = typename std::conditional<
            SizeT == SIZE_DEFAULT,
            ndarray::array_dynamic::NDArrayDynamicIndexConstMap<DType, ValueType, Hasher, EqualTo>,
            ndarray::array_static::NDArrayStaticIndexConstMap<DType, SizeT, ValueType, Hasher, EqualTo>>::type;
}// namespace np

#include "np/ndarray/internal/NDArrayBaseStreamIoImpl.hpp"
#include <np/Agg.hpp>
#include <np/Axis.hpp>
#include <np/Comp.hpp>
#include <np/Copy.hpp>
#include <np/Creators.hpp>
#include <np/Index.hpp>
#include <np/Inspect.hpp>
#include <np/Io.hpp>
#include <np/Manip.hpp>
#include <np/Math.hpp>
#include <np/Sort.hpp>
#include <np/ndarray/dynamic/internal/NDArrayDynamicStorageStreamIoImpl.hpp>
#include <np/ndarray/internal/NDArrayBaseImpl.hpp>
#include <np/ndarray/internal/NDArrayShapedImpl.hpp>
#include <np/ndarray/static/internal/NDArrayStaticStorageStreamIoImpl.hpp>

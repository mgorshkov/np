/*
C++ numpy-like template-based array implementation

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <math.h>

#include <np/DType.hpp>
#include <np/Constants.hpp>

#include <np/ndarray/static/NDArrayStatic.hpp>
#include <np/ndarray/dynamic/NDArrayDynamic.hpp>

#include <np/ndarray/internal/Tools.hpp>

#include <np/internal/Tools.hpp>

namespace np {
    using ndarray::array_static::NDArrayStatic;
    using ndarray::array_static::NDArrayStaticStub;
    using ndarray::array_dynamic::NDArrayDynamic;

    // Static arrays
    // Single element vs a static array
    template<typename DType, Size SizeT, Size... SizeTs>
    inline static bool array_equal(const DType &value, const NDArrayStatic<DType, SizeT, SizeTs...> &array) {
        return array.array_equal(value);
    }

    template<typename DType, Size SizeT>
    inline static bool array_equal(const DType &value, const NDArrayStaticStub<DType> &array) {
        return array.array_equal(value);
    }

    // Static array vs a single element
    template<typename DType, Size SizeT, Size... SizeTs>
    inline static bool array_equal(const NDArrayStatic<DType, SizeT, SizeTs...> &array, const DType &value) {
        return array.array_equal(value);
    }

    template<typename DType, Size SizeT>
    inline static bool array_equal(const NDArrayStaticStub<DType> &array, const DType &value) {
        return array.array_equal(value);
    }

    // Static array vs a static array
    // Arraywise comparison
    template<typename DType, Size SizeT, Size... SizeTs>
    inline static bool array_equal(const NDArrayStatic<DType, SizeT, SizeTs...> &a, const NDArrayStatic<DType, SizeT, SizeTs...> &b) {
        if (a.shape() != b.shape())
            return false;
        for (Size index = 0; index < SizeT; ++index) {
            if (!array_equal(a[index], b[index]))
                return false;
        }
        return true;
    }

    // Dynamic arrays
    // Single element vs a dynamic array
    template<typename DType>
    inline static bool array_equal(const DType &value, const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &array) {
        return array.array_equal(value);
    }

    // Single element vs a dynamic array
    template<typename DType>
    inline static bool array_equal(const DType &value, const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &array) {
        return array.array_equal(value);
    }

    template<typename DType>
    inline static bool array_equal(const DType &value, const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &array) {
        return array.array_equal(value);
    }

    // Dynamic array vs a single element
    template<typename DType>
    inline static bool array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &array, const DType &value) {
        return array.array_equal(value);
    }

    // Dynamic array vs a single element
    template<typename DType>
    inline static bool array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &array, const DType &value) {
        return array.array_equal(value);
    }

    template<typename DType>
    inline static bool array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &array, const DType &value) {
        return array.array_equal(value);
    }

    // Dynamic array vs a dynamic array
    // Arraywise comparison
    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &a, 
        const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &b) {
        return a.array_equal(b);
    }

    // Arraywise comparison
    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &a, 
        const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &b) {
        return a.array_equal(b);
    }

    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &a,
                const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &b) {
        return a.array_equal(b);
    }

    // Arraywise comparison
    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &a, 
        const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &b) {
        return a.array_equal(b);
    }

    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &a,
                const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageVector<DType>> &b) {
        return a.array_equal(b);
    }

    // Arraywise comparison
    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &a, 
        const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &b) {
        return a.array_equal(b);
    }

    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &a,
                const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &b) {
        return a.array_equal(b);
    }

    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &a,
                const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &b) {
        return a.array_equal(b);
    }

    template<typename DType>
    inline static bool
    array_equal(const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageConstSpan<DType>> &a,
                const NDArrayDynamic <DType, ndarray::array_dynamic::internal::NDArrayDynamicInternalStorageSpan<DType>> &b) {
        return a.array_equal(b);
    }

    // Static vs dynamic arrays
    // Arraywise comparison
    template<typename DType, typename Storage>
    inline static bool array_equal(const NDArrayStaticStub<DType> &a, const NDArrayDynamic <DType, Storage> &b) {
        auto shape1 = a.shape();
        auto shape2 = b.shape();
        if (a.shape() != b.shape())
            return false;
        return a.get(0) == b.get(0);
    }

    template<typename DType, typename Storage, Size SizeT, Size... SizeTs>
    inline static bool array_equal(const NDArrayStatic<DType, SizeT, SizeTs...> &a, const NDArrayDynamic <DType, Storage> &b) {
        auto shape1 = a.shape();
        auto shape2 = b.shape();
        if (a.shape() != b.shape())
            return false;
        for (Size index = 0; index < SizeT; ++index) {
            if (!array_equal(a[index], b[index]))
                return false;
        }
        return true;
    }

    // Dynamic vs static arrays
    // Arraywise comparison
    template<typename DType, typename Storage>
    inline static bool
    array_equal(const NDArrayDynamic <DType, Storage> &a, const NDArrayStaticStub<DType> &b) {
        return array_equal(b, a);
    }

    template<typename DType, typename Storage, Size SizeT, Size... SizeTs>
    inline static bool
    array_equal(const NDArrayDynamic <DType, Storage> &a, const NDArrayStatic<DType, SizeT, SizeTs...> &b) {
        return array_equal(b, a);
    }

}

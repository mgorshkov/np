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

#include <np/ndarray/static/NDArrayStaticDecl.hpp>

namespace np::ndarray::array_static {
    // Returns a new Array with axes transposed.
    /*
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> transpose(const NDArrayStatic<DType, SizeT, SizeTs...> &array) {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Returns a new Array with axes transposed.
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::T() const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Flatten the array
    template<typename DType, Size SizeT, Size... SizeTs>
    auto NDArrayStatic<DType, SizeT, SizeTs...>::ravel() const {
        NDArrayStatic<DType, (SizeT * ... * SizeTs)> result;
        for (Size i = 0; i < SizeT; ++i) {
            const auto& element = at(i);
            const auto& r = element.ravel();
            //result = result + ;
        }
        return result;
    }

    // Reshape, but don’t change data
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::reshape(Shape shape) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Adding and removing elements
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::resize(Shape shape) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Append items to an array
    template<typename DType, Size SizeT, Size... SizeTs>
    auto NDArrayStatic<DType, SizeT, SizeTs...>::append(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, 2 * (SizeT * ... * SizeTs)> result;
        return result;
    }

    // Insert items in an array
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::insert(Size index, const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Delete items from an array
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::del(Size index) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Concatenate arrays
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::concatenate(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Stack arrays vertically (rowwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::vstack(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Stack arrays vertically (rowwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::r_(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Stack arrays horizontally (columnwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::hstack(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Create stacked columnwise arrays
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::column_stack(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Stack arrays horizontally (columnwise)
    template<typename DType, Size SizeT, Size... SizeTs>
    NDArrayStatic<DType, SizeT, SizeTs...> NDArrayStatic<DType, SizeT, SizeTs...>::c_(const NDArrayStatic<DType, SizeT, SizeTs...>& array) const {
        NDArrayStatic<DType, SizeT, SizeTs...> result;
        return result;
    }

    // Splitting arrays
    // Split the array horizontally
    template<typename DType, Size SizeT, Size... SizeTs>
    std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> NDArrayStatic<DType, SizeT, SizeTs...>::hsplit(Size index) const {
        std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> result;
        return result;
    }

    // Split the array vertically at the 2nd index
    template<typename DType, Size SizeT, Size... SizeTs>
    std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> NDArrayStatic<DType, SizeT, SizeTs...>::vsplit(Size index) const {
        std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> result;
        return result;
    }
    */
} // namespace np::ndarray::array_static


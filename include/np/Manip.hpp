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

#include <np/Constants.hpp>
#include <np/DType.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    //////////////////////////////////////////////////////////////
    /// \brief Array transpose
    ///
    /// Returns a new Array with axes transposed.
    ///
    /// \param array Array to transpose
    ///
    /// \return The transposed array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    auto transpose(const Array<DType, SizeT, SizeTs...> &array) {
        return array.transpose();
    }

    //////////////////////////////////////////////////////////////
    /// \brief Append items to an array
    ///
    /// Append items to an array from another array
    /// Both arrays are flattened before append
    ///
    /// \param array1 Array to append to
    /// \param array2 Array to append from
    ///
    /// \return The 1D resulting array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType, SizeT, SizeTs...> append(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.append(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Insert items in an array
    ///
    /// Insert items in an array from another array at a given index.
    /// Both arrays are flattened before append
    ///
    /// \param array1 Array to insert to
    /// \param array2 Array to insert from
    ///
    /// \return The 1D resulting array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType, SizeT, SizeTs...> insert(const Array<DType, SizeT, SizeTs...> &array1, Size index, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.insert(index, array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Delete an item from an array
    ///
    /// Delete an item from an array at a given index.
    /// Array is flattened before deletion
    ///
    /// \param array Array to delete an item from
    /// \param index Index to delete
    ///
    /// \return The 1D resulting array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType, SizeT, SizeTs...> del(const Array<DType, SizeT, SizeTs...> &array, Size index) {
        return array.del(index);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Concatenate arrays
    ///
    /// Join a sequence of arrays along an existing axis.
    ///
    /// \param array1 Array to concatenate
    /// \param array2 Array to concatenate
    /// \param axis   Axis to use
    ///
    /// \return The resulting array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType, SizeT, SizeTs...> concatenate(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2, std::optional<std::size_t> axis = std::nullopt) {
        return array1.concatenate(array2, axis);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack arrays vertically
    ///
    /// Stack arrays vertically (row wise)
    /// The arrays must have the same shape along all but the first axis.
    /// 1D arrays must have the same length.
    ///
    /// \param array1 Array to stack
    /// \param array2 Array to stack
    ///
    /// \return The array formed by stacking the given arrays, will be at least 2-D.
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    auto vstack(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.vstack(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack arrays vertically
    ///
    /// Stack arrays vertically (row wise)
    /// The arrays must have the same shape along all but the first axis.
    /// 1D arrays must have the same length.
    ///
    /// \param array1 Array to stack
    /// \param array2 Array to stack
    ///
    /// \return The array formed by stacking the given arrays, will be at least 2-D.
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    auto r_(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.r_(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack arrays horizontally
    ///
    /// Stack arrays in sequence horizontally (column wise).
    /// This is equivalent to concatenation along the second axis, except for 1-D arrays where it concatenates along the first axis.
    /// The arrays must have the same shape along all but the second axis.
    /// 1D arrays must have the same length.
    ///
    /// \param array1 Array to stack
    /// \param array2 Array to stack
    ///
    /// \return The array formed by stacking the given arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType> hstack(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.hstack(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack 1D arrays as columns into a 2D array
    ///
    /// Take a sequence of 1D arrays and stack them as columns to make a single 2D array.
    /// 2D arrays are stacked as-is, just like with hstack.
    /// 1D arrays are turned into 2D columns first.
    ///
    /// \param array1 Array to stack
    /// \param array2 Array to stack
    ///
    /// \return The array formed by stacking the given arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType> column_stack(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.column_stack(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack arrays horizontally
    ///
    /// Stack arrays horizontally (column wise)
    /// The arrays must have the same shape along all but the second axis.
    /// 1D arrays must have the same length.
    /// Currently it's the same as hstack
    ///
    /// \param array1 Array to stack
    /// \param array2 Array to stack
    ///
    /// \return The array formed by stacking the given arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    Array<DType> c_(const Array<DType, SizeT, SizeTs...> &array1, const Array<DType, SizeT, SizeTs...> &array2) {
        return array1.c_(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Split an array horizontally
    ///
    /// Split an array into multiple sub-arrays horizontally (column-wise).
    /// The array is always split along the second axis regardless of the array dimension.
    ///
    /// \param array Array to split
    /// \param sections Number of sections
    ///
    /// \return Vector of split array parts
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    std::vector<Array<DType>> hsplit(const Array<DType, SizeT, SizeTs...> &array, std::size_t sections) {
        return array.hsplit(sections);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Split the array vertically
    ///
    /// Split an array into multiple sub-arrays vertically (row-wise).
    /// The array is always split along the first axis regardless of the array dimension.
    ///
    /// \param array Array to split
    /// \param sections Number of sections
    ///
    /// \return Vector of split array parts
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT, Size... SizeTs>
    std::vector<Array<DType>> vsplit(const Array<DType, SizeT, SizeTs...> &array, std::size_t sections) {
        return array.vsplit(sections);
    }

}// namespace np

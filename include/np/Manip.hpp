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
    template<typename DType, typename Derived, typename Storage>
    auto transpose(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto append(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto insert(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, Size index, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
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
    template<typename DType, typename Derived, typename Storage>
    auto del(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, Size index) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto concatenate(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2, std::optional<std::size_t> axis = std::nullopt) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto vstack(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto r_(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto hstack(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
        return array1.hstack(array2);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Stack 1D arrays as columns into a 2D array
    ///
    /// Take a sequence of 1D arrays and stack them as columns to make a single 2D array.
    /// 2D arrays are stacked as-is, just like with hstack.
    /// 1D arrays are turned into 2D columns first.
    ///
    /// \param arg1, args Arrays to stack
    ///
    /// \return The array formed by stacking the given arrays
    ///
    //////////////////////////////////////////////////////////////
    template<typename Arg>
    auto all_empty(Arg &&array) {
        return array.empty();
    }

    template<typename Arg, typename... Args>
    auto all_empty(Arg &&arg1, Args &&...args) {
        return arg1.empty() && all_empty(args...);
    }

    template<typename Arg>
    auto check_ndim(Arg &&array) {
        return array.ndim();
    }

    template<typename Arg, typename... Args>
    auto check_ndim(Arg &&arg1, Args &&...args) {
        auto ndim_args = check_ndim(args...);
        if (check_ndim(arg1) != ndim_args) {
            throw std::runtime_error("Number of dims should be equal");
        }
        return ndim_args;
    }

    template<typename Target, typename Arg>
    void fill(Target &&target, Size indexSource, Size &indexTarget, Arg &&arg1) {
        target.set(indexTarget++, arg1.get(indexSource));
    }

    template<typename Target, typename Arg, typename... Args>
    void fill(Target &&target, Size indexSource, Size &indexTarget, Arg &&arg1, Args &&...args) {
        fill(target, indexSource, indexTarget, arg1);
        fill(target, indexSource, indexTarget, args...);
    }

    template<typename DType, typename Derived, typename Storage, typename... Args>
    auto column_stack(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &arg1, Args &&...args) {
        if (all_empty(arg1, args...)) {
            return ndarray::array_dynamic::NDArrayDynamic<DType>{};
        }
        auto ndim = check_ndim(arg1, args...);
        if (ndim == 1) {
            Shape shape{arg1.shape()[0], sizeof...(args) + 1};
            ndarray::array_dynamic::NDArrayDynamic<DType> result{shape};
            Size indexTarget = 0;
            for (Size indexSource = 0; indexSource < arg1.size(); ++indexSource) {
                fill(result, indexSource, indexTarget, arg1, args...);
            }
            return result;
        }
        //concatenation along 2nd axis
        ndarray::array_dynamic::NDArrayDynamic<DType> result = arg1.copy();
        for (const auto &array: {args...}) {
            result = result.concatenate(array, 1);
        }
        return result;
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
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto c_(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2) {
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
    template<typename DType, typename Derived, typename Storage>
    auto hsplit(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, std::size_t sections) {
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
    template<typename DType, typename Derived, typename Storage>
    auto vsplit(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, std::size_t sections) {
        return array.vsplit(sections);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Expand the shape of an array.
    //
    /// Insert a new axis that will appear at the axis position in the expanded array shape.
    ///
    /// \param a Input array
    /// \param axis Position in the expanded axes where the new axis (or axes) is placed.
    ///
    /// \return View of a with the number of dimensions increased.
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, typename Derived, typename Storage>
    auto expand_dims(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &a, Size axis) {
        return a.expand_dims(axis);
    }

}// namespace np

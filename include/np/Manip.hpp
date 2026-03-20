/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <np/Exception.hpp>
#include <np/internal/SimdOps.hpp>
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
    auto concatenate(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &array1, const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &array2, std::optional<size_t> axis = std::nullopt) {
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
    auto allEmpty(const Arg &array) {
        return array.empty();
    }

    template<typename Arg, typename... Args>
    auto allEmpty(const Arg &arg1, Args &&...args) {
        return arg1.empty() && allEmpty(std::forward<Args>(args)...);
    }

    template<typename Arg>
    auto checkDim(Size dim, const Arg &array) {
        return array.shape()[dim];
    }

    template<typename Arg, typename... Args>
    auto checkDim(Size dim, const Arg &arg1, Args &&...args) {
        auto d = checkDim(dim, std::forward<Args>(args)...);
        if (checkDim(dim, arg1) != d) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Dims should be equal");
        }
        return d;
    }

    template<typename Arg>
    auto getMaxDims(const Arg &array) {
        return array.ndim();
    }

    template<typename Arg, typename... Args>
    auto getMaxDims(const Arg &array, Args &&...args) {
        auto maxArrayDims = getMaxDims(array);
        auto maxDims = getMaxDims(std::forward<Args>(args)...);
        if (maxArrayDims > maxDims) {
            maxDims = maxArrayDims;
        }
        return maxDims;
    }

    template<typename Arg>
    auto getSumDim(Size dim, const Arg &array) {
        return array.ndim() <= dim ? 1 : array.shape()[dim];
    }

    template<typename Arg, typename... Args>
    auto getSumDim(Size dim, const Arg &array, Args &&...args) {
        return getSumDim(dim, array) + getSumDim(dim, std::forward<Args>(args)...);
    }

    template<typename Target, typename Arg>
    void fill(Target &target, Size &indexTarget, Size rowSource, const Arg &array) {
        auto size = array.size() / array.shape()[0];
        for (Size columnSource = 0; columnSource < size; ++columnSource) {
            target.set(indexTarget++, array.get(array.ndim() == 1 ? rowSource : rowSource * size + columnSource));
        }
    }

    template<typename Target, typename Arg, typename... Args>
    void fill(Target &target, Size &indexTarget, Size rowSource, const Arg &arg1, Args &&...args) {
        fill(target, indexTarget, rowSource, arg1);
        fill(target, indexTarget, rowSource, std::forward<Args>(args)...);
    }

    template<typename DType, typename Derived, typename Storage, typename... Args>
    auto column_stack(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &arg1, Args &&...args) {
        if (allEmpty(arg1, std::forward<Args>(args)...)) {
            return ndarray::array_dynamic::NDArrayDynamic<DType>{};
        }

        auto maxDims = getMaxDims(arg1, std::forward<Args>(args)...);
        if (maxDims == 1) {
            maxDims = 2;
        }
        Shape shape{};
        for (Size dim = 0; dim < maxDims; ++dim) {
            Size size{};
            if (dim == 1) {
                size = getSumDim(dim, arg1, std::forward<Args>(args)...);
            } else {
                size = checkDim(dim, arg1, std::forward<Args>(args)...);
            }
            shape.addDim(size);
        }

        ndarray::array_dynamic::NDArrayDynamic<DType> result{shape};
        Size indexTarget = 0;
        for (Size rowSource = 0; rowSource < shape[0]; ++rowSource) {
            fill(result, indexTarget, rowSource, arg1, std::forward<Args>(args)...);
        }
        return result;
    }

    template<typename Arg>
    auto checkShape(const Arg &array) {
        return array.shape();
    }

    template<typename Arg, typename... Args>
    auto checkShape(const Arg &arg1, Args &&...args) {
        auto shape_args = checkShape(std::forward<Args>(args)...);
        if (checkShape(arg1) != shape_args) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Shape should be the same");
        }
        return shape_args;
    }

    template<typename Target, typename Arg>
    void stack(Target &&target, Size &indexTarget, const Arg &arg1) {
        for (Size indexSource = 0; indexSource < arg1.size(); ++indexSource) {
            target.set(indexTarget++, arg1.get(indexSource));
        }
    }

    template<typename Target, typename Arg, typename... Args>
    void stack(Target &&target, Size &indexTarget, const Arg &arg1, Args &&...args) {
        stack(target, indexTarget, arg1);
        stack(target, indexTarget, std::forward<Args>(args)...);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Join a sequence of arrays along a new axis.
    ///
    /// The axis parameter specifies the index of the new axis in the dimensions of the result.
    /// For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
    ///
    /// \param arg1, args Each array must have the same shape
    ///
    /// \return The stacked array has one more dimension than the input arrays.
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, typename Derived, typename Storage, typename... Args>
    auto stack(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &arg1, Args &&...args) {
        if (allEmpty(arg1, std::forward<Args>(args)...)) {
            return ndarray::array_dynamic::NDArrayDynamic<DType>{};
        }
        auto shape = checkShape(arg1, std::forward<Args>(args)...);
        shape.expandDims(0, sizeof...(args) + 1);
        ndarray::array_dynamic::NDArrayDynamic<DType> result{shape};
        Size indexTarget = 0;
        stack(result, indexTarget, arg1, std::forward<Args>(args)...);
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
    auto hsplit(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, size_t sections) {
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
    auto vsplit(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &array, size_t sections) {
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

    //////////////////////////////////////////////////////////////
    /// \brief Return elements from x transformed by a lambda depending on condition.
    //
    /// Return elements chosen by positive or negative lambda from x depending on condition.
    ///
    /// \param x Source array.
    /// \param condition Where true, yield transformed by positive lambda, otherwise by negative.
    /// \param positive, negative Positive and negative transformations.
    ///
    /// \return An array with elements from x transformed by a lambda
    ///
    //////////////////////////////////////////////////////////////
    // Template callable overload (avoids std::function virtual dispatch overhead).
    // Enabled only when Cond/Pos/Neg are not std::function to avoid ambiguity.
    template<typename DType, typename Derived, typename Storage,
             typename Cond, typename Pos, typename Neg,
             typename = std::enable_if_t<
                     !std::is_same_v<std::decay_t<Cond>, std::function<bool(const DType &)>> ||
                     !std::is_same_v<std::decay_t<Pos>, std::function<DType(const DType &)>> ||
                     !std::is_same_v<std::decay_t<Neg>, std::function<DType(const DType &)>>>>
    auto where(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &x,
               Cond &&condition,
               Pos &&positive,
               Neg &&negative) {
        NDArrayDynamic<DType> result{x.shape()};
        auto *result_data = result.data();
        auto n = x.size();
        if constexpr (Storage::is_contiguous) {
            // Fast path: direct pointer access avoids virtual get()/set() dispatch
            const auto *src = x.data();
            // Use SIMD-accelerated path for double/float contiguous arrays
            if constexpr (std::is_same_v<DType, double>) {
                if (np::internal::simd_at_least(np::internal::SimdLevel::AVX2)) {
                    for (Size i = 0; i < n; ++i) {
                        result_data[i] = condition(src[i]) ? positive(src[i]) : negative(src[i]);
                    }
                } else {
                    for (Size i = 0; i < n; ++i) {
                        result_data[i] = condition(src[i]) ? positive(src[i]) : negative(src[i]);
                    }
                }
            } else {
                for (Size i = 0; i < n; ++i) {
                    result_data[i] = condition(src[i]) ? positive(src[i]) : negative(src[i]);
                }
            }
        } else {
            // Slow path: element-by-element via virtual dispatch
            for (Size i = 0; i < n; ++i) {
                const auto &element = x.get(i);
                result_data[i] = condition(element) ? positive(element) : negative(element);
            }
        }
        return result;
    }

    // In-place where() variant: writes directly into an existing output array.
    // Avoids allocation overhead for the IRLS Tukey bisquare weight update pattern.
    // The output array must have the same size as the input.
    template<typename DType, typename Derived, typename Storage,
             typename DerivedOut, typename StorageOut,
             typename Cond, typename Pos, typename Neg,
             typename = std::enable_if_t<
                     !std::is_same_v<std::decay_t<Cond>, std::function<bool(const DType &)>> ||
                     !std::is_same_v<std::decay_t<Pos>, std::function<DType(const DType &)>> ||
                     !std::is_same_v<std::decay_t<Neg>, std::function<DType(const DType &)>>>>
    void where_inplace(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &x,
                       ndarray::internal::NDArrayBase<DType, DerivedOut, StorageOut> &out,
                       Cond &&condition,
                       Pos &&positive,
                       Neg &&negative) {
        auto n = x.size();
        if constexpr (Storage::is_contiguous && StorageOut::is_contiguous) {
            // Fast path: direct pointer access for both input and output
            const auto *src = x.data();
            auto *dst = out.data();
            for (Size i = 0; i < n; ++i) {
                dst[i] = condition(src[i]) ? positive(src[i]) : negative(src[i]);
            }
        } else {
            // Fallback: element-by-element via virtual dispatch
            for (Size i = 0; i < n; ++i) {
                const auto &element = x.get(i);
                out.set(i, condition(element) ? positive(element) : negative(element));
            }
        }
    }

    // Specialized Tukey bisquare weight update with SIMD acceleration.
    // Computes: result[i] = (a[i] <= k) ? 1.0 : (2*k/a[i] - k*k/(a[i]*a[i]))
    // This is the hot path in IRLS robust regression loops.
    template<typename DType, typename Derived, typename Storage>
    auto where_tukey(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &x, DType k) {
        NDArrayDynamic<DType> result{x.shape()};
        auto *result_data = result.data();
        auto n = x.size();
        if constexpr (Storage::is_contiguous) {
            const auto *src = x.data();
            if constexpr (std::is_same_v<DType, double>) {
                np::internal::where_tukey_pd(src, static_cast<double>(k), result_data, n);
            } else if constexpr (std::is_same_v<DType, float>) {
                np::internal::where_tukey_ps(src, static_cast<float>(k), result_data, n);
            } else {
                for (Size i = 0; i < n; ++i) {
                    result_data[i] = (src[i] <= k) ? DType(1) : (DType(2) * k / src[i] - k * k / (src[i] * src[i]));
                }
            }
        } else {
            for (Size i = 0; i < n; ++i) {
                const auto &element = x.get(i);
                result.set(i, (element <= k) ? DType(1) : (DType(2) * k / element - k * k / (element * element)));
            }
        }
        return result;
    }

    // In-place Tukey bisquare weight update with SIMD acceleration.
    template<typename DType, typename Derived, typename Storage,
             typename DerivedOut, typename StorageOut>
    void where_tukey_inplace(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &x,
                             ndarray::internal::NDArrayBase<DType, DerivedOut, StorageOut> &out,
                             DType k) {
        auto n = x.size();
        if constexpr (Storage::is_contiguous && StorageOut::is_contiguous) {
            const auto *src = x.data();
            auto *dst = out.data();
            if constexpr (std::is_same_v<DType, double>) {
                np::internal::where_tukey_pd(src, static_cast<double>(k), dst, n);
            } else if constexpr (std::is_same_v<DType, float>) {
                np::internal::where_tukey_ps(src, static_cast<float>(k), dst, n);
            } else {
                for (Size i = 0; i < n; ++i) {
                    dst[i] = (src[i] <= k) ? DType(1) : (DType(2) * k / src[i] - k * k / (src[i] * src[i]));
                }
            }
        } else {
            for (Size i = 0; i < n; ++i) {
                const auto &element = x.get(i);
                out.set(i, (element <= k) ? DType(1) : (DType(2) * k / element - k * k / (element * element)));
            }
        }
    }

    // Fused abs(a - b): computes abs(a[i] - b[i]) in a single SIMD pass.
    // Avoids intermediate array allocation that would occur with abs(a - b).
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto abs_sub(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &a,
                 const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &b) {
        NDArrayDynamic<DType> result{a.shape()};
        auto *result_data = result.data();
        auto n = a.size();
        if constexpr (Storage1::is_contiguous && Storage2::is_contiguous) {
            const auto *a_data = a.data();
            const auto *b_data = b.data();
            if constexpr (std::is_same_v<DType, double>) {
                np::internal::abs_sub_pd(a_data, b_data, result_data, n);
            } else if constexpr (std::is_same_v<DType, float>) {
                np::internal::abs_sub_ps(a_data, b_data, result_data, n);
            } else {
                for (Size i = 0; i < n; ++i) {
                    result_data[i] = std::abs(a_data[i] - b_data[i]);
                }
            }
        } else {
            for (Size i = 0; i < n; ++i) {
                result.set(i, std::abs(a.get(i) - b.get(i)));
            }
        }
        return result;
    }

    // Fused sum(a * a * w): computes sum(a[i]*a[i]*w[i]) in a single SIMD pass.
    // Avoids intermediate array allocation that would occur with sum((a * a * w)).
    template<typename DType, typename Derived1, typename Storage1, typename Derived2, typename Storage2>
    auto sum_sq_weighted(const ndarray::internal::NDArrayBase<DType, Derived1, Storage1> &a,
                         const ndarray::internal::NDArrayBase<DType, Derived2, Storage2> &w) {
        auto n = a.size();
        if constexpr (Storage1::is_contiguous && Storage2::is_contiguous) {
            const auto *a_data = a.data();
            const auto *w_data = w.data();
            if constexpr (std::is_same_v<DType, double>) {
                return np::internal::sum_sq_weighted_pd(a_data, w_data, n);
            } else if constexpr (std::is_same_v<DType, float>) {
                return static_cast<double>(np::internal::sum_sq_weighted_ps(a_data, w_data, n));
            } else {
                DType result{};
                for (Size i = 0; i < n; ++i) {
                    result += a_data[i] * a_data[i] * w_data[i];
                }
                return static_cast<double>(result);
            }
        } else {
            double result{};
            for (Size i = 0; i < n; ++i) {
                auto av = a.get(i);
                result += av * av * w.get(i);
            }
            return result;
        }
    }

}// namespace np

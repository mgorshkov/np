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

#include <cstddef>
#include <vector>
#include <tuple>
#include <array>
#include <ostream>
#include <optional>
#include <type_traits>

#include <np/Axis.hpp>
#include <np/Shape.hpp>

#include <np/internal/Tools.hpp>

#include <np/ndarray/static/internal/NDArrayStaticInternal.hpp>

namespace np::ndarray::array_static {

    template <typename DType, Size... SizeTs>
    class NDArrayStatic;

    template <typename DType, Size SizeT, Size... SizeTs>
    void set(NDArrayStatic<DType, SizeT, SizeTs...> &array, Size i, const typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType& data);

    // Termination template
    template <typename DType>
    class NDArrayStaticStub {
    public:
        using CArrayType = DType[1]; // ISO C++ forbids zero-size array [-Werror=pedantic]
        using StdArrayType = std::array<DType, 1>;
        using StdVectorType = std::vector<DType>;

        inline NDArrayStaticStub() noexcept {
        }

        inline NDArrayStaticStub(const DType& data) 
            : m_ArrayImpl{data}
        {
        }

        // Array dimensions
        Shape shape() const {
            return Shape{1};
        }

        inline bool array_equal(const DType& element) const {
            return np::array_equal(m_ArrayImpl, element);
        }

        inline bool array_equal(const NDArrayStaticStub &array) const {
            return np::array_equal(m_ArrayImpl, array.m_ArrayImpl);
        }

        inline DType sum() const {
            return m_ArrayImpl;
        }

        inline DType minimum() const {
            return m_ArrayImpl;
        }

        inline DType maximum() const {
            return m_ArrayImpl;
        }

        inline auto cumsum() const {
            return m_ArrayImpl;
        }

        inline DType mean() const {
            return m_ArrayImpl;
        }

        inline DType median() const {
            return m_ArrayImpl;
        }

        inline DType corrcoef() const {
            return m_ArrayImpl;
        }

        inline DType std_() const {
            return m_ArrayImpl;
        }

        inline operator DType() const {
            return m_ArrayImpl;
        }

        inline DType get() const {
            return m_ArrayImpl;
        }

        inline DType ravel() const {
            return m_ArrayImpl;
        }
        
        inline bool operator == (const NDArrayStaticStub& other) const {
            return m_ArrayImpl == other.m_ArrayImpl;
        }

        inline bool operator > (const NDArrayStaticStub& other) const {
            return m_ArrayImpl > other.m_ArrayImpl;
        }

        inline bool operator < (const NDArrayStaticStub& other) const {
            return m_ArrayImpl < other.m_ArrayImpl;
        }

        template <typename DTypeOther, Size SizeTOther, Size... SizeTsOther> 
        friend inline void set(NDArrayStatic<DTypeOther, SizeTOther, SizeTsOther...> &array, Size i, 
            const typename NDArrayStatic<DTypeOther, SizeTOther, SizeTsOther...>::ReducedType& data);

        friend inline bool array_equal(const NDArrayStaticStub<double> &value1, const NDArrayStaticStub<double> &value2);

    private:
        DType m_ArrayImpl;
    };

    inline bool array_equal(const NDArrayStaticStub<double> &value1, const NDArrayStaticStub<double> &value2) {
        return np::internal::almost_equal(value1.m_ArrayImpl, value2.m_ArrayImpl, ULP_TOLERANCE);
    }

    template <typename DType, Size SizeT, Size... SizeTs>
    class NDArrayStatic<DType, SizeT, SizeTs...> {
    public:
        using ReducedNDArray = NDArrayStatic<DType, SizeTs...>;

        using ReducedType = typename std::conditional<
            sizeof...(SizeTs) == 0,
            NDArrayStaticStub<DType>,
            ReducedNDArray>::type;

        using ReducedCArrayType = typename std::conditional<
            sizeof...(SizeTs) == 0,
            DType,
            typename ReducedType::CArrayType>::type;
            
        using ReducedStdArrayType = typename std::conditional<
            sizeof...(SizeTs) == 0,
            DType,
            typename ReducedType::StdArrayType>::type;

        using ReducedStdVectorType = typename std::conditional<
            sizeof...(SizeTs) == 0,
            DType,
            typename ReducedType::StdVectorType>::type;
                
        using CArrayType = ReducedCArrayType[SizeT];
        using StdArrayType = std::array<ReducedStdArrayType, SizeT>;
        using StdVectorType = std::vector<ReducedStdVectorType>;

        // Creating arrays
        inline NDArrayStatic() noexcept;

        inline explicit NDArrayStatic(const DType &value) noexcept;

        inline NDArrayStatic(CArrayType data) noexcept;

        inline NDArrayStatic(const NDArrayStatic &another) noexcept;

        inline NDArrayStatic(NDArrayStatic &&another) noexcept;

        inline explicit NDArrayStatic(const internal::NDArrayStaticInternal<DType, SizeT, SizeTs...> &array) noexcept;

        inline explicit NDArrayStatic(internal::NDArrayStaticInternal<DType, SizeT, SizeTs...> &&array) noexcept;

        inline explicit NDArrayStatic(const StdArrayType &array) noexcept;

        inline explicit NDArrayStatic(StdArrayType &&array) noexcept;

        inline explicit NDArrayStatic(const StdVectorType &vector) noexcept;

        inline explicit NDArrayStatic(StdVectorType &&vector) noexcept;

        inline explicit NDArrayStatic(std::initializer_list<DType> init_list) noexcept;

        inline ~NDArrayStatic() noexcept;

        inline NDArrayStatic &operator=(const NDArrayStatic &another) noexcept;

        inline NDArrayStatic &operator=(NDArrayStatic &&another) noexcept;

        inline NDArrayStatic &operator=(const StdVectorType &vector) noexcept;

        // Indexing arrays
        template <typename DTypeOther, Size SizeTOther, Size... SizeTsOther> 
        friend inline void set(NDArrayStatic<DTypeOther, SizeTOther, SizeTsOther...> &array, Size i, 
            const typename NDArrayStatic<DTypeOther, SizeTOther, SizeTsOther...>::ReducedType& data);

        inline ReducedType operator[](Size i) const;

        //TODO
        // inline ReducedType& operator[](const std::string& i);
        // inline ReducedType operator[](const std::string& i) const;

        inline ReducedType at(Size i) const;
        
        //TODO
        //inline ReducedType& at(const std::string& i);
        //inline ReducedType at(const std::string& i) const;

        // Stream output
        inline friend std::ostream &operator<<(std::ostream &stream, const NDArrayStatic &array) {
            return stream << array.m_ArrayImpl;
        }

        // Save data
        // For static arrays only save is implemented, they are loaded as dynamic arras
        inline void save(const char* filename);

        inline void savez(const char* filename);

        inline void savetxt(const char* filename, const char* delimiter);

        // Array dimensions
        Shape shape() const;

        // Array length
        Size len() const;

        // Number of array dimensions
        inline Size ndim();

        // Number of array elements
        inline Size size();

        // Data type of array elements
        inline constexpr DType dtype();

        // Convert an array to a different type
        template<typename DTypeNew>
        inline NDArrayStatic<DTypeNew, SizeT, SizeTs...> astype() const;

        // Array mathematics
        inline NDArrayStatic<DType, SizeT, SizeTs...> operator + (const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> add(const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> operator - (const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> subtract(const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> operator * (const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> multiply(const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> operator / (const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> divide(const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> exp(const NDArrayStatic& array) const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> sqrt() const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> sin() const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> cos() const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> log() const;
        // Dot product
        inline NDArrayStatic<DType, SizeT, SizeTs...> dot(const NDArrayStatic &array) const;

        // Elementwise comparison
        inline NDArrayStatic<bool_, SizeT, SizeTs...> operator==(const NDArrayStatic &array) const;
        inline NDArrayStatic<bool_, SizeT, SizeTs...> operator<(const NDArrayStatic &array) const;
        inline NDArrayStatic<bool_, SizeT, SizeTs...> operator>(const NDArrayStatic &array) const;

        // Array-wise comparison
        inline bool array_equal(const DType& element) const;
        inline bool array_equal(const NDArrayStatic &array) const;
        // Aggregate functions
        // Array-wise sum
        inline DType sum() const;

        // Array-wise minimum value
        inline DType min() const;

        // Maximum value of an Array row
        inline DType max() const;

        // Cumulative sum of the elements
        inline auto cumsum() const;

        // Mean
        inline DType mean() const;

        // Median
        inline DType median() const;

        // Covariance
        inline auto cov() const;

        // Correlation coefficient
        inline auto corrcoef() const;

        // Compute the standard deviation along the specified axis.
        inline DType std_() const;

        // Create a view of the array with the same data
        inline NDArrayStatic<DType, SizeT, SizeTs...> view() const;

        // Create a deep copy of the array
        inline NDArrayStatic<DType, SizeT, SizeTs...> copy() const;

        // Sort an array
        inline void sort();

        // template<Size N>
        // inline void sort(Axis<N> axis = Axis<0>{});

        // Permute array dimensions
        inline NDArrayStatic<DType, SizeT, SizeTs...> transpose() const;
        inline NDArrayStatic<DType, SizeT, SizeTs...> T() const;

        // Flatten the array
        inline auto ravel() const;

        // Reshape, but don’t change data
        inline NDArrayStatic<DType, SizeT, SizeTs...> reshape(Shape shape) const;

        // Adding and removing elements
        // Return a new array with shape (2, 6)
        inline NDArrayStatic<DType, SizeT, SizeTs...> resize(Shape shape) const;

        // Append items to the array
        inline auto append(const NDArrayStatic& array) const;

        // Insert items in the array
        inline NDArrayStatic<DType, SizeT, SizeTs...> insert(Size index, const NDArrayStatic& array) const;

        // Delete items from the array
        inline NDArrayStatic<DType, SizeT, SizeTs...> del(Size index) const;

        // Concatenate arrays
        inline NDArrayStatic<DType, SizeT, SizeTs...> concatenate(const NDArrayStatic& array) const;

        // Stack arrays vertically (row-wise)
        inline NDArrayStatic<DType, SizeT, SizeTs...> vstack(const NDArrayStatic& array) const;

        // Stack arrays vertically (row-wise)
        inline NDArrayStatic<DType, SizeT, SizeTs...> r_(const NDArrayStatic& array) const;

        // Stack arrays horizontally (column-wise)
        inline NDArrayStatic<DType, SizeT, SizeTs...> hstack(const NDArrayStatic& array) const;

        // Create stacked column-wise arrays
        inline NDArrayStatic<DType, SizeT, SizeTs...> column_stack(const NDArrayStatic& array) const;

        // Create stacked column-wise arrays
        inline NDArrayStatic<DType, SizeT, SizeTs...> c_(const NDArrayStatic& array) const;

        // Split the array horizontally
        inline std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> hsplit(Size index) const;

        // Split the array vertically
        inline std::vector<NDArrayStatic<DType, SizeT, SizeTs...>> vsplit(Size index) const;

    private:
        inline void save(std::ostream& stream);
        
        internal::NDArrayStaticInternal<DType, SizeT, SizeTs...> m_ArrayImpl;
    
        static const constexpr std::tuple m_Shape = std::make_tuple(SizeT, SizeTs...);
    };
}
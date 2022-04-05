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
#include <initializer_list>
#include <iostream>
#include <array>
#include <utility>
#include <vector>
#include <iterator>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np::ndarray::array_dynamic::internal {

    template <typename DType, typename Storage>
    class NDArrayDynamicInternal;

    template <typename DType, typename Storage>
    class NDArrayDynamic;

    template <typename DType, typename Storage>
    inline std::ostream & operator<< (std::ostream &stream, const NDArrayDynamicInternal<DType, Storage> &array);

    template <typename DType>
    using NDArrayDynamicInternalStorageVector = std::vector<DType>;

    template <typename Storage>
    struct Span {
        Span() {

        }

        Span(typename Storage::const_iterator begin, typename Storage::const_iterator end) 
            : begin_{begin}
            , end_  {end}
        {
        }

        Span(typename Storage::const_iterator begin, std::size_t size)
            : begin_{begin}
            , end_  {begin + size}
        {
        }

        template <typename DType>
        Span(const std::vector<DType> vector)
            : begin_{vector.begin()}
            , end_{vector.end()}
        {   
        }

        std::size_t size() const {
            return end_ - begin_;
        }

        typename Storage::const_iterator begin() const {
            return begin_;
        }

        typename Storage::const_iterator end() const {
            return end_;
        }

        typename Storage::value_type operator[](std::size_t i) const {
            return *(begin_ + i);
        }

        bool operator == (const Storage& storage) const {
            auto it1 = storage.begin();
            auto it2 = begin_;
            while (it1 != storage.end()) {
                if (*it1 != *it2)
                    return false;
                ++it1;
                ++it2;
            }
            return it2 == end_;
        }
    
        typename Storage::const_iterator begin_;
        typename Storage::const_iterator end_;
    };

    template <typename DType>
    using NDArrayDynamicInternalStorageSpan = Span<NDArrayDynamicInternalStorageVector<DType>>;

    template<typename DType, typename Storage = NDArrayDynamicInternalStorageVector<DType>>
    class NDArrayDynamicInternal {
    public:
        template<std::size_t Size1T>
        using CArray1DType = DType[Size1T];
        template<std::size_t Size1T, std::size_t Size2T>
        using CArray2DType = CArray1DType<Size1T>[Size2T];
        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        using CArray3DType = CArray2DType<Size1T, Size2T>[Size3T];
        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        using CArray4DType = CArray3DType<Size1T, Size2T, Size3T>[Size4T];

        template<std::size_t Size1T>
        using StdArray1DType = std::array<DType, Size1T>;
        template<std::size_t Size1T, std::size_t Size2T>
        using StdArray2DType = std::array<StdArray1DType<Size1T>, Size2T>;
        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        using StdArray3DType = std::array<StdArray2DType<Size1T, Size2T>, Size3T>;
        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        using StdArray4DType = std::array<StdArray3DType<Size1T, Size2T, Size3T>, Size4T>;

        using StdVector1DType = std::vector<DType>;
        using StdVector2DType = std::vector<StdVector1DType>;
        using StdVector3DType = std::vector<StdVector2DType>;
        using StdVector4DType = std::vector<StdVector3DType>;

        // Creating Arrays
        inline NDArrayDynamicInternal() noexcept 
            : m_Shape{}
            , m_Impl{}
        {
        }

        inline explicit NDArrayDynamicInternal(Shape shape) noexcept
            : m_Shape{std::move(shape)}
            , m_Impl{}
        {
            std::size_t size = np::ndarray::internal::calcSizeByShape(m_Shape);
            m_Impl.resize(size);
        }

        inline explicit NDArrayDynamicInternal(Shape shape, const DType &value)
            : m_Shape{std::move(shape)}
        {
            std::size_t size = np::ndarray::internal::calcSizeByShape(m_Shape);
            m_Impl = std::move(std::vector(size, value));
        }

        inline explicit NDArrayDynamicInternal(const DType &value)
            : m_Shape{1}
            , m_Impl{1, value}
        {
        }

        template <typename InternalStorage>
        inline NDArrayDynamicInternal(const NDArrayDynamicInternal<DType, InternalStorage> &another) noexcept
            : m_Shape{another.m_Shape}
            , m_Impl{another.m_Impl}
        {
        }

        template <typename InternalStorage>
        inline NDArrayDynamicInternal(NDArrayDynamicInternal<DType, InternalStorage> &&another) noexcept
            : m_Shape{std::move(another.m_Shape)}
            , m_Impl{std::move(another.m_Impl)}
        {
        }

        // create 1D array
        template <typename It>
        inline NDArrayDynamicInternal(It it, Size size) noexcept
            : m_Shape{size}
            , m_Impl(it, it + size)
        {
        }

        template <typename It>
        inline NDArrayDynamicInternal(It it, const Shape& shape) noexcept
            : m_Shape{shape}
            , m_Impl(it, it + ndarray::internal::calcSizeByShape(shape))
        {
        }

        inline NDArrayDynamicInternal(const StdVector1DType& impl, Shape shape) noexcept
            : m_Shape{std::move(shape)}
            , m_Impl{impl}
        {
        }

        template <std::size_t Size1T>
        inline NDArrayDynamicInternal(const CArray1DType<Size1T>& array) noexcept
            : m_Shape{Size1T}
            , m_Impl{std::begin(array), std::end(array)} {
        }

        template<std::size_t Size1T>
        inline NDArrayDynamicInternal(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept
            : NDArrayDynamicInternal{array}
        {
            if (isColumnVector) {
                m_Shape.insert(m_Shape.begin(), 1);
            }
        }

        template <std::size_t Size1T, std::size_t Size2T>
        inline NDArrayDynamicInternal(const CArray2DType<Size1T, Size2T>& array) noexcept
             : m_Shape{Size2T, Size1T}
        {
            for (std::size_t i = 0; i < Size2T; ++i) {
                std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        inline NDArrayDynamicInternal(const CArray3DType<Size1T, Size2T, Size3T>& array) noexcept
            : m_Shape{Size3T, Size2T, Size1T}
        {
            for (std::size_t i = 0; i < Size3T; ++i) {
                for (std::size_t j = 0; j < Size2T; ++j) {
                    std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                }
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        inline NDArrayDynamicInternal(const CArray4DType<Size1T, Size2T, Size3T, Size4T>& array) noexcept
            : m_Shape{Size4T, Size3T, Size2T, Size1T}
        {
            for (std::size_t i = 0; i < Size4T; ++i) {
                for (std::size_t j = 0; j < Size3T; ++j) {
                    for (std::size_t k = 0; k < Size2T; ++k) {
                        std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), std::back_inserter(m_Impl));
                    }
                }
            }
        }

        template<std::size_t Size1T>
        inline explicit NDArrayDynamicInternal(const StdArray1DType<Size1T> &array) noexcept
            : m_Shape{Size1T}
            , m_Impl{std::begin(array), std::end(array)} {
        }

        template<std::size_t Size1T>
        inline NDArrayDynamicInternal(const StdArray1DType<Size1T> &array, bool isColumnVector) noexcept
            : NDArrayDynamicInternal{array}
        {
            if (isColumnVector) {
                m_Shape.insert(m_Shape.begin(), 1);
            }
        }

        template<std::size_t Size1T, std::size_t Size2T>
        inline explicit NDArrayDynamicInternal(const StdArray2DType<Size1T, Size2T> &array) noexcept {
            m_Shape = {Size2T, Size1T};

            for (std::size_t i = 0; i < Size2T; ++i) {
                std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        inline explicit NDArrayDynamicInternal(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
            m_Shape = {Size3T, Size2T, Size1T};

            for (std::size_t i = 0; i < Size3T; ++i) {
                for (std::size_t j = 0; j < Size2T; ++j) {
                    std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                }
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        inline explicit
        NDArrayDynamicInternal(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
            m_Shape = {Size4T, Size3T, Size2T, Size1T};

            for (std::size_t i = 0; i < Size4T; ++i) {
                for (std::size_t j = 0; j < Size3T; ++j) {
                    for (std::size_t k = 0; k < Size2T; ++k) {
                        std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), std::back_inserter(m_Impl));
                    }
                }
            }
        }

        inline explicit NDArrayDynamicInternal(const StdVector1DType &vector) noexcept
            : m_Shape{static_cast<Size>(vector.size())}
            , m_Impl{vector}
        {
        }

        inline explicit NDArrayDynamicInternal(const StdVector1DType &vector, bool isColumnVector) noexcept
            : NDArrayDynamicInternal{vector}
        {
            if (isColumnVector) {
                m_Shape.insert(m_Shape.begin(), 1);
            }
        }

        inline explicit NDArrayDynamicInternal(const StdVector2DType &vector) noexcept
        {
            m_Shape = {static_cast<Size>(vector.size())};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                std::copy(std::begin(vector[i]), std::end(vector[i]), std::back_inserter(m_Impl));
            }
        }

        inline explicit NDArrayDynamicInternal(const StdVector3DType &vector) noexcept
        {
            m_Shape = {static_cast<Size>(vector.size())};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
                if (!vector[0].empty()) {
                    m_Shape.push_back(vector[0][0].size());
                }
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                for (std::size_t j = 0; j < vector[i].size(); ++j) {
                    std::copy(std::begin(vector[i][j]), std::end(vector[i][j]), std::back_inserter(m_Impl));
                }
            }
        }

        inline explicit NDArrayDynamicInternal(const StdVector4DType &vector) noexcept {
            m_Shape = {static_cast<Size>(vector.size())};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
                if (!vector[0].empty()) {
                    m_Shape.push_back(vector[0][0].size());
                    if (!vector[0][0].empty()) {
                        m_Shape.push_back(vector[0][0][0].size());
                    }
                }
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                for (std::size_t j = 0; j < vector[i].size(); ++j) {
                    for (std::size_t k = 0; j < vector[j].size(); ++k) {
                        std::copy(std::begin(vector[i][j][k]), std::end(vector[i][j][k]), std::back_inserter(m_Impl));
                    }
                }
            }
        }

        inline NDArrayDynamicInternal(std::initializer_list<DType> init_list) noexcept
            : m_Shape{static_cast<Size>(init_list.size())}
            , m_Impl{init_list} {
        }

        inline NDArrayDynamicInternal &operator=(const DType &value) noexcept {
            m_Shape = {1};
            m_Impl.reserve(1);
            m_Impl.fill(value);
            return *this;
        }

        template<std::size_t Size1T>
        inline NDArrayDynamicInternal &operator=(CArray1DType<Size1T> array) noexcept {
            m_Shape = {Size1T};
            m_Impl = std::vector{array, Size1T * sizeof(DType)};
        }

        template<std::size_t Size1T, std::size_t Size2T>
        inline NDArrayDynamicInternal &operator=(CArray2DType<Size1T, Size2T> array) noexcept {
            m_Shape = {Size2T, Size1T};

            for (std::size_t i = 0; i < Size2T; ++i) {
                std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        inline NDArrayDynamicInternal &operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept {
            m_Shape = {Size3T, Size2T, Size1T};
            for (std::size_t i = 0; i < Size3T; ++i) {
                for (std::size_t j = 0; j < Size2T; ++j) {
                    std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                }
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        inline NDArrayDynamicInternal &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept {
            m_Shape = {Size4T, Size3T, Size2T, Size1T};
            for (std::size_t i = 0; i < Size4T; ++i) {
                for (std::size_t j = 0; j < Size3T; ++j) {
                    for (std::size_t k = 0; k < Size2T; ++k) {
                        std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                    }
                }
            }
        }

        template<std::size_t Size1T>
        inline NDArrayDynamicInternal &operator=(const StdArray1DType<Size1T> &array) noexcept {
            m_Shape = {Size1T};
            m_Impl = std::vector{array.data(), Size1T};
            return *this;
        }

        template<std::size_t Size1T, std::size_t Size2T>
        inline NDArrayDynamicInternal &operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept {
            m_Shape = {Size2T, Size1T};

            for (std::size_t i = 0; i < Size2T; ++i) {
                std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
        inline NDArrayDynamicInternal &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
            m_Shape = {Size3T, Size2T, Size1T};
            for (std::size_t i = 0; i < Size3T; ++i) {
                for (std::size_t j = 0; j < Size2T; ++j) {
                    std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                }
            }
        }

        template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
        inline NDArrayDynamicInternal &
        operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
            m_Shape = {Size4T, Size3T, Size2T, Size1T};
            for (std::size_t i = 0; i < Size4T; ++i) {
                for (std::size_t j = 0; j < Size3T; ++j) {
                    for (std::size_t k = 0; k < Size2T; ++k) {
                        std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                    }
                }
            }
        }

        inline NDArrayDynamicInternal &operator=(const StdVector1DType &vector) noexcept {
            m_Shape = {vector.size()};
            m_Impl = vector;
            return *this;
        }

        inline NDArrayDynamicInternal &operator=(const StdVector2DType &vector) noexcept {
            m_Shape = {vector.size()};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                std::copy(std::begin(vector[i]), std::end(vector[i]), std::back_inserter(m_Impl));
            }
            return *this;
        }

        inline NDArrayDynamicInternal &operator=(const StdVector3DType &vector) noexcept {
            m_Shape = {vector.size()};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
                if (!vector[0].empty()) {
                    m_Shape.push_back(vector[0][0].size());
                }
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                for (std::size_t j = 0; j < vector[i].size(); ++j) {
                    std::copy(std::begin(vector[i][j]), std::end(vector[i][j]), std::back_inserter(m_Impl));
                }
            }
            return *this;
        }

        inline NDArrayDynamicInternal &operator=(const StdVector4DType &vector) noexcept {
            m_Shape = {vector.size()};
            if (!vector.empty()) {
                m_Shape.push_back(vector[0].size());
                if (!vector[0].empty()) {
                    m_Shape.push_back(vector[0][0].size());
                    if (!vector[0][0].empty()) {
                        m_Shape.push_back(vector[0][0][0].size());
                    }
                }
            }

            for (std::size_t i = 0; i < vector.size(); ++i) {
                for (std::size_t j = 0; j < vector[i].size(); ++j) {
                    for (std::size_t k = 0; k < vector[j].size(); ++k) {
                        std::copy(std::begin(vector[i][j][k]), std::end(vector[i][j][k]), std::back_inserter(m_Impl));
                    }
                }
            }
            return *this;
        }

        inline NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>> operator[](std::size_t i) const {
            if (m_Shape.size() == 0) {
                throw std::runtime_error("Index " + std::to_string(i) + " of an empty array requested");
            }
            auto itBegin = m_Impl.begin();
            if (m_Shape.size() == 1) {
                return NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>>(itBegin, 1);
            }

            auto startIt = std::begin(m_Shape);
            std::advance(startIt, 1);
            Shape shape{std::vector(startIt, std::end(m_Shape))};
            auto layerSize = m_Shape[0] == 0 ? 0 : size() / m_Shape[0];
            std::advance(itBegin, i * layerSize);
            return NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>>(itBegin, shape);
        }

        inline DType get(std::size_t i) const {
            return m_Impl[i];
        }

        inline void set(std::size_t i, const DType& value) {
            m_Impl[i] = value;
        }

        friend std::ostream & operator<< <DType, Storage>(std::ostream &stream, const NDArrayDynamicInternal<DType, Storage> &array);

        inline Shape getShape() const {
            return m_Shape;
        }

        inline std::size_t size() const {
            return m_Impl.size();
        }

        inline void dumpToStreamAsBinary(std::ostream &stream) {
            for (std::size_t index = 0; index < size(); ++index) {
                stream.write(reinterpret_cast<const char*>(&m_Impl[index]), sizeof(m_Impl[index]));
            }
        }

        inline bool operator == (const DType& value) const {
            if (size() == 1) {
                return m_Impl[0] == value;
            }
            return false;
        }

        inline bool operator != (const DType& value) const {
            return !operator == (value);
        }

        inline bool operator == (const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>>& other) const {
            return m_Shape == other.m_Shape && m_Impl == other.m_Impl;
        }

        inline bool operator == (const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageVector<DType>>& other) const {
            return m_Shape == other.m_Shape && m_Impl == other.m_Impl;
        }

        inline bool operator != (const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>>& other) const {
            return !operator == (other);
        }

        inline bool operator != (const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageVector<DType>>& other) const {
            return !operator == (other);
        }

        inline void sort() {
            std::sort(m_Impl.begin(), m_Impl.end());
        }

        friend class NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageVector<DType>>;
        friend class NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>>;

        friend class NDArrayDynamic<DType, NDArrayDynamicInternalStorageVector<DType>>;
        friend class NDArrayDynamic<DType, NDArrayDynamicInternalStorageSpan<DType>>;

        //TODO fix this. access from NDArrayDynamic
    //private:
        Shape m_Shape;
        Storage m_Impl;
    };

} // namespace np::dynamic::internal
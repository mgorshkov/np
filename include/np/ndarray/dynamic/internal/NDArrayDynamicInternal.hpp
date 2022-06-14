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

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/dynamic/internal/Tools.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            namespace internal {

                template<typename DType, typename Storage>
                class NDArrayDynamicInternal;

                template<typename DType, typename Storage>
                class NDArrayDynamic;

                template<typename DType, typename Storage>
                inline std::ostream &operator<<(std::ostream &stream, const NDArrayDynamicInternal<DType, Storage> &array);

                template<typename DType>
                using NDArrayDynamicInternalStorageVector = std::vector<DType>;

                template<typename Storage>
                class ConstSpan {
                public:
                    using iterator = typename Storage::iterator;
                    using const_iterator = typename Storage::const_iterator;

                    typedef ptrdiff_t difference_type;

                    ConstSpan() = default;

                    ConstSpan(const_iterator cbegin, const_iterator cend)
                        : cbegin_{cbegin}, cend_{cend} {
                    }

                    ConstSpan(const_iterator cbegin, Size size)
                        : cbegin_{cbegin}, cend_{cbegin + size} {
                    }

                    template<typename DType>
                    ConstSpan(const std::vector<DType> &vector)
                        : cbegin_{vector.cbegin()}, cend_{vector.cend()} {
                    }

                    std::size_t size() const {
                        return cend_ - cbegin_;
                    }

                    const_iterator cbegin() const {
                        return cbegin_;
                    }

                    const_iterator cend() const {
                        return cend_;
                    }

                    typename const_iterator::reference operator[](std::size_t i) const {
                        return *(cbegin_ + i);
                    }

                    bool operator==(const Storage &storage) const {
                        auto it1 = storage.cbegin();
                        auto it2 = cbegin_;
                        while (it1 != storage.cend()) {
                            if (*it1 != *it2)
                                return false;
                            ++it1;
                            ++it2;
                        }
                        return it2 == cend_;
                    }

                    inline iterator operator+=(difference_type diff) {
                        return iterator(cbegin_ += diff, cend_);
                    }

                    inline iterator operator-=(difference_type diff) {
                        return iterator(cbegin_ -= diff, cend_);
                    }

                    template<typename StorageSpan>
                    friend class Span;

                private:
                    const_iterator cbegin_;
                    const_iterator cend_;
                };

                template<typename Storage>
                class Span {
                public:
                    using iterator = typename Storage::iterator;
                    using const_iterator = typename Storage::const_iterator;
                    typedef ptrdiff_t difference_type;

                    Span() = default;

                    Span(iterator begin, iterator end)
                        : begin_{begin}, end_{end} {
                    }

                    Span(iterator begin, Size size)
                        : begin_{begin}, end_{begin + size} {
                    }

                    template<typename DType>
                    Span(const std::vector<DType> &vector)
                        : begin_{vector.begin()}, end_{vector.end()} {
                    }

                    std::size_t size() const {
                        return end_ - begin_;
                    }

                    iterator begin() const {
                        return begin_;
                    }

                    iterator end() const {
                        return end_;
                    }

                    iterator cbegin() const {
                        return begin_;
                    }

                    iterator cend() const {
                        return end_;
                    }

                    typename const_iterator::reference operator[](std::size_t i) const {
                        return *(begin_ + i);
                    }

                    typename iterator::reference operator[](std::size_t i) {
                        return *(begin_ + i);
                    }

                    inline iterator operator+=(difference_type diff) {
                        return iterator(begin_ += diff, end_);
                    }

                    inline iterator operator-=(difference_type diff) {
                        return iterator(begin_ -= diff, end_);
                    }

                    bool operator==(const Storage &storage) const {
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

                private:
                    iterator begin_;
                    iterator end_;
                };

                template<typename DType>
                using NDArrayDynamicInternalStorageSpan = Span<NDArrayDynamicInternalStorageVector<DType>>;

                template<typename DType>
                using NDArrayDynamicInternalStorageConstSpan = ConstSpan<NDArrayDynamicInternalStorageVector<DType>>;

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
                        : m_Shape{}, m_Impl{} {
                    }

                    explicit NDArrayDynamicInternal(Shape shape) noexcept
                        : m_Shape{std::move(shape)}, m_Impl{} {
                        std::size_t size = np::ndarray::internal::calcSizeByShape(m_Shape);
                        m_Impl.resize(size);
                    }

                    explicit NDArrayDynamicInternal(Shape shape, const DType &value)
                        : m_Shape{std::move(shape)} {
                        std::size_t size = np::ndarray::internal::calcSizeByShape(m_Shape);
                        m_Impl = std::move(std::vector(size, value));
                    }

                    explicit NDArrayDynamicInternal(const DType &value)
                        : m_Shape{1}, m_Impl{1, value} {
                    }

                    template<typename InternalStorage>
                    NDArrayDynamicInternal(const NDArrayDynamicInternal<DType, InternalStorage> &another) noexcept
                        : m_Shape{another.m_Shape}, m_Impl{another.m_Impl} {
                    }

                    template<typename InternalStorage>
                    NDArrayDynamicInternal(NDArrayDynamicInternal<DType, InternalStorage> &&another) noexcept
                        : m_Shape{std::move(another.m_Shape)}, m_Impl{std::move(another.m_Impl)} {
                    }

                    // create 1D array
                    NDArrayDynamicInternal(const Storage &impl, Size size) noexcept
                        : m_Shape{size}, m_Impl{impl} {
                    }

                    NDArrayDynamicInternal(Storage &&impl, Size size) noexcept
                        : m_Shape{size}, m_Impl{std::move(impl)} {
                    }

                    NDArrayDynamicInternal(typename Storage::const_iterator it, Size size) noexcept
                        : m_Shape{size}, m_Impl(it, size) {
                    }

                    NDArrayDynamicInternal(const Storage &impl, Shape shape) noexcept
                        : m_Shape{std::move(shape)}, m_Impl{impl} {
                    }

                    NDArrayDynamicInternal(Storage &&impl, Shape shape) noexcept
                        : m_Shape{std::move(shape)}, m_Impl{std::move(impl)} {
                    }

                    NDArrayDynamicInternal(typename Storage::const_iterator it, Shape shape) noexcept
                        : m_Shape{std::move(shape)}, m_Impl(it, np::ndarray::internal::calcSizeByShape(m_Shape)) {
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicInternal(const CArray1DType<Size1T> &array) noexcept
                        : m_Shape{Size1T}, m_Impl{std::begin(array), std::end(array)} {
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicInternal(const CArray1DType<Size1T> &array, bool isColumnVector) noexcept
                        : NDArrayDynamicInternal{array} {
                        if (isColumnVector) {
                            m_Shape.insert(m_Shape.begin(), 1);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayDynamicInternal(const CArray2DType<Size1T, Size2T> &array) noexcept
                        : m_Shape{Size2T, Size1T} {
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayDynamicInternal(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                        : m_Shape{Size3T, Size2T, Size1T} {
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayDynamicInternal(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                        : m_Shape{Size4T, Size3T, Size2T, Size1T} {
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), std::back_inserter(m_Impl));
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T>
                    explicit NDArrayDynamicInternal(const StdArray1DType<Size1T> &array) noexcept
                        : m_Shape{Size1T}, m_Impl{std::begin(array), std::end(array)} {
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicInternal(const StdArray1DType<Size1T> &array, bool isColumnVector) noexcept
                        : NDArrayDynamicInternal{array} {
                        if (isColumnVector) {
                            m_Shape.insert(m_Shape.begin(), 1);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    explicit NDArrayDynamicInternal(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                        m_Shape = {Size2T, Size1T};

                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    explicit NDArrayDynamicInternal(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                        m_Shape = {Size3T, Size2T, Size1T};

                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    explicit NDArrayDynamicInternal(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                        m_Shape = {Size4T, Size3T, Size2T, Size1T};

                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), std::back_inserter(m_Impl));
                                }
                            }
                        }
                    }

                    explicit NDArrayDynamicInternal(const StdVector1DType &vector) noexcept
                        : m_Shape{static_cast<Size>(vector.size())}, m_Impl{vector} {
                    }

                    explicit NDArrayDynamicInternal(const StdVector1DType &vector, bool isColumnVector) noexcept
                        : NDArrayDynamicInternal{vector} {
                        if (isColumnVector) {
                            m_Shape.insert(m_Shape.begin(), 1);
                        }
                    }

                    explicit NDArrayDynamicInternal(const StdVector2DType &vector) noexcept {
                        m_Shape = {static_cast<Size>(vector.size())};
                        if (!vector.empty()) {
                            m_Shape.push_back(vector[0].size());
                        }

                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            std::copy(std::begin(vector[i]), std::end(vector[i]), std::back_inserter(m_Impl));
                        }
                    }

                    explicit NDArrayDynamicInternal(const StdVector3DType &vector) noexcept {
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

                    explicit NDArrayDynamicInternal(const StdVector4DType &vector) noexcept {
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

                    NDArrayDynamicInternal(std::initializer_list<DType> init_list) noexcept
                        : m_Shape{static_cast<Size>(init_list.size())}, m_Impl{init_list} {
                    }

                    NDArrayDynamicInternal &operator=(const DType &value) noexcept {
                        m_Shape = {1};
                        m_Impl.reserve(1);
                        m_Impl.fill(value);
                        return *this;
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicInternal &operator=(CArray1DType<Size1T> array) noexcept {
                        m_Shape = {Size1T};
                        m_Impl = std::vector{array, Size1T * sizeof(DType)};
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayDynamicInternal &operator=(CArray2DType<Size1T, Size2T> array) noexcept {
                        m_Shape = {Size2T, Size1T};

                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayDynamicInternal &operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept {
                        m_Shape = {Size3T, Size2T, Size1T};
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayDynamicInternal &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept {
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
                    NDArrayDynamicInternal &operator=(const StdArray1DType<Size1T> &array) noexcept {
                        m_Shape = {Size1T};
                        m_Impl = std::vector{array.data(), Size1T};
                        return *this;
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayDynamicInternal &operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                        m_Shape = {Size2T, Size1T};

                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), std::back_inserter(m_Impl));
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayDynamicInternal &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                        m_Shape = {Size3T, Size2T, Size1T};
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), std::back_inserter(m_Impl));
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayDynamicInternal &
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

                    NDArrayDynamicInternal &operator=(const StdVector1DType &vector) noexcept {
                        m_Shape = {vector.size()};
                        m_Impl = vector;
                        return *this;
                    }

                    NDArrayDynamicInternal &operator=(const StdVector2DType &vector) noexcept {
                        m_Shape = {vector.size()};
                        if (!vector.empty()) {
                            m_Shape.push_back(vector[0].size());
                        }

                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            std::copy(std::begin(vector[i]), std::end(vector[i]), std::back_inserter(m_Impl));
                        }
                        return *this;
                    }

                    NDArrayDynamicInternal &operator=(const StdVector3DType &vector) noexcept {
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

                    NDArrayDynamicInternal &operator=(const StdVector4DType &vector) noexcept {
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

                    NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageConstSpan<DType>> operator[](std::size_t i) const {
                        if (m_Shape.empty()) {
                            throw std::runtime_error("Index " + std::to_string(i) + " of an empty array requested");
                        }
                        auto itBeginImpl = m_Impl.cbegin();
                        if (m_Shape.size() == 1) {
                            if (static_cast<Size>(i) >= m_Shape[0]) {
                                throw std::runtime_error("Index " + std::to_string(i) + " out of bounds");
                            }
                            std::advance(itBeginImpl, i);
                            return NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageConstSpan<DType>>(itBeginImpl, Shape{1});
                        }

                        auto itBeginShape = std::cbegin(m_Shape);
                        std::advance(itBeginShape, 1);
                        Shape shape{std::vector(itBeginShape, std::cend(m_Shape))};
                        auto layerSize = m_Shape[0] == 0 ? 0 : size() / m_Shape[0];
                        std::advance(itBeginImpl, i * layerSize);
                        return NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageConstSpan<DType>>(itBeginImpl, shape);
                    }

                    const DType &get(std::size_t i) const {
                        return m_Impl[i];
                    }

                    DType &get(std::size_t i) {
                        return m_Impl[i];
                    }

                    void set(std::size_t i, const DType &value) {
                        m_Impl[i] = value;
                    }

                    friend std::ostream &operator<< <DType, Storage>(std::ostream &stream, const NDArrayDynamicInternal<DType, Storage> &array);

                    Shape getShape() const {
                        return m_Shape;
                    }

                    void setShape(const Shape &shape) {
                        m_Shape = shape;
                    }

                    std::size_t size() const {
                        return m_Impl.size();
                    }

                    void dumpToStreamAsBinary(std::ostream &stream) {
                        for (std::size_t index = 0; index < size(); ++index) {
                            ndarray::internal::dumpObject(stream, m_Impl[index]);
                        }
                    }

                    bool operator==(const DType &value) const {
                        return size() == 1 && m_Impl[0] == value;
                    }

                    bool operator!=(const DType &value) const {
                        return !operator==(value);
                    }

                    bool operator==(const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>> &other) const {
                        return m_Shape == other.m_Shape && m_Impl == other.m_Impl;
                    }

                    bool operator==(const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageVector<DType>> &other) const {
                        return m_Shape == other.m_Shape && m_Impl == other.m_Impl;
                    }

                    bool operator!=(const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageSpan<DType>> &other) const {
                        return !operator==(other);
                    }

                    bool operator!=(const NDArrayDynamicInternal<DType, NDArrayDynamicInternalStorageVector<DType>> &other) const {
                        return !operator==(other);
                    }

                    void sort() {
                        std::sort(m_Impl.begin(), m_Impl.end());
                    }

                    class iterator {
                    public:
                        typedef ptrdiff_t difference_type;
                        typedef DType value_type;
                        typedef DType *pointer;
                        typedef DType &reference;
                        typedef std::random_access_iterator_tag iterator_category;

                        iterator(NDArrayDynamicInternal *container_, std::size_t offset_)
                            : container{container_}, offset{offset_} {
                        }

                        iterator(const iterator &it)
                            : container{it.container}, offset{it.offset} {
                        }

                        iterator &operator=(const iterator &it) {
                            if (this != &it) {
                                container = it.container;
                                offset = it.offset;
                            }
                            return *this;
                        }

                        bool operator==(const iterator &it) const {
                            return container == it.container && offset == it.offset;
                        }

                        bool operator!=(const iterator &it) const {
                            return !(*this == it);
                        }

                        bool operator>(const iterator &it) const {
                            assert(container == it.container);
                            return offset > it.offset;
                        }

                        bool operator>=(const iterator &it) const {
                            assert(container == it.container);
                            return offset >= it.offset;
                        }

                        bool operator<(const iterator &it) const {
                            assert(container == it.container);
                            return offset < it.offset;
                        }

                        bool operator<=(const iterator &it) const {
                            assert(container == it.container);
                            return offset <= it.offset;
                        }

                        iterator operator++() {
                            return iterator(container, ++offset);
                        }

                        iterator operator++(int) {
                            return iterator(container, offset++);
                        }

                        iterator operator--() {
                            return iterator(container, --offset);
                        }

                        iterator operator--(int) {
                            return iterator(container, offset--);
                        }

                        iterator operator-(difference_type diff) {
                            return iterator(container, offset - diff);
                        }

                        iterator operator+(difference_type diff) {
                            return iterator(container, offset + diff);
                        }

                        iterator operator+=(difference_type diff) {
                            return iterator(container, offset += diff);
                        }

                        iterator operator-=(difference_type diff) {
                            return iterator(container, offset -= diff);
                        }

                        difference_type operator-(const iterator &it) const {
                            assert(container == it.container);
                            return offset - it.offset;
                        }

                        value_type &operator*() {
                            return container->get(offset);
                        }

                    private:
                        NDArrayDynamicInternal *container;
                        std::size_t offset;
                    };

                    iterator begin() {
                        return iterator{this, 0};
                    }

                    iterator end() {
                        return iterator{this, m_Impl.size()};
                    }

                    class const_iterator {
                    public:
                        typedef ptrdiff_t difference_type;
                        typedef DType value_type;
                        typedef DType *pointer;
                        typedef DType &reference;
                        typedef std::random_access_iterator_tag iterator_category;

                        const_iterator(const NDArrayDynamicInternal *container_, std::size_t offset_)
                            : container{container_}, offset{offset_} {
                        }

                        const_iterator(const const_iterator &it)
                            : container{it.container}, offset{it.offset} {
                        }

                        const_iterator &operator=(const const_iterator &it) {
                            if (this != &it) {
                                container = it.container;
                                offset = it.offset;
                            }
                            return *this;
                        }

                        bool operator==(const const_iterator &it) const {
                            return container == it.container && offset == it.offset;
                        }

                        bool operator!=(const const_iterator &it) const {
                            return !(*this == it);
                        }

                        bool operator>(const const_iterator &it) const {
                            assert(container == it.container);
                            return offset > it.offset;
                        }

                        bool operator>=(const const_iterator &it) const {
                            assert(container == it.container);
                            return offset >= it.offset;
                        }

                        bool operator<(const const_iterator &it) const {
                            assert(container == it.container);
                            return offset < it.offset;
                        }

                        bool operator<=(const const_iterator &it) const {
                            assert(container == it.container);
                            return offset <= it.offset;
                        }

                        const_iterator operator++() {
                            return const_iterator(container, ++offset);
                        }

                        const_iterator operator++(int) {
                            return const_iterator(container, offset++);
                        }

                        const_iterator operator--() {
                            return const_iterator(container, --offset);
                        }

                        const_iterator operator--(int) {
                            return const_iterator(container, offset--);
                        }

                        const_iterator operator-(difference_type diff) {
                            return const_iterator(container, offset - diff);
                        }

                        const_iterator operator+(difference_type diff) {
                            return const_iterator(container, offset + diff);
                        }

                        const_iterator operator+=(difference_type diff) {
                            return const_iterator(container, offset += diff);
                        }

                        const_iterator operator-=(difference_type diff) {
                            return const_iterator(container, offset -= diff);
                        }

                        difference_type operator-(const const_iterator &it) const {
                            assert(container == it.container);
                            return offset - it.offset;
                        }

                        auto operator*() const {
                            return container->get(offset);
                        }

                    private:
                        const NDArrayDynamicInternal *container;
                        std::size_t offset;
                    };

                    const_iterator cbegin() const {
                        return const_iterator{this, 0};
                    }

                    const_iterator cend() const {
                        return const_iterator{this, m_Impl.size()};
                    }

                    template<typename DTypeOther, typename StorageOther>
                    friend class NDArrayDynamicInternal;

                    template<typename DTypeOther, typename StorageOther>
                    friend class NDArrayDynamic;

                private:
                    Shape m_Shape;
                    Storage m_Impl;
                };

                template<typename DType, typename Storage1, typename Storage2>
                static inline bool array_equal(const internal::NDArrayDynamicInternal<DType, Storage1> &array1,
                                               const internal::NDArrayDynamicInternal<DType, Storage2> &array2) {
                    return array1 == array2;
                }

                template<typename Storage1, typename Storage2>
                static inline bool array_equal(const internal::NDArrayDynamicInternal<double, Storage1> &array1,
                                               const internal::NDArrayDynamicInternal<double, Storage2> &array2) {
                    if (array1.size() != array2.size())
                        return false;
                    for (std::size_t i = 0; i < array1.size(); ++i) {
                        if (!np::internal::almost_equal(array1.get(i), array2.get(i), ULP_TOLERANCE))
                            return false;
                    }
                    return true;
                }
            }// namespace internal
        }    // namespace array_dynamic
    }        // namespace ndarray
}// namespace np
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include <np/internal/Tools.hpp>
#include <np/ndarray/dynamic/internal/Tools.hpp>
#include <np/ndarray/dynamic/internal/Using.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            namespace internal {

                template<typename DType>
                class NDArrayDynamicStorage {
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
                    NDArrayDynamicStorage() noexcept
                        : m_size{}, m_ptr{} {
                    }

                    NDArrayDynamicStorage(const NDArrayDynamicStorage &storage)
                        : m_size{storage.m_size}, m_ptr{} {
                        if (m_size != 0) {
                            m_ptr.reset(new DType[m_size]);

                            for (std::size_t i = 0; i < m_size; ++i) {
                                m_ptr[i] = storage.m_ptr[i];
                            }
                        }
                    }

                    NDArrayDynamicStorage(NDArrayDynamicStorage &&storage) noexcept
                        : m_size{storage.m_size}, m_ptr{storage.m_ptr} {
                        storage.m_ptr.reset();
                        storage.m_size = 0;
                    }

                    explicit NDArrayDynamicStorage(Size size, const DType &value = DType())
                        : m_size{size}, m_ptr{} {
                        if (m_size != 0) {
                            m_ptr.reset(new DType[m_size]);

                            for (std::size_t i = 0; i < m_size; ++i) {
                                m_ptr[i] = value;
                            }
                        }
                    }

                    // create 1D array
                    NDArrayDynamicStorage(const std::vector<DType> &vector, Size size) noexcept
                        : m_size{size}, m_ptr{new DType[vector.size()]} {
                        for (std::size_t i = 0; i < size; ++i) {
                            m_ptr[i] = vector[i];
                        }
                    }

                    template<std::size_t Size1T>
                    explicit NDArrayDynamicStorage(const CArray1DType<Size1T> &array) noexcept
                        : m_size{Size1T}, m_ptr{new DType[Size1T]} {
                        std::copy(std::begin(array), std::end(array), m_ptr.get());
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    explicit NDArrayDynamicStorage(const CArray2DType<Size1T, Size2T> &array) noexcept
                        : m_size{Size2T * Size1T}, m_ptr{new DType[Size2T * Size1T]} {
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            for (std::size_t j = 0; j < Size1T; ++j) {
                                m_ptr[i * Size1T + j] = array[i][j];
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    explicit NDArrayDynamicStorage(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                        : m_size{Size3T * Size2T * Size1T}, m_ptr{new DType[Size3T * Size2T * Size1T]} {
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                for (std::size_t k = 0; k < Size1T; ++k) {
                                    m_ptr[i * Size1T * Size2T + j * Size1T + k] = array[i][j][k];
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    explicit NDArrayDynamicStorage(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                        : m_size{Size4T * Size3T * Size2T * Size1T}, m_ptr{new DType[Size4T * Size3T * Size2T * Size1T]} {
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    for (std::size_t l = 0; l < Size1T; ++l) {
                                        m_ptr[i * Size1T * Size2T * Size3T + j * Size1T * Size2T + k * Size1T + l] = array[i][j][k][l];
                                    }
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T>
                    explicit NDArrayDynamicStorage(const StdArray1DType<Size1T> &array) noexcept
                        : m_size{Size1T}, m_ptr{new DType[Size1T]} {
                        std::copy(std::begin(array), std::end(array), m_ptr.get());
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    explicit NDArrayDynamicStorage(const StdArray2DType<Size1T, Size2T> &array) noexcept
                        : m_size{Size2T * Size1T}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size1T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    explicit NDArrayDynamicStorage(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept
                        : m_size{Size3T * Size2T * Size1T}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    explicit NDArrayDynamicStorage(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept
                        : m_size{Size4T * Size3T * Size2T * Size1T}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    explicit NDArrayDynamicStorage(const StdVector1DType &vector) noexcept
                        : m_size{vector.size()}, m_ptr{new DType[m_size]} {
                        std::copy(std::begin(vector), std::end(vector), m_ptr.get());
                    }

                    explicit NDArrayDynamicStorage(const StdVector2DType &vector) noexcept
                        : m_size{vector.size() * vector[0].size()}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            std::copy(std::begin(vector[i]), std::end(vector[i]), it);
                            std::advance(it, vector[i].size());
                        }
                    }

                    explicit NDArrayDynamicStorage(const StdVector3DType &vector) noexcept
                        : m_size{vector.size() * vector[0].size() * vector[0][0].size()}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                std::copy(std::begin(vector[i][j]), std::end(vector[i][j]), it);
                                std::advance(it, vector[i][j].size());
                            }
                        }
                    }

                    explicit NDArrayDynamicStorage(const StdVector4DType &vector) noexcept
                        : m_size{vector.size() * vector[0].size() * vector[0][0].size() * vector[0][0][0].size()}, m_ptr{new DType[m_size]} {
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                for (std::size_t k = 0; k < vector[0][0].size; ++k) {
                                    std::copy(std::begin(vector[i][j][k]), std::end(vector[i][j][k]), it);
                                    std::advance(it, vector[i][j][k].size());
                                }
                            }
                        }
                    }

                    NDArrayDynamicStorage(std::initializer_list<DType> init_list) noexcept
                        : m_size{init_list.size()}, m_ptr{new DType[init_list.size()]} {
                        std::copy(std::begin(init_list), std::end(init_list), m_ptr.get());
                    }

                    ~NDArrayDynamicStorage() = default;

                    NDArrayDynamicStorage &operator=(const NDArrayDynamicStorage &storage) {
                        if (this != &storage) {
                            m_size = storage.m_size;
                            m_ptr.reset(new DType[m_size]);
                            std::copy(storage.m_ptr.get(), storage.m_ptr.get() + m_size, m_ptr.get());
                        }
                        return *this;
                    }

                    NDArrayDynamicStorage &operator=(NDArrayDynamicStorage &&storage) noexcept {
                        if (this != &storage) {
                            m_size = std::move(storage.m_size);
                            m_ptr = std::move(storage.m_ptr);
                            storage.m_ptr.reset();
                            storage.m_size = 0;
                        }
                        return *this;
                    }

                    NDArrayDynamicStorage &operator=(const DType &value) {
                        m_size = 1;
                        m_ptr = new DType[1];
                        m_ptr[0] = value;
                        return *this;
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicStorage &operator=(CArray1DType<Size1T> array) {
                        m_size = Size1T;
                        m_ptr = new DType[Size1T];
                        std::copy(std::begin(array), std::end(array), std::begin(*m_ptr));
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayDynamicStorage &operator=(CArray2DType<Size1T, Size2T> array) {
                        m_size = Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size1T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayDynamicStorage &operator=(CArray3DType<Size1T, Size2T, Size3T> array) {
                        m_size = Size3T * Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i]), std::end(array[i]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayDynamicStorage &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) {
                        m_size = Size4T * Size3T * Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i]), std::end(array[i]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T>
                    NDArrayDynamicStorage &operator=(const StdArray1DType<Size1T> &array) {
                        m_size = Size1T;
                        m_ptr = new DType[Size1T];
                        auto it = m_ptr.get();
                        std::copy(std::begin(array), std::end(array), it);
                        return *this;
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayDynamicStorage &operator=(const StdArray2DType<Size1T, Size2T> &array) {
                        m_size = Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size1T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayDynamicStorage &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) {
                        m_size = Size3T * Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayDynamicStorage &
                    operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) {
                        m_size = Size4T * Size3T * Size2T * Size1T;
                        m_ptr = new DType[m_size];
                        auto it = m_ptr.get();
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    NDArrayDynamicStorage &operator=(const StdVector1DType &vector) {
                        m_size = vector.size();
                        m_ptr = new DType[m_size];
                        return *this;
                    }

                    NDArrayDynamicStorage &operator=(const StdVector2DType &vector) {
                        m_size = vector.size() * vector[0].size();
                        m_ptr = new DType[m_size];
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                m_ptr[i * vector[0].size() + j] = vector[i][j];
                            }
                        }
                        return *this;
                    }

                    NDArrayDynamicStorage &operator=(const StdVector3DType &vector) {
                        m_size = vector.size() * vector[0].size() * vector[0][0].size();
                        m_ptr = new DType[m_size];
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                for (std::size_t k = 0; k < vector[0][0].size(); ++k) {
                                    m_ptr[k * vector[0][0].size() + i * vector[0].size() + j] = vector[i][j][k];
                                }
                            }
                        }
                    }

                    NDArrayDynamicStorage &operator=(const StdVector4DType &vector) {
                        m_size = vector.size() * vector[0].size() * vector[0][0].size() * vector[0][0][0].size();
                        m_ptr = new DType[m_size];
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                for (std::size_t k = 0; k < vector[0][0].size; ++k) {
                                    for (std::size_t l = 0; l < vector[0][0].size; ++l) {
                                        m_ptr[l * vector[0][0][0].size() + k * vector[0][0].size() + i * vector[0].size() + j] = vector[i][j][k][l];
                                    }
                                }
                            }
                        }
                    }

                    [[nodiscard]] const DType &get(std::size_t i) const {
                        return m_ptr[i];
                    }

                    DType &get(std::size_t i) {
                        return m_ptr[i];
                    }

                    void set(std::size_t i, DType value) {
                        m_ptr[i] = value;
                    }

                    [[nodiscard]] std::size_t size() const {
                        return m_size;
                    }

                    bool operator==(const DType &value) const {
                        return size() == 1 && m_ptr[0] == value;
                    }

                    bool operator!=(const DType &value) const {
                        return !operator==(value);
                    }

                    void sort() {
                        std::sort(begin(), end());
                    }

                    class iterator {
                    public:
                        using difference_type = std::ptrdiff_t;
                        using value_type = DType;
                        using pointer = DType *;
                        using reference = DType &;
                        using iterator_category = std::random_access_iterator_tag;

                        iterator(NDArrayDynamicStorage *container_, std::size_t offset_)
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

                        reference operator*() const {
                            return container->get(offset);
                        }

                    private:
                        NDArrayDynamicStorage *container;
                        std::size_t offset;
                    };

                    iterator begin() {
                        return iterator{this, 0};
                    }

                    iterator end() {
                        return iterator{this, static_cast<std::size_t>(m_size)};
                    }

                    class const_iterator {
                    public:
                        using difference_type = std::ptrdiff_t;
                        using value_type = DType;
                        using pointer = DType *;
                        using reference = DType &;
                        using iterator_category = std::random_access_iterator_tag;

                        const_iterator(const NDArrayDynamicStorage *container_, std::size_t offset_)
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

                        const value_type &operator*() const {
                            return container->get(offset);
                        }

                    private:
                        const NDArrayDynamicStorage *container;
                        std::size_t offset;
                    };

                    [[nodiscard]] const_iterator cbegin() const {
                        return const_iterator{this, 0};
                    }

                    [[nodiscard]] const_iterator cend() const {
                        return const_iterator{this, m_size};
                    }

                    [[nodiscard]] Size index(Size i) const {
                        return i;
                    }

                    static constexpr int kDepth = 0;

                private:
                    Size m_size;
                    std::shared_ptr<DType[]> m_ptr;// not using std::vector, due to vector<bool> behavior
                };

                template<typename DType>
                static inline bool array_equal(const internal::NDArrayDynamicStorage<DType> &array1,
                                               const internal::NDArrayDynamicStorage<DType> &array2) {
                    return array1 == array2;
                }

                template<typename Storage1, typename Storage2>
                static inline bool array_equal(const internal::NDArrayDynamicStorage<double> &array1,
                                               const internal::NDArrayDynamicStorage<double> &array2) {
                    if (array1.size() != array2.size())
                        return false;
                    for (std::size_t i = 0; i < array1.size(); ++i) {
                        if (!np::internal::almost_equal(array1.get(i), array2.get(i), np::internal::ULP_TOLERANCE))
                            return false;
                    }
                    return true;
                }

            }// namespace internal
        }    // namespace array_dynamic
    }        // namespace ndarray
}// namespace np
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

#include <array>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include <np/Constants.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            namespace internal {

                template<typename DType, Size SizeT>
                class NDArrayStaticStorage;

                template<typename DType, Size SizeT>
                std::ostream &operator<<(std::ostream &stream, const NDArrayStaticStorage<DType, SizeT> &array);

                template<Size SizeT>
                std::ostream &operator<<(std::ostream &stream, const NDArrayStaticStorage<std::wstring, SizeT> &array);

                template<typename DType, Size SizeT>
                std::wostream &operator<<(std::wostream &stream, const NDArrayStaticStorage<DType, SizeT> &array);

                template<Size SizeT>
                std::wostream &operator<<(std::wostream &stream, const NDArrayStaticStorage<std::wstring, SizeT> &array);

                template<typename DType, Size SizeT>
                class NDArrayStaticStorage {
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
                    NDArrayStaticStorage() noexcept
                        : m_storage{} {
                    }

                    explicit NDArrayStaticStorage(const DType &value) {
                        for (std::size_t i = 0; i < SizeT; ++i) {
                            m_storage[i] = value;
                        }
                    }

                    explicit NDArrayStaticStorage(Size size) {
                        static_assert(SizeT == size);
                    }

                    template<std::size_t Size1T>
                    explicit NDArrayStaticStorage(const CArray1DType<Size1T> &array) noexcept {
                        static_assert(SizeT == Size1T);
                        for (std::size_t i = 0; i < std::size(array); ++i) {
                            m_storage[i] = array[i];
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    explicit NDArrayStaticStorage(const CArray2DType<Size1T, Size2T> &array) noexcept {
                        static_assert(SizeT == Size2T * Size1T);
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            for (std::size_t j = 0; j < Size1T; ++j) {
                                m_storage[i * Size1T + j] = array[i][j];
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    explicit NDArrayStaticStorage(const CArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                        static_assert(SizeT == Size3T * Size2T * Size1T);
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                for (std::size_t k = 0; k < Size1T; ++k) {
                                    m_storage[i * Size1T * Size2T + j * Size1T + k] = array[i][j][k];
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    explicit NDArrayStaticStorage(const CArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                        static_assert(SizeT == Size4T * Size3T * Size2T * Size1T);
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    for (std::size_t l = 0; l < Size1T; ++l) {
                                        m_storage[i * Size1T * Size2T * Size3T + j * Size1T * Size2T + k * Size1T + l] = array[i][j][k][l];
                                    }
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T>
                    explicit NDArrayStaticStorage(const StdArray1DType<Size1T> &array) noexcept {
                        static_assert(SizeT == Size1T);
                        for (std::size_t i = 0; i < Size1T; ++i) {
                            std::copy(std::begin(array), std::end(array), std::begin(m_storage));
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    explicit NDArrayStaticStorage(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                        static_assert(SizeT == Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size2T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size1T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    explicit NDArrayStaticStorage(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                        static_assert(SizeT == Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    explicit NDArrayStaticStorage(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                        static_assert(SizeT == Size4T * Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    explicit NDArrayStaticStorage(const StdVector1DType &vector) noexcept {
                        assert(vector.size() == SizeT);
                        std::copy(std::begin(vector), std::end(vector), std::begin(m_storage));
                    }

                    explicit NDArrayStaticStorage(const StdVector2DType &vector) noexcept {
                        assert(vector.size() * vector[0].size() == SizeT);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            std::copy(std::begin(vector[i]), std::end(vector[i]), it);
                            std::advance(it, vector[i].size());
                        }
                    }

                    explicit NDArrayStaticStorage(const StdVector3DType &vector) noexcept {
                        assert(vector.size() * vector[0].size() * vector[0][0].size() == SizeT);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                std::copy(std::begin(vector[i][j]), std::end(vector[i][j]), it);
                                std::advance(it, vector[i][j].size());
                            }
                        }
                    }

                    explicit NDArrayStaticStorage(const StdVector4DType &vector) noexcept {
                        assert(vector.size() * vector[0].size() * vector[0][0].size() * vector[0][0][0].size() == SizeT);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < vector.size(); ++i) {
                            for (std::size_t j = 0; j < vector[0].size(); ++j) {
                                for (std::size_t k = 0; k < vector[0][0].size(); ++k) {
                                    std::copy(std::begin(vector[i][j][k]), std::end(vector[i][j][k]), it);
                                    std::advance(it, vector[i][j][k].size());
                                }
                            }
                        }
                    }

                    NDArrayStaticStorage(std::initializer_list<DType> init_list) noexcept {
                        for (auto it = std::begin(m_storage); it != std::end(m_storage); std::advance(it, std::size(init_list))) {
                            std::copy(std::begin(init_list), std::end(init_list), it);
                        }
                    }

                    NDArrayStaticStorage &operator=(const DType &value) noexcept {
                        static_assert(SizeT == 1);
                        m_storage[0] = value;
                        return *this;
                    }

                    template<std::size_t Size1T>
                    NDArrayStaticStorage &operator=(CArray1DType<Size1T> array) noexcept {
                        static_assert(SizeT == Size1T);
                        std::copy(std::begin(array), std::end(array), std::begin(m_storage));
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayStaticStorage &operator=(CArray2DType<Size1T, Size2T> array) noexcept {
                        static_assert(SizeT == Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size1T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size2T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayStaticStorage &operator=(CArray3DType<Size1T, Size2T, Size3T> array) noexcept {
                        static_assert(SizeT == Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayStaticStorage &operator=(CArray4DType<Size1T, Size2T, Size3T, Size4T> array) noexcept {
                        static_assert(SizeT == Size4T * Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    template<std::size_t Size1T>
                    NDArrayStaticStorage &operator=(const StdArray1DType<Size1T> &array) noexcept {
                        static_assert(SizeT == Size1T);
                        std::copy(std::begin(array), std::end(array), std::begin(m_storage));
                        return *this;
                    }

                    template<std::size_t Size1T, std::size_t Size2T>
                    NDArrayStaticStorage &operator=(const StdArray2DType<Size1T, Size2T> &array) noexcept {
                        static_assert(SizeT == Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size1T; ++i) {
                            std::copy(std::begin(array[i]), std::end(array[i]), it);
                            std::advance(it, Size2T);
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T>
                    NDArrayStaticStorage &operator=(const StdArray3DType<Size1T, Size2T, Size3T> &array) noexcept {
                        static_assert(SizeT == Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size3T; ++i) {
                            for (std::size_t j = 0; j < Size2T; ++j) {
                                std::copy(std::begin(array[i][j]), std::end(array[i][j]), it);
                                std::advance(it, Size1T);
                            }
                        }
                    }

                    template<std::size_t Size1T, std::size_t Size2T, std::size_t Size3T, std::size_t Size4T>
                    NDArrayStaticStorage &
                    operator=(const StdArray4DType<Size1T, Size2T, Size3T, Size4T> &array) noexcept {
                        static_assert(SizeT == Size4T * Size3T * Size2T * Size1T);
                        auto it = std::begin(m_storage);
                        for (std::size_t i = 0; i < Size4T; ++i) {
                            for (std::size_t j = 0; j < Size3T; ++j) {
                                for (std::size_t k = 0; k < Size2T; ++k) {
                                    std::copy(std::begin(array[i][j][k]), std::end(array[i][j][k]), it);
                                    std::advance(it, Size1T);
                                }
                            }
                        }
                    }

                    const DType &get(std::size_t i) const {
                        return m_storage[i];
                    }

                    DType &get(std::size_t i) {
                        return m_storage[i];
                    }

                    void set(std::size_t i, const DType &value) {
                        m_storage[i] = value;
                    }

                    [[nodiscard]] std::size_t size() const {
                        return SizeT;
                    }

                    void sort() {
                        std::sort(m_storage.begin(), m_storage.end());
                    }

                    class iterator {
                    public:
                        using difference_type = std::ptrdiff_t;
                        using value_type = DType;
                        using pointer = DType *;
                        using reference = DType &;
                        using iterator_category = std::random_access_iterator_tag;

                        iterator(NDArrayStaticStorage *container_, std::size_t offset_)
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
                        NDArrayStaticStorage *container;
                        std::size_t offset;
                    };

                    iterator begin() {
                        return iterator{this, 0};
                    }

                    iterator end() {
                        return iterator{this, SizeT};
                    }

                    class const_iterator {
                    public:
                        using difference_type = std::ptrdiff_t;
                        using value_type = DType;
                        using pointer = DType *;
                        using reference = DType &;
                        using iterator_category = std::random_access_iterator_tag;

                        const_iterator(const NDArrayStaticStorage *container_, std::size_t offset_)
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
                        const NDArrayStaticStorage *container;
                        std::size_t offset;
                    };

                    const_iterator cbegin() const {
                        return const_iterator{this, 0};
                    }

                    const_iterator cend() const {
                        return const_iterator{this, SizeT};
                    }

                    [[nodiscard]] Size index(Size i) const {
                        return i;
                    }

                    [[nodiscard]] Shape shape() const {
                        throw std::runtime_error("shape() is not implemented");
                    }

                    void setShape(const Shape &) {
                        throw std::runtime_error("setShape() is not implemented");
                    }

                    void push_back(const DType &) {
                        throw std::runtime_error("push_back is not implemented");
                    }

                    static constexpr int kDepth = 0;

                private:
                    std::array<DType, SizeT> m_storage;
                };
            }// namespace internal
        }    // namespace array_static
    }        // namespace ndarray
}// namespace np
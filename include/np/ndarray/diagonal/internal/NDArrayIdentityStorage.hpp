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

#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <utility>

#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace array_diagonal {
            namespace internal {

                template<typename DType>
                class NDArrayIdentityStorage {
                public:
                    NDArrayIdentityStorage() noexcept
                        : m_size{0} {
                    }

                    explicit NDArrayIdentityStorage(Size size) noexcept
                        : m_size{size} {
                    }

                    NDArrayIdentityStorage(const NDArrayIdentityStorage &storage)
                        : m_size{storage.m_size} {
                    }

                    NDArrayIdentityStorage(NDArrayIdentityStorage &&storage) noexcept
                        : m_size{storage.m_size} {
                    }

                    ~NDArrayIdentityStorage() noexcept = default;

                    NDArrayIdentityStorage &operator=(const NDArrayIdentityStorage &storage) {
                        if (this != &storage) {
                            m_size = storage.m_size;
                        }
                        return *this;
                    }

                    NDArrayIdentityStorage &operator=(NDArrayIdentityStorage &&storage) noexcept {
                        if (this != &storage) {
                            m_size = storage.m_size;
                        }
                        return *this;
                    }

                    [[nodiscard]] const DType &get(Size i) const {
                        Size row = i / m_size;
                        Size column = i % m_size;

                        static DType zero{0};
                        static DType one{1};

                        return row == column ? one : zero;
                    }

                    DType &get(Size) {
                        throw std::runtime_error("non-constant get method is not implemented for NDArrayIdentityStorage");
                        static DType result{};
                        return result;
                    }

                    void set(Size, DType) {
                        throw std::runtime_error("set method is not implemented for NDArrayIdentityStorage");
                    }

                    void sort() {
                        throw std::runtime_error("sort method is not implemented for NDArrayIdentityStorage");
                    }

                    class iterator {
                    public:
                        using difference_type = std::ptrdiff_t;
                        using value_type = DType;
                        using pointer = DType *;
                        using reference = DType &;
                        using iterator_category = std::random_access_iterator_tag;

                        iterator(NDArrayIdentityStorage *container_, std::size_t offset_)
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
                        NDArrayIdentityStorage *container;
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

                        const_iterator(const NDArrayIdentityStorage *container_, std::size_t offset_)
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
                        const NDArrayIdentityStorage *container;
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

                    [[nodiscard]] Shape shape() const {
                        throw std::runtime_error("shape() is not implemented");
                    }

                    void setShape(const Shape &) {
                        throw std::runtime_error("setShape() is not implemented");
                    }

                    void push_back(const DType &) {
                        throw std::runtime_error("push_back() is not implemented");
                    }

                    static constexpr std::size_t kDepth = 0;

                private:
                    Size m_size;
                };

            }// namespace internal
        }    // namespace array_diagonal
    }        // namespace ndarray
}// namespace np
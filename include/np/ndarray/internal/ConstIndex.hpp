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

#include <cstddef>
#include <utility>
#include <vector>

#include <np/Shape.hpp>
#include <np/internal/Tools.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>
#include <np/ndarray/internal/NDArrayShaped.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            class ConstIndex;

            template<typename DType, typename Derived, typename Storage>
            class ConstIndexStorage {
            public:
                ConstIndexStorage() = default;

                ConstIndexStorage(const NDArrayBase<DType, Derived, Storage> *parent, Size indexStart, Size indexEnd)
                    : m_parent{parent}, m_indexStart{indexStart}, m_indexEnd{indexEnd} {
                }

                ConstIndexStorage(const NDArrayBase<DType, Derived, Storage> *parent, const std::vector<Size> &indices)
                    : m_parent{parent}, m_indices{indices}, m_indexStart{static_cast<Size>(-1)}, m_indexEnd{static_cast<Size>(-1)} {
                }

                const DType &get(Size i) const {
                    return m_parent->get(index(i));
                }

                DType &get(Size i) {
                    return m_parent->get(index(i));
                }

                void set(Size i, const DType &value) {
                    m_parent->set(index(i), value);
                }

                class iterator {
                public:
                    using difference_type = std::ptrdiff_t;
                    using value_type = DType;
                    using pointer = DType *;
                    using reference = DType &;
                    using iterator_category = std::random_access_iterator_tag;

                    iterator(ConstIndexStorage *container_, std::size_t offset_)
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

                    iterator next(difference_type diff) {
                        return iterator(container, offset + diff);
                    }

                    reference operator*() const {
                        return container->get(offset);
                    }

                private:
                    ConstIndexStorage *container;
                    std::size_t offset;
                };

                iterator begin() {
                    return iterator{this, m_indexStart == static_cast<Size>(-1) ? m_indices[0] : m_indexStart};
                }

                iterator end() {
                    return iterator{this, m_indexStart == static_cast<Size>(-1) ? m_indices[m_indices.size() - 1] : m_indexEnd};
                }

                class const_iterator {
                public:
                    using difference_type = std::ptrdiff_t;
                    using value_type = DType;
                    using pointer = DType *;
                    using reference = DType &;
                    using iterator_category = std::random_access_iterator_tag;

                    const_iterator(const ConstIndexStorage *container_, std::size_t offset_)
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

                    const_iterator next(difference_type diff) {
                        return const_iterator(container, offset + diff);
                    }

                    const value_type &operator*() const {
                        return container->get(offset);
                    }

                private:
                    const ConstIndexStorage *container;
                    std::size_t offset;
                };

                const_iterator cbegin() const {
                    return const_iterator{this, m_indexStart == static_cast<Size>(-1) ? m_indices[0] : m_indexStart};
                }

                const_iterator cend() const {
                    return const_iterator{this, m_indexStart == static_cast<Size>(-1) ? m_indices.size() : m_indexEnd - m_indexStart};
                }

                [[nodiscard]] Size index(Size i) const {
                    return m_indexStart == static_cast<Size>(-1) ? m_indices[i] : m_indexStart + i;
                }

                static constexpr int kDepth = Storage::kDepth + 1;

            private:
                const NDArrayBase<DType, Derived, Storage> *m_parent;
                std::vector<Size> m_indices;
                Size m_indexStart{0};
                Size m_indexEnd{0};
            };

            template<typename DType, typename Parent, typename Storage, typename ParentStorage>
            class ConstIndex : public NDArrayShaped<DType, Parent, Storage> {
            public:
                ConstIndex() = default;

                ConstIndex(const ConstIndex &another);
                ConstIndex(ConstIndex &&another) noexcept;

                ConstIndex &operator=(const ConstIndex &another) = default;
                ConstIndex &operator=(ConstIndex &&another) noexcept = default;

                ConstIndex(const NDArrayBase<DType, Parent, ParentStorage> *parent, Size indexStart, Shape shape);
                ConstIndex(const NDArrayBase<DType, Parent, ParentStorage> *parent, const std::vector<Size> &indices, Shape shape);
            };

            template<typename DType, typename Parent, typename ParentStorage>
            using ConstIndexParent = ConstIndex<DType, Parent, ConstIndexStorage<DType, Parent, ParentStorage>, ParentStorage>;

            template<typename DType, typename Parent, typename ParentStorage>
            using ConstIndexParentPtr = std::shared_ptr<ConstIndexParent<DType, Parent, ParentStorage>>;
        }// namespace internal
    }    // namespace ndarray
}// namespace np
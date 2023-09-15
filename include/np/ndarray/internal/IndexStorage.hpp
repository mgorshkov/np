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
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include <np/Shape.hpp>
#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/internal/OffsetType.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage>
            class NDArrayBase;

            template<typename DType, typename Storage, typename Parent>
            class IndexStorage {
            public:
                IndexStorage() = default;

                IndexStorage(const IndexStorage &) = default;

                IndexStorage(IndexStorage &&) noexcept = default;

                IndexStorage(Parent parent, const IndicesType<DType> &indices)
                    : m_parent{parent}, m_shape{}, m_weights{} {
                    initWeights();
                    initIndices(indices);
                    initShape(indices);
                }

                IndexStorage &operator=(const IndexStorage &) = default;
                IndexStorage &operator=(IndexStorage &&) noexcept = default;

                const DType &get(std::size_t i) const {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    return m_parent->get(m_indices[i]);
                }

                DType &get(std::size_t i) {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    return m_parent->get(m_indices[i]);
                }

                void set(std::size_t i, const DType &value) {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    return m_parent->set(m_indices[i], value);
                }

                class iterator {
                public:
                    using difference_type = OffsetType::difference_type;
                    using value_type = DType;
                    using pointer = DType *;
                    using reference = DType &;
                    using iterator_category = std::random_access_iterator_tag;

                    explicit iterator(IndexStorage *container)
                        : m_container{container}, m_offset{container->m_indices} {
                    }

                    iterator(IndexStorage *container, const OffsetType &offset)
                        : m_container{container}, m_offset{offset} {
                    }

                    iterator(const iterator &it)
                        : m_container{it.container}, m_offset{it.m_offset} {
                    }

                    iterator &operator=(const iterator &it) {
                        if (this != &it) {
                            m_container = it.m_container;
                            m_offset = it.m_offset;
                        }
                        return *this;
                    }

                    bool operator==(const iterator &it) const {
                        return m_container == it.m_container && m_offset == it.m_offset;
                    }

                    bool operator!=(const iterator &it) const {
                        return !(*this == it);
                    }

                    bool operator>(const iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset > it.m_offset;
                    }

                    bool operator>=(const iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset >= it.m_offset;
                    }

                    bool operator<(const iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset < it.m_offset;
                    }

                    bool operator<=(const iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset <= it.m_offset;
                    }

                    iterator operator++() {
                        return iterator(m_container, ++m_offset);
                    }

                    iterator operator++(int) {
                        return iterator(m_container, m_offset++);
                    }

                    iterator operator--() {
                        return iterator(m_container, --m_offset);
                    }

                    iterator operator--(int) {
                        return iterator(m_container, m_offset--);
                    }

                    iterator operator-(difference_type diff) {
                        return iterator(m_container, m_offset - diff);
                    }

                    iterator operator+(difference_type diff) {
                        return iterator(m_container, m_offset + diff);
                    }

                    iterator operator+=(difference_type diff) {
                        return iterator(m_container, m_offset += diff);
                    }

                    iterator operator-=(difference_type diff) {
                        return iterator(m_container, m_offset -= diff);
                    }

                    difference_type operator-(const iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset - it.m_offset;
                    }

                    iterator next(difference_type diff) {
                        return iterator(m_container, m_offset + diff);
                    }

                    reference operator*() const {
                        return m_container->get(static_cast<Size>(m_offset));
                    }

                private:
                    IndexStorage *m_container;
                    OffsetType m_offset;
                };

                iterator begin() {
                    return iterator{this, OffsetType{m_indices}};
                }

                iterator end() {
                    return iterator{this, OffsetType{m_indices, static_cast<Size>(m_indices.size())}};
                }

                class const_iterator {
                public:
                    using difference_type = std::ptrdiff_t;
                    using value_type = DType;
                    using pointer = DType *;
                    using reference = DType &;
                    using iterator_category = std::random_access_iterator_tag;

                    explicit const_iterator(const IndexStorage *container)
                        : m_container{container}, m_offset{container->m_indices} {
                    }

                    const_iterator(const IndexStorage *container, const OffsetType &offset)
                        : m_container{container}, m_offset{offset} {
                    }

                    const_iterator(const const_iterator &it)
                        : m_container{it.m_container}, m_offset{it.m_offset} {
                    }

                    const_iterator &operator=(const const_iterator &it) {
                        if (this != &it) {
                            m_container = it.m_container;
                            m_offset = it.m_offset;
                        }
                        return *this;
                    }

                    bool operator==(const const_iterator &it) const {
                        return m_container == it.m_container && m_offset == it.m_offset;
                    }

                    bool operator!=(const const_iterator &it) const {
                        return !(*this == it);
                    }

                    bool operator>(const const_iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset > it.m_offset;
                    }

                    bool operator>=(const const_iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset >= it.m_offset;
                    }

                    bool operator<(const const_iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset < it.m_offset;
                    }

                    bool operator<=(const const_iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset <= it.m_offset;
                    }

                    const_iterator operator++() {
                        return const_iterator(m_container, ++m_offset);
                    }

                    const_iterator operator++(int) {
                        return const_iterator(m_container, m_offset++);
                    }

                    const_iterator operator--() {
                        return const_iterator(m_container, --m_offset);
                    }

                    const_iterator operator--(int) {
                        return const_iterator(m_container, m_offset--);
                    }

                    const_iterator operator-(difference_type diff) {
                        return const_iterator(m_container, m_offset - diff);
                    }

                    const_iterator operator+(difference_type diff) {
                        return const_iterator(m_container, m_offset + diff);
                    }

                    const_iterator operator+=(difference_type diff) {
                        return const_iterator(m_container, m_offset += diff);
                    }

                    const_iterator operator-=(difference_type diff) {
                        return const_iterator(m_container, m_offset -= diff);
                    }

                    difference_type operator-(const const_iterator &it) const {
                        assert(m_container == it.m_container);
                        return m_offset - it.m_offset;
                    }

                    const_iterator next(difference_type diff) {
                        return const_iterator(m_container, m_offset + diff);
                    }

                    const value_type &operator*() const {
                        return m_container->get(static_cast<Size>(m_offset));
                    }

                private:
                    const IndexStorage *m_container;
                    OffsetType m_offset;
                };

                const_iterator cbegin() const {
                    return const_iterator{this, OffsetType{m_indices}};
                }

                const_iterator cend() const {
                    return const_iterator{this, OffsetType{m_indices, static_cast<Size>(m_indices.size())}};
                }

                [[nodiscard]] Shape shape() const {
                    return m_shape;
                }

                void setShape(const Shape &shape) {
                    m_shape = shape;
                }

                static constexpr std::size_t kDepth = Storage::kDepth + 1;

            private:
                static bool hasBooleanIndexing(const IndicesType<DType> &indices) {
                    bool hasBooleanIndexing = false;
                    for (std::size_t i = 0; i < indices.size(); ++i) {
                        hasBooleanIndexing = std::holds_alternative<BooleanIndexType<DType>>(indices[i]);
                        if (i != indices.size() - 1 && hasBooleanIndexing) {
                            throw std::runtime_error("Boolean index must be the last one");
                        }
                    }
                    return hasBooleanIndexing;
                }

                void initWeights() {
                    m_weights.resize(m_parent->ndim());
                    Size multiplier{1UL};
                    for (std::size_t pos = m_weights.size(); pos--;) {
                        m_weights[pos] = multiplier;
                        multiplier *= m_parent->shape()[pos];
                    }
                }

                [[nodiscard]] Shape::Storage getDimIndices(Size index) const {
                    Shape::Storage dimIndices(m_parent->ndim());
                    for (std::size_t pos = 0; pos < dimIndices.size(); ++pos) {
                        dimIndices[pos] = index / m_weights[pos];
                        index -= dimIndices[pos] * m_weights[pos];
                    }
                    return dimIndices;
                }

                void initIndices(const IndicesType<DType> &indices) {
                    std::vector<std::uint8_t> absentElements(m_parent->size());
                    for (std::size_t dim = 0; dim < indices.size(); ++dim) {
#pragma omp parallel for default(none) shared(indices, dim, absentElements)
                        // index variable in OpenMP 'for' statement must have signed integral type
                        for (std::int32_t i = 0; i < static_cast<std::int32_t>(absentElements.size()); ++i) {
                            auto dimIndices = getDimIndices(i);
                            if (std::holds_alternative<SubsettingIndexType>(indices[dim])) {
                                if (dimIndices[dim] != std::get<SubsettingIndexType>(indices[dim])) {
                                    absentElements[i] = 1;
                                }
                            } else if (std::holds_alternative<SlicingIndexType>(indices[dim])) {
                                auto [start, stop, step] = std::get<SlicingIndexType>(indices[dim]);
                                if (dimIndices[dim] < start || dimIndices[dim] >= stop || (dimIndices[dim] - start) % step != 0) {
                                    absentElements[i] = 1;
                                }
                            } else if (std::holds_alternative<BooleanIndexType<DType>>(indices[dim])) {
                                const DType &value = m_parent->get(i);
                                auto booleanIndex = std::get<BooleanIndexType<DType>>(indices[dim]);
                                auto pred = [&booleanIndex](const DType &value) {
                                    switch (booleanIndex.m_operator) {
                                        case Operator::More:
                                            return value > booleanIndex.m_arg;
                                        case Operator::MoreOrEqual:
                                            return value >= booleanIndex.m_arg;
                                        case Operator::Equal:
                                            return value == booleanIndex.m_arg;
                                        case Operator::LessOrEqual:
                                            return value <= booleanIndex.m_arg;
                                        case Operator::NotEqual:
                                            return value != booleanIndex.m_arg;
                                        case Operator::Less:
                                            return value < booleanIndex.m_arg;
                                        default:
                                            throw std::runtime_error("Invalid operator");
                                    }
                                };
                                if (!pred(value)) {
                                    absentElements[i] = 1;
                                }
                            } else {
                                throw std::runtime_error("Invalid index type");
                            }
                        }
                    }
                    for (Size i = 0; i < absentElements.size(); ++i) {
                        if (absentElements[i] == 0) {
                            m_indices.emplace_back(i);
                        }
                    }
                }

                void initShape(const IndicesType<DType> &indices) {
                    if (hasBooleanIndexing(indices)) {
                        m_shape.addDim(static_cast<Size>(m_indices.size()));
                    } else {
                        auto dimIndicesFront = getDimIndices(m_indices.front());
                        auto dimIndicesBack = getDimIndices(m_indices.back());
                        for (std::size_t pos = dimIndicesBack.size(); pos--;) {
                            dimIndicesBack[pos] -= dimIndicesFront[pos];// delta + 1 is dim
                            ++dimIndicesBack[pos];
                        }
                        for (std::size_t dim = 0; dim < dimIndicesBack.size(); ++dim) {
                            if (dim >= indices.size() || std::holds_alternative<SlicingIndexType>(indices[dim])) {
                                m_shape.addDim(dimIndicesBack[dim]);
                            }
                        }
                        if (m_shape.empty()) {
                            m_shape.addDim(1UL);
                        }
                    }
                }

                Parent m_parent;
                std::vector<Size> m_indices;
                Shape m_shape;

                Shape::Storage m_weights;

                friend class iterator;
                friend class const_iterator;
            };
        }// namespace internal
    }    // namespace ndarray
}// namespace np
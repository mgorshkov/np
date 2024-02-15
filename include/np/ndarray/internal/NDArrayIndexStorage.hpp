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

#include <np/Index.hpp>
#include <np/Shape.hpp>
#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/internal/OffsetType.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage>
            class NDArrayBase;

            template<typename DType, typename Storage, typename Parent>
            class NDArrayIndexStorage {
            public:
                NDArrayIndexStorage() = default;

                NDArrayIndexStorage(const NDArrayIndexStorage &) = default;

                NDArrayIndexStorage(NDArrayIndexStorage &&) noexcept = default;

                NDArrayIndexStorage(Parent parent, const IndicesType<DType> &indices)
                    : m_parent{parent}, m_indices{parent} {
                    initIndices(indices);
                    initShape(indices);
                }

                NDArrayIndexStorage &operator=(const NDArrayIndexStorage &) = default;
                NDArrayIndexStorage &operator=(NDArrayIndexStorage &&) noexcept = default;

                const DType &get(Size i) const {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    if (m_parent == nullptr) {
                        throw std::runtime_error("Parent is nullptr");
                    }
                    return m_parent->get(m_indices[i]);
                }

                DType &get(Size i) {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    if (m_parent == nullptr) {
                        throw std::runtime_error("Parent is nullptr");
                    }
                    return m_parent->get(m_indices[i]);
                }

                void set(Size i, const DType &value) {
                    if (i >= m_indices.size()) {
                        throw std::runtime_error("Index out of bounds");
                    }
                    if (m_parent == nullptr) {
                        throw std::runtime_error("Parent is nullptr");
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

                    explicit iterator(NDArrayIndexStorage *container)
                        : m_container{container}, m_offset{container->m_indices} {
                    }

                    iterator(NDArrayIndexStorage *container, const OffsetType &offset)
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
                    NDArrayIndexStorage *m_container;
                    OffsetType m_offset;
                };

                iterator begin() {
                    return iterator{this, OffsetType{}};
                }

                iterator end() {
                    return iterator{this, OffsetType{m_indices.size()}};
                }

                class const_iterator {
                public:
                    using difference_type = std::ptrdiff_t;
                    using value_type = DType;
                    using pointer = DType *;
                    using reference = DType &;
                    using iterator_category = std::random_access_iterator_tag;

                    explicit const_iterator(const NDArrayIndexStorage *container)
                        : m_container{container}, m_offset{container->m_indices} {
                    }

                    const_iterator(const NDArrayIndexStorage *container, const OffsetType &offset)
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
                    const NDArrayIndexStorage *m_container;
                    OffsetType m_offset;
                };

                const_iterator cbegin() const {
                    return const_iterator{this, OffsetType{}};
                }

                const_iterator cend() const {
                    return const_iterator{this, OffsetType{m_indices.size()}};
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

                class Indices {
                public:
                    explicit Indices(Parent parent) : m_parent{parent}, m_weights{} {
                        if (parent != nullptr) {
                            m_shape = parent->shape();
                        }
                        initWeights();
                    }

                    void add(std::size_t dim, const SubsettingIndexType &index) {
                        auto ror = createROR(dim, Range{index, index, 0});
                        intersectRORs(ror, dim);
                    }

                    void add(std::size_t dim, const SlicingIndexType &index) {
                        auto [start, stop, step] = index;
                        if (start == 0 && stop == m_shape[dim] && step == 1) {
                            return;// no limitation on complete index
                        }
                        auto ror = createROR(dim, Range{start, stop - 1, step});
                        intersectRORs(ror, dim);
                    }

                    void add(const BooleanIndexType<DType> &index) {
                        auto booleanIndex = [&index](const DType &value) {
                            switch (index.m_operator) {
                                case Operator::More:
                                    return value > index.m_arg;
                                case Operator::MoreOrEqual:
                                    return value >= index.m_arg;
                                case Operator::Equal:
                                    return value == index.m_arg;
                                case Operator::LessOrEqual:
                                    return value <= index.m_arg;
                                case Operator::NotEqual:
                                    return value != index.m_arg;
                                case Operator::Less:
                                    return value < index.m_arg;
                                default:
                                    throw std::runtime_error("Invalid operator");
                            }
                        };
                        if (m_parent == nullptr) {
                            throw std::runtime_error("Parent is nullptr");
                        }
                        std::vector<Size> bi;
                        for (Size i = 0; i < size(); ++i) {
                            auto offset = operator[](i);
                            const auto &element = m_parent->get(offset);
                            if (booleanIndex(element)) {
                                bi.emplace_back(offset);
                            }
                        }
                        m_booleanIndex = bi;
                    }

                    [[nodiscard]] Size size() const {
                        if (!m_booleanIndex.empty()) {
                            return static_cast<Size>(m_booleanIndex.size());
                        }
                        Size size = m_ror.size();
                        if (size == 0) {
                            if (m_parent == nullptr) {
                                throw std::runtime_error("Parent is nullptr");
                            }
                            return m_parent->size();
                        }
                        return size * m_ror.start.size();
                    }

                    [[nodiscard]] Size front() const {
                        auto range = m_ror.start;
                        return range.start;
                    }

                    [[nodiscard]] Size back() const {
                        auto range = m_ror.stop;
                        return range.stop;
                    }

                    auto getFrontIndices() const {
                        return getDimIndices(front());
                    }

                    auto getBackIndices() const {
                        return getDimIndices(back());
                    }

                    Size operator[](Size i) const {
                        // use boolean index
                        if (!m_booleanIndex.empty()) {
                            return m_booleanIndex[i];
                        }

                        // use ranges
                        if (m_ror.empty()) {
                            return i;
                        }

                        auto start = m_ror.start;
                        if (i >= m_ror.size() * start.size()) {
                            throw std::runtime_error("Access over array bounds");
                        }

                        if (m_ror.size() == 1) {
                            return start.start + i * start.step;
                        }

                        auto start2 = start.start + m_ror.step;
                        if (start2 < start.stop) {
                            // overlapping ranges
                            auto range = Range{m_ror.start.start + (i % m_ror.size()) * m_ror.start.step,
                                               m_ror.start.stop + (i % m_ror.size()) * m_ror.start.step, m_ror.start.step};
                            return range.start + i / m_ror.size() * range.step;
                        }
                        // successive ranges
                        auto range = Range{m_ror.start.start + (i / m_ror.start.size()) * m_ror.step,
                                           m_ror.start.stop + (i / m_ror.start.size()) * m_ror.step, m_ror.start.step};
                        return range.start + (i % range.size()) * range.step;
                    }

                    explicit operator OffsetType() const {
                        return OffsetType();
                    }

                private:
                    [[nodiscard]] Shape::Storage getDimIndices(Size index) const {
                        Shape::Storage dimIndices(m_shape.size());
                        for (std::size_t pos = 0; pos < dimIndices.size(); ++pos) {
                            dimIndices[pos] = index / m_weights[pos];
                            index -= dimIndices[pos] * m_weights[pos];
                        }
                        return dimIndices;
                    }

                    void initWeights() {
                        m_weights.resize(m_shape.size());
                        Size multiplier{1UL};
                        for (std::size_t pos = m_weights.size(); pos--;) {
                            m_weights[pos] = multiplier;
                            multiplier *= m_shape[pos];
                        }
                    }

                    struct Range {
                        Range() noexcept : start{}, stop{}, step{-1} {
                        }

                        Range(Size start_, Size stop_, SignedSize step_) noexcept : start{start_}, stop{stop_}, step{step_} {
                        }

                        Size start;
                        Size stop;
                        SignedSize step;

                        [[nodiscard]] bool empty() const {
                            return start == 0 && stop == 0 && step == -1;
                        }

                        [[nodiscard]] Size size() const {
                            return step == 0 ? 1 : (stop - start) / step + 1;
                        }

                        void clear() {
                            start = 0;
                            stop = 0;
                            step = -1;
                        }

                        auto operator<=>(const Range &) const = default;

                        friend std::ostream &operator<<(std::ostream &stream, const Range &range) {
                            return stream << "Range[" << range.start << ", " << range.stop << ", " << range.step << "], size=" << range.size();
                        }
                    };

                    using Ranges = std::set<Range>;

                    struct ROR {
                        ROR()
                        noexcept : start{}, stop{}, step{-1} {
                        }

                        ROR(const Range &start_, const Range &stop_, SignedSize step_)
                        noexcept : start{start_}, stop{stop_}, step{step_} {
                        }

                        explicit ROR(const Ranges &ranges) {
                            if (ranges.empty()) {
                                return;
                            }
                            if (ranges.size() == 1) {
                                auto it = ranges.cbegin();
                                start = *it;
                                stop = *it;
                                step = 0;
                                return;
                            }
                            if (ranges.size() == 2) {
                                auto it = ranges.cbegin();
                                start = *it;
                                it = ranges.cend();
                                stop = *std::prev(it);
                                step = static_cast<SignedSize>(stop.start - start.start);
                                return;
                            }
                            auto it = ranges.cbegin();
                            start = *it;
                            it = std::next(it);
                            auto next = *it;
                            auto end = ranges.cend();
                            end = std::prev(end);
                            stop = *end;
                            step = static_cast<SignedSize>(next.start - start.start);
                        }

                        Range start;
                        Range stop;
                        SignedSize step;

                        [[nodiscard]] bool empty() const {
                            return start.empty() && stop.empty() && step == -1;
                        }

                        [[nodiscard]] Size size() const {
                            return empty() ? 0 : step == 0 ? 1
                                                           : (stop.start - start.start) / step + 1;
                        }

                        void clear() {
                            start.clear();
                            stop.clear();
                            step = -1;
                        }

                        auto operator<=>(const ROR &) const = default;

                        friend std::ostream &operator<<(std::ostream &stream, const ROR &ror) {
                            return stream << "ROR[" << ror.start << ", " << ror.stop << ", " << ror.step << "], size=" << ror.size();
                        }
                    };

                    ROR createROR(std::size_t dim, const Range &indexRange) {
                        // required dimension is the first: one range
                        if (dim == 0) {
                            auto size = m_shape.calcSizeByShape() / m_shape[dim];
                            // low/high bounds of all slice parts
                            Range range{indexRange.start * size, (indexRange.stop + 1) * size - 1, indexRange.start == indexRange.stop && size == 1 ? 0 : 1};
                            return ROR{range, range, 0};
                        }
                        // required dimension is the last: one range with non-trivial step
                        if (dim == m_shape.size() - 1) {
                            // low bound of all slice parts
                            Range low{indexRange.start, indexRange.stop, indexRange.start == indexRange.stop ? 0 : indexRange.step};
                            // high bound of all slice parts
                            Range high{m_shape.calcSizeByShape() - m_shape[dim] + indexRange.start,
                                       m_shape.calcSizeByShape() - m_shape[dim] + indexRange.stop,
                                       indexRange.start == indexRange.stop ? 0 : indexRange.step};
                            return ROR{low, high, low == high ? 0 : static_cast<SignedSize>(m_shape[dim])};
                        }
                        // multiple ranges for the rest dimensions
                        if (indexRange.step != 0 && indexRange.step != 1) {
                            throw std::runtime_error("Non-contiguous ranges are not currently supported");
                        }
                        Shape::Storage storage;
                        storage.resize(m_shape.size());
                        Shape shape{storage};
                        Size high = m_shape[0];
                        for (Size d = 1; d < dim; ++d) {
                            high *= m_shape[d];
                        }
                        Ranges ranges;
                        for (Size h = 0; h < high; ++h) {
                            shape[dim] = indexRange.start;
                            for (std::size_t low = shape.size() - 1; low > dim; low--) {
                                shape[low] = 0;
                            }
                            auto minIndex = ravel_multi_index(shape, m_shape);

                            shape[dim] = indexRange.stop;
                            for (std::size_t low = shape.size() - 1; low > dim; low--) {
                                shape[low] = m_shape[low] - 1;
                            }
                            auto maxIndex = ravel_multi_index(shape, m_shape);
                            ranges.insert({minIndex, maxIndex, minIndex == maxIndex ? 0 : 1});

                            // increment high digits
                            SignedSize d = static_cast<SignedSize>(dim) - 1;
                            while (d >= 0) {
                                ++shape[d];
                                if (shape[d] >= m_shape[d]) {
                                    shape[d] = 0;
                                    --d;
                                } else {
                                    break;
                                }
                            }
                        }

                        return ROR{ranges};
                    }

                    [[nodiscard]] bool isTrivial(const Range &range) const {
                        return range.start == 0 && range.stop + 1 == m_shape.calcSizeByShape() && range.step == 1;
                    }

                    [[nodiscard]] bool isTrivial(const Range &range, std::size_t dim) const {
                        return range.start == 0 && range.stop + 1 == m_shape[dim] && range.step == 1;
                    }

                    std::optional<Range> intersectRanges(const Range &range1, const Range &range2, std::size_t dim) {
                        if (range1.start > range2.stop || range2.start > range1.stop) {
                            return std::nullopt;
                        }

                        if (isTrivial(range2, dim) || isTrivial(range2)) {
                            return range1;
                        }

                        if (isTrivial(range1, dim) || isTrivial(range2)) {
                            return range2;
                        }

                        if (range1.step == 0 && range2.step == 0) {
                            if (range1.start == range2.start) {
                                return range1;
                            }
                        } else if (range1.step == 0) {
                            if (range2.start <= range1.start && range1.start <= range2.stop) {
                                if ((range2.start - range1.start) % range2.step == 0 &&
                                    (range2.start - range1.start) / range2.step >= 0) {
                                    return range1;
                                }
                            }
                        } else if (range2.step == 0) {
                            if (range1.start <= range2.start && range2.start <= range1.stop) {
                                if ((range1.start - range2.start) % range1.step == 0 &&
                                    (range1.start - range2.start) / range1.step >= 0) {
                                    return range2;
                                }
                            }
                        }

                        // https://math.stackexchange.com/questions/1656120/formula-to-find-the-first-intersection-of-two-arithmetic-progressions
                        auto extGCD = extendedGCD(-range2.step, range1.step);
                        auto g = extGCD.gcd;

                        if ((range2.start - range1.start) % g != 0) {
                            return std::nullopt;
                        }

                        auto [u, v] = extGCD.bezout_coeff;
                        SignedSize c = range2.start - range1.start;
                        Size min_t = static_cast<Size>(std::max(
                                std::ceil(static_cast<float_>(-c) * static_cast<float_>(u) / range1.step),
                                std::ceil(static_cast<float_>(-c) * static_cast<float_>(v) / range2.step)));
                        Size max_t = std::min(((range2.size() - 1) * g - c * u) / range1.step,
                                              ((range1.size() - 1) * g - c * v) / range2.step);
                        assert(min_t <= max_t);

                        Size Xmin = (c * u + min_t * range1.step) / g;

                        Size X0 = range2.start + Xmin * range2.step;
                        assert(X0 == range1.start + (c * v + min_t * range2.step) / g * range1.step);
                        Size X1 = range2.start + (c * u + (min_t + 1) * range1.step) / g * range2.step;

                        Size Xmax = (c * u + max_t * range1.step) / g;
                        Size Xend = range2.start + Xmax * range2.step;

                        return Range{X0, Xend, Xend > X0 ? static_cast<SignedSize>(X1 - X0) : 0};
                    }

                    void intersectRORs(const ROR &ror, std::size_t dim) {
                        if (m_ror.empty()) {
                            m_ror = ror;
                            return;
                        }
                        // Non-intersecting RORs
                        if (ror.start.start > m_ror.stop.stop || ror.stop.stop < m_ror.start.start) {
                            m_ror.clear();
                            return;
                        }

                        // Zero step in RORs
                        Ranges ranges;
                        if (ror.step == 0 && m_ror.step == 0) {
                            if (ror.start == m_ror.start) {
                                ranges.insert(ror.start);
                            }
                        } else if (ror.step == 0) {
                            for (Size i = 0; i < m_ror.size(); ++i) {
                                Range range2{m_ror.start.start + i * m_ror.step, m_ror.start.stop + i * m_ror.step,
                                             m_ror.start.step};
                                auto range = intersectRanges(ror.start, range2, dim);
                                if (range) {
                                    ranges.insert(*range);
                                }
                            }
                        } else if (m_ror.step == 0) {
                            for (Size i = 0; i < ror.size(); ++i) {
                                Range range2{ror.start.start + i * ror.step, ror.start.stop + i * ror.step,
                                             ror.start.step};
                                auto range = intersectRanges(m_ror.start, range2, dim);
                                if (range) {
                                    ranges.insert(*range);
                                }
                            }
                        } else {
                            for (Size i = 0; i < m_ror.size(); ++i) {
                                for (Size j = 0; j < ror.size(); ++j) {
                                    Range range1{m_ror.start.start + i * m_ror.step, m_ror.start.stop + i * m_ror.step,
                                                 m_ror.start.step};
                                    Range range2{ror.start.start + j * ror.step, ror.start.stop + j * ror.step,
                                                 ror.start.step};

                                    auto range = intersectRanges(range1, range2, dim);
                                    if (range) {
                                        ranges.insert(*range);
                                    }
                                }
                            }
                        }

                        m_ror = ROR{ranges};
                    }

                    Parent m_parent;
                    Shape m_shape;
                    Shape::Storage m_weights;

                    ROR m_ror;// ror to iterate through ("OR" condition)
                    std::vector<Size> m_booleanIndex;
                };

                void initIndices(const IndicesType<DType> &indices) {
                    for (std::size_t dim = 0; dim < indices.size(); ++dim) {
                        if (std::holds_alternative<SubsettingIndexType>(indices[dim])) {
                            m_indices.add(dim, std::get<SubsettingIndexType>(indices[dim]));
                        } else if (std::holds_alternative<SlicingIndexType>(indices[dim])) {
                            m_indices.add(dim, std::get<SlicingIndexType>(indices[dim]));
                        } else if (std::holds_alternative<BooleanIndexType<DType>>(indices[dim])) {
                            m_indices.add(std::get<BooleanIndexType<DType>>(indices[dim]));
                        } else {
                            throw std::runtime_error("Invalid index type");
                        }
                    }
                }

                void initShape(const IndicesType<DType> &indices) {
                    m_shape.clear();
                    if (hasBooleanIndexing(indices)) {
                        m_shape.addDim(m_indices.size());
                    } else {
                        auto dimIndicesFront = m_indices.getFrontIndices();
                        auto dimIndicesBack = m_indices.getBackIndices();
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
                Indices m_indices;
                Shape m_shape;

                friend class iterator;
                friend class const_iterator;
            };
        }// namespace internal
    }    // namespace ndarray
}// namespace np
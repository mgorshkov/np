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
#include <vector>
#include <string>

#include <np/Constants.hpp>

namespace np::ndarray::array_static::internal {

    template <typename DType, Size... Sizes>
    class NDArrayStaticInternal;

    template <typename DType, Size SizeT, Size... SizeTs>
    std::ostream& operator<< (std::ostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array);
    
    template <Size SizeT, Size... SizeTs>
    std::ostream& operator<< (std::ostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);

    template <typename DType, Size SizeT, Size... SizeTs>
    std::wostream& operator<< (std::wostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array);

    template <Size SizeT, Size... SizeTs>
    std::wostream& operator<< (std::wostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);

    template <typename DType>
    class NDArrayStaticInternal<DType> {
    };

    // Termination template
    template <typename DType, Size SizeT>
    class NDArrayStaticInternal<DType, SizeT> {
    public:
        using CArrayType = DType[SizeT];
        using StdArrayType = std::array<DType, SizeT>;
        using StdVectorType = std::vector<DType>;

        NDArrayStaticInternal() noexcept = default;

        NDArrayStaticInternal(const DType& data) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = data;
            }
        }

        NDArrayStaticInternal(CArrayType array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
        }

        NDArrayStaticInternal(CArrayType&& array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                std::move(m_Impl[i], array[i]);
            }
        }

        NDArrayStaticInternal(const NDArrayStaticInternal &another) noexcept
            : m_Impl{another.m_Impl}
        {
        }

        NDArrayStaticInternal(NDArrayStaticInternal &&another) noexcept
            : m_Impl{std::move(another.m_Impl)}
        {
        }

        NDArrayStaticInternal(const StdArrayType &array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
        }

        NDArrayStaticInternal(StdArrayType &&array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(array[i]);
            }
        }

        NDArrayStaticInternal(const StdVectorType &vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = vector[i];
            }
        }

        NDArrayStaticInternal(StdVectorType &&vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(vector[i]);
            }
        }

        NDArrayStaticInternal(std::initializer_list<DType> init_list) noexcept {
            if (init_list.size() == 1 && SizeT > 1) {
                // fill
                m_Impl.fill(*init_list.begin());
            } else {
                std::size_t index = 0;
                for (const auto& element: init_list) {
                    m_Impl[index++] = element;
                }
            }
        }

        NDArrayStaticInternal &operator=(DType data[SizeT]) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = data[i];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const StdArrayType &array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const StdVectorType &vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = vector[i];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const NDArrayStaticInternal &another) noexcept {
            if (this != &another) {
                for (auto i = 0; i < SizeT; ++i) {
                    m_Impl[i] = another[i];
                }
            }
            return *this;
        }

        DType &operator[](std::size_t i) {
            return m_Impl[i];
        }

        const DType &operator[](std::size_t i) const {
            return m_Impl[i];
        }

        void set(std::size_t i, const DType& element) {
            m_Impl[i] = element;
        }

        const DType& get(std::size_t i) const {
            return m_Impl[i];
        }

        DType& get(std::size_t i) {
            return m_Impl[i];
        }

        friend std::ostream& operator<< <>(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT> &array);
        friend std::ostream& operator<< <>(std::ostream &stream, const NDArrayStaticInternal<std::wstring, SizeT> &array);
        friend std::wostream& operator<< <>(std::wostream &stream, const NDArrayStaticInternal<std::wstring, SizeT> &array);

        inline friend void dumpToStreamAsBinary(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT> &array) {
            for (auto i = 0; i < SizeT; ++i) {
                stream.write(reinterpret_cast<const char*>(&array.m_Impl[i]), sizeof(array.m_Impl[i]));
            }
        }

        class iterator {
        public:
            typedef ptrdiff_t difference_type;
            typedef DType value_type;
            typedef DType* pointer;
            typedef DType& reference;
            typedef std::random_access_iterator_tag iterator_category;

            iterator(NDArrayStaticInternal* container_, std::size_t offset_)
                    : container{container_}
                    , offset{offset_}{
            }

            iterator(const iterator& it)
                    : container{it.container}
                    , offset{it.offset}{
            }

            iterator& operator = (const iterator& it) {
                if (this != &it) {
                    container = it.container;
                    offset = it.offset;
                }
                return *this;
            }

            bool operator == (const iterator& it) const {
                return container == it.container && offset == it.offset;
            }

            bool operator != (const iterator& it) const {
                return !(*this == it);
            }

            bool operator > (const iterator& it) const {
                assert(container == it.container);
                return offset > it.offset;
            }

            bool operator >= (const iterator& it) const {
                assert(container == it.container);
                return offset >= it.offset;
            }

            bool operator < (const iterator& it) const {
                assert(container == it.container);
                return offset < it.offset;
            }

            bool operator <= (const iterator& it) const {
                assert(container == it.container);
                return offset <= it.offset;
            }

            iterator operator ++ () {
                return iterator(container, ++offset);
            }

            iterator operator ++ (int) {
                return iterator(container, offset++);
            }

            iterator operator -- () {
                return iterator(container, --offset);
            }

            iterator operator -- (int) {
                return iterator(container, offset--);
            }

            difference_type operator - (const iterator& it) const {
                assert(container == it.container);
                return offset - it.offset;
            }

            iterator operator - (difference_type diff) {
                return iterator(container, offset - diff);
            }

            iterator operator + (difference_type diff) {
                return iterator(container, offset + diff);
            }

            iterator operator += (difference_type diff) {
                return iterator(container, offset += diff);
            }

            iterator operator -= (difference_type diff) {
                return iterator(container, offset -= diff);
            }

            value_type& operator * () {
                return container->get(offset);
            }

            const value_type& operator * () const {
                return container->get(offset);
            }

        private:
            NDArrayStaticInternal* container;
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
            typedef ptrdiff_t difference_type;
            typedef DType value_type;
            typedef DType* pointer;
            typedef DType& reference;
            typedef std::random_access_iterator_tag iterator_category;

            const_iterator(const NDArrayStaticInternal* container_, std::size_t offset_)
                    : container{container_}
                    , offset{offset_}{
            }

            const_iterator(const const_iterator& it)
                    : container{it.container}
                    , offset{it.offset}{
            }

            const_iterator& operator = (const const_iterator& it) {
                if (this != &it) {
                    container = it.container;
                    offset = it.offset;
                }
                return *this;
            }

            bool operator == (const const_iterator& it) const {
                return container == it.container && offset == it.offset;
            }

            bool operator != (const const_iterator& it) const {
                return !(*this == it);
            }

            bool operator > (const const_iterator& it) const {
                assert(container == it.container);
                return offset > it.offset;
            }

            bool operator >= (const const_iterator& it) const {
                assert(container == it.container);
                return offset >= it.offset;
            }

            bool operator < (const const_iterator& it) const {
                assert(container == it.container);
                return offset < it.offset;
            }

            bool operator <= (const const_iterator& it) const {
                assert(container == it.container);
                return offset <= it.offset;
            }

            const_iterator operator ++ () {
                return const_iterator(container, ++offset);
            }

            const_iterator operator ++ (int) {
                return const_iterator(container, offset++);
            }

            const_iterator operator -- () {
                return const_iterator(container, --offset);
            }

            const_iterator operator -- (int) {
                return const_iterator(container, offset--);
            }

            difference_type operator - (const const_iterator& it) const {
                assert(container == it.container);
                return offset - it.offset;
            }

            const_iterator operator - (difference_type diff) {
                return const_iterator(container, offset - diff);
            }

            const_iterator operator + (difference_type diff) {
                return const_iterator(container, offset + diff);
            }

            const_iterator operator += (difference_type diff) {
                return const_iterator(container, offset += diff);
            }

            const_iterator operator -= (difference_type diff) {
                return const_iterator(container, offset -= diff);
            }

            const value_type& operator * () const {
                return container->get(offset);
            }

        private:
            const NDArrayStaticInternal* container;
            std::size_t offset;
        };

        const_iterator cbegin() const {
            return const_iterator{this, 0};
        }

        const_iterator cend() const {
            return const_iterator{this, SizeT};
        }

    private:
        StdArrayType m_Impl;
    };

    template <typename DType, Size SizeT, Size... SizeTs>
    class NDArrayStaticInternal<DType, SizeT, SizeTs...> {
    public:
        using ReducedNDArrayStaticInternal = NDArrayStaticInternal<DType, SizeTs...>;

        using ReducedType = typename std::conditional<
            sizeof...(SizeTs) == 0,
            NDArrayStaticInternal<DType, SizeT>,
            ReducedNDArrayStaticInternal>::type;

        using ReducedCArrayType = typename ReducedType::CArrayType;
        using ReducedStdArrayType = typename ReducedType::StdArrayType;
        using ReducedStdVectorType = typename ReducedType::StdVectorType;

        using CArrayType = ReducedCArrayType[SizeT];
        using StdArrayType = std::array<ReducedStdArrayType, SizeT>;
        using StdVectorType = std::vector<ReducedStdVectorType>;

        NDArrayStaticInternal() noexcept = default;

        NDArrayStaticInternal(const DType &value) {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = value;
            }
        }

        NDArrayStaticInternal(const NDArrayStaticInternal &another) noexcept {
            for (std::size_t i = 0; i < SizeT; ++i) {
                m_Impl[i] = another[i];
            }
        }

        NDArrayStaticInternal(NDArrayStaticInternal &&another) noexcept {
            for (std::size_t i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(another[i]);
            }
        }

        NDArrayStaticInternal(ReducedCArrayType data[SizeT]) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = data[index];
            }
        }

        NDArrayStaticInternal(const std::array<ReducedStdArrayType, SizeT> &array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = array[index];
            }
        }

        NDArrayStaticInternal(std::array<ReducedStdArrayType, SizeT> &&array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = std::move(array.m_Impl[index]);
            }
        }

        NDArrayStaticInternal(const std::vector<ReducedStdVectorType> &vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
        }

        NDArrayStaticInternal(std::vector<ReducedStdVectorType> &&vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
        }

        NDArrayStaticInternal(std::initializer_list<DType> init_list) noexcept {
            if (init_list.size() == 1 && SizeT > 1) {
                // fill
                for (std::size_t index = 0; index < SizeT; ++index) {
                    m_Impl[index] = *init_list.begin();
                }
            } else {
                std::size_t index = 0;
                for (const auto& element : init_list) {
                    m_Impl[index++] = element;
                }
            }
        }

        NDArrayStaticInternal &operator=(const DType &value) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = value;
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(ReducedCArrayType data[SizeT]) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = data[index];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const std::array<ReducedStdArrayType, SizeT> &array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = array[index];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const std::vector<ReducedStdVectorType> &vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
            return *this;
        }

        NDArrayStaticInternal &operator=(const NDArrayStaticInternal &another) noexcept {
            if (this != &another) {
                m_Impl = another.m_Impl;
            }
            return *this;
        }

        ReducedType &operator[](std::size_t i) {
            return m_Impl[i];
        }

        const ReducedType &operator[](std::size_t i) const {
            return m_Impl[i];
        }

        void set(std::size_t i, const ReducedType &array) {
            m_Impl.set(i, array);
        }

        const DType& get(std::size_t i) const {
            static constexpr Size size = (SizeT * ... * SizeTs);
            auto index1 = i / (size / SizeT);
            auto index2 = i % (size / SizeT);
            return m_Impl[index1].get(index2);
        }

        DType& get(std::size_t i) {
            static constexpr Size size = (SizeT * ... * SizeTs);
            auto index1 = i / (size / SizeT);
            auto index2 = i % (size / SizeT);
            return m_Impl[index1].get(index2);
        }

        void set(std::size_t i, const DType& value) {
            static constexpr Size size = (SizeT * ... * SizeTs);
            auto index1 = i / (size / SizeT);
            auto& subArray = m_Impl[index1];

            if constexpr (std::is_same<DType, typename std::remove_reference<decltype(subArray)>::type>::value) {
                subArray = value;
            } else {
                auto index2 = i % (size / SizeT);
                subArray.set(index2, value);
            }
        }

        friend std::ostream & operator<< <> (std::ostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array);
        friend std::ostream & operator<< <> (std::ostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);
        friend std::wostream& operator<< <> (std::wostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);

        inline friend void dumpToStreamAsBinary(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array) {
            for (std::size_t index = 0; index < SizeT; ++index) {
                dumpToStreamAsBinary(stream, array[index]);
            }
        }

        class iterator {
        public:
            typedef ptrdiff_t difference_type;
            typedef DType value_type;
            typedef DType* pointer;
            typedef DType& reference;
            typedef std::random_access_iterator_tag iterator_category;

            iterator(NDArrayStaticInternal* container_, std::size_t offset_)
                : container{container_}
                , offset{offset_}{
            }

            iterator(const iterator& it)
                : container{it.container}
                , offset{it.offset}{
            }

            iterator& operator = (const iterator& it) {
                if (this != &it) {
                    container = it.container;
                    offset = it.offset;
                }
                return *this;
            }

            bool operator == (const iterator& it) const {
                return container == it.container && offset == it.offset;
            }

            bool operator != (const iterator& it) const {
                return !(*this == it);
            }

            bool operator > (const iterator& it) const {
                assert(container == it.container);
                return offset > it.offset;
            }

            bool operator >= (const iterator& it) const {
                assert(container == it.container);
                return offset >= it.offset;
            }

            bool operator < (const iterator& it) const {
                assert(container == it.container);
                return offset < it.offset;
            }

            bool operator <= (const iterator& it) const {
                assert(container == it.container);
                return offset <= it.offset;
            }

            iterator operator ++ () {
                return iterator(container, ++offset);
            }

            iterator operator ++ (int) {
                return iterator(container, offset++);
            }

            iterator operator -- () {
                return iterator(container, --offset);
            }

            iterator operator -- (int) {
                return iterator(container, offset--);
            }

            iterator operator - (difference_type diff) {
                return iterator(container, offset - diff);
            }

            iterator operator + (difference_type diff) {
                return iterator(container, offset + diff);
            }

            iterator operator += (difference_type diff) {
                return iterator(container, offset += diff);
            }

            iterator operator -= (difference_type diff) {
                return iterator(container, offset -= diff);
            }

            difference_type operator - (const iterator& it) const {
                assert(container == it.container);
                return offset - it.offset;
            }

            value_type& operator * () {
                return container->get(offset);
            }

        private:
            NDArrayStaticInternal* container;
            std::size_t offset;
        };

        iterator begin() {
            return iterator{this, 0};
        }

        iterator end() {
            static constexpr Size size = (SizeT * ... * SizeTs);
            return iterator{this, size};
        }

        class const_iterator {
        public:
            typedef ptrdiff_t difference_type;
            typedef DType value_type;
            typedef DType* pointer;
            typedef DType& reference;
            typedef std::random_access_iterator_tag iterator_category;

            const_iterator(const NDArrayStaticInternal* container_, std::size_t offset_)
                    : container{container_}
                    , offset{offset_}{
            }

            const_iterator(const const_iterator& it)
                    : container{it.container}
                    , offset{it.offset}{
            }

            const_iterator& operator = (const const_iterator& it) {
                if (this != &it) {
                    container = it.container;
                    offset = it.offset;
                }
                return *this;
            }

            bool operator == (const const_iterator& it) const {
                return container == it.container && offset == it.offset;
            }

            bool operator != (const const_iterator& it) const {
                return !(*this == it);
            }

            bool operator > (const const_iterator& it) const {
                assert(container == it.container);
                return offset > it.offset;
            }

            bool operator >= (const const_iterator& it) const {
                assert(container == it.container);
                return offset >= it.offset;
            }

            bool operator < (const const_iterator& it) const {
                assert(container == it.container);
                return offset < it.offset;
            }

            bool operator <= (const const_iterator& it) const {
                assert(container == it.container);
                return offset <= it.offset;
            }

            const_iterator operator ++ () {
                return const_iterator(container, ++offset);
            }

            const_iterator operator ++ (int) {
                return const_iterator(container, offset++);
            }

            const_iterator operator -- () {
                return const_iterator(container, --offset);
            }

            const_iterator operator -- (int) {
                return const_iterator(container, offset--);
            }

            const_iterator operator - (difference_type diff) {
                return const_iterator(container, offset - diff);
            }

            const_iterator operator + (difference_type diff) {
                return const_iterator(container, offset + diff);
            }

            const_iterator operator += (difference_type diff) {
                return const_iterator(container, offset += diff);
            }

            const_iterator operator -= (difference_type diff) {
                return const_iterator(container, offset -= diff);
            }

            difference_type operator - (const const_iterator& it) const {
                assert(container == it.container);
                return offset - it.offset;
            }

            const value_type& operator * () const {
                return container->get(offset);
            }

        private:
            const NDArrayStaticInternal* container;
            std::size_t offset;
        };

        const_iterator cbegin() const {
            return const_iterator{this, 0};
        }

        const_iterator cend() const {
            static constexpr Size size = (SizeT * ... * SizeTs);
            return const_iterator{this, size};
        }

    private:
        NDArrayStaticInternal<ReducedType, SizeT> m_Impl;
    };
}

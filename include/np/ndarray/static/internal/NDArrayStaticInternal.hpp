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

        inline NDArrayStaticInternal() noexcept {
        }

        inline NDArrayStaticInternal(const DType& data) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = data;
            }
        }

        inline NDArrayStaticInternal(CArrayType array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
        }

        inline NDArrayStaticInternal(CArrayType&& array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                std::move(m_Impl[i], array[i]);
            }
        }

        inline NDArrayStaticInternal(const NDArrayStaticInternal &another) noexcept
            : m_Impl{another.m_Impl}
        {
        }

        inline NDArrayStaticInternal(NDArrayStaticInternal &&another) noexcept
            : m_Impl{std::move(another.m_Impl)}
        {
        }

        inline NDArrayStaticInternal(const StdArrayType &array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
        }

        inline NDArrayStaticInternal(StdArrayType &&array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(array[i]);
            }
        }

        inline NDArrayStaticInternal(const StdVectorType &vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = vector[i];
            }
        }

        inline NDArrayStaticInternal(StdVectorType &&vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(vector[i]);
            }
        }

        inline NDArrayStaticInternal(std::initializer_list<DType> init_list) noexcept {
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

        inline NDArrayStaticInternal &operator=(DType data[SizeT]) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = data[i];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const StdArrayType &array) noexcept {
            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = array[i];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const StdVectorType &vector) noexcept {
            assert(SizeT == vector.size());

            for (auto i = 0; i < SizeT; ++i) {
                m_Impl[i] = vector[i];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const NDArrayStaticInternal &another) noexcept {
            if (this != &another) {
                for (auto i = 0; i < SizeT; ++i) {
                    m_Impl[i] = another[i];
                }
            }
            return *this;
        }

        inline DType &operator[](std::size_t i) {
            return m_Impl[i];
        }

        inline const DType &operator[](std::size_t i) const {
            return m_Impl[i];
        }

        inline void set(std::size_t i, const DType& element) {
            m_Impl[i] = element;
        }

        friend std::ostream& operator<< <>(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT> &array);
        friend std::ostream& operator<< <>(std::ostream &stream, const NDArrayStaticInternal<std::wstring, SizeT> &array);
        friend std::wostream& operator<< <>(std::wostream &stream, const NDArrayStaticInternal<std::wstring, SizeT> &array);

        inline friend void dumpToStreamAsBinary(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT> &array) {
            for (auto i = 0; i < SizeT; ++i) {
                stream.write(reinterpret_cast<const char*>(&array.m_Impl[i]), sizeof(array.m_Impl[i]));
            }
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

        inline NDArrayStaticInternal() noexcept {
        }

        inline NDArrayStaticInternal(const DType &value) {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = value;
            }
        }

        inline NDArrayStaticInternal(const NDArrayStaticInternal &another) noexcept {
            for (std::size_t i = 0; i < SizeT; ++i) {
                m_Impl[i] = another[i];
            }
        }

        inline NDArrayStaticInternal(NDArrayStaticInternal &&another) noexcept {
            for (std::size_t i = 0; i < SizeT; ++i) {
                m_Impl[i] = std::move(another[i]);
            }
        }

        inline NDArrayStaticInternal(ReducedCArrayType data[SizeT]) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = data[index];
            }
        }

        inline NDArrayStaticInternal(const std::array<ReducedStdArrayType, SizeT> &array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = array[index];
            }
        }

        inline NDArrayStaticInternal(std::array<ReducedStdArrayType, SizeT> &&array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = std::move(array.m_Impl[index]);
            }
        }

        inline NDArrayStaticInternal(const std::vector<ReducedStdVectorType> &vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
        }

        inline NDArrayStaticInternal(std::vector<ReducedStdVectorType> &&vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
        }

        inline NDArrayStaticInternal(std::initializer_list<DType> init_list) noexcept {
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

        inline NDArrayStaticInternal &operator=(const DType &value) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = value;
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(ReducedCArrayType data[SizeT]) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = data[index];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const std::array<ReducedStdArrayType, SizeT> &array) noexcept {
            for (std::size_t index = 0; index < SizeT; ++index) {
                m_Impl[index] = array[index];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const std::vector<ReducedStdVectorType> &vector) noexcept {
            assert(SizeT == vector.size());

            for (std::size_t index = 0; index < vector.size(); ++index) {
                m_Impl[index] = vector[index];
            }
            return *this;
        }

        inline NDArrayStaticInternal &operator=(const NDArrayStaticInternal &another) noexcept {
            if (this != &another) {
                m_Impl = another.m_Impl;
            }
            return *this;
        }

        inline ReducedType &operator[](std::size_t i) {
            return m_Impl[i];
        }

        inline const ReducedType &operator[](std::size_t i) const {
            return m_Impl[i];
        }

        inline void set(std::size_t i, const ReducedType &array) {
            m_Impl.set(i, array);
        }

        friend std::ostream & operator<< <> (std::ostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array);
        friend std::ostream & operator<< <> (std::ostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);
        friend std::wostream& operator<< <> (std::wostream &stream, const NDArrayStaticInternal<std::wstring, SizeT, SizeTs...> &array);

        inline friend void dumpToStreamAsBinary(std::ostream &stream, const NDArrayStaticInternal<DType, SizeT, SizeTs...> &array) {
            for (std::size_t index = 0; index < SizeT; ++index) {
                dumpToStreamAsBinary(stream, array[index]);
            }
        }

        static const constexpr std::tuple m_Shape{std::make_tuple(SizeT, SizeTs...)};
        static const constexpr std::size_t m_Dims{sizeof...(SizeTs) + 1};

    private:
        NDArrayStaticInternal<ReducedType, SizeT> m_Impl;
    };
}

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

                template<typename Storage>
                class Span {
                public:
                    using iterator = typename Storage::iterator;
                    using const_iterator = typename Storage::const_iterator;
                    using reference = typename Storage::reference;
                    using const_reference = typename Storage::const_reference;
                    using value_type = typename Storage::value_type;

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

                    iterator operator+=(difference_type diff) {
                        return iterator(begin_ += diff, end_);
                    }

                    iterator operator-=(difference_type diff) {
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
            }// namespace internal
        }    // namespace array_dynamic
    }        // namespace ndarray
}// namespace np
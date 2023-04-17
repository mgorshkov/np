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

#include <np/Shape.hpp>
#include <vector>

namespace np {
    namespace ndarray {
        namespace internal {
            using Positions = std::vector<Size>;

            class OffsetType {
            public:
                using difference_type = std::ptrdiff_t;

                OffsetType() : m_positions{}, m_currentPosition{0} {
                }

                explicit OffsetType(const Positions &positions) : m_positions{positions}, m_currentPosition{0} {
                }

                explicit OffsetType(const Positions &positions, Size currentPosition) : m_positions{positions}, m_currentPosition{currentPosition} {
                }

                OffsetType(const OffsetType &offsetType) : m_positions{offsetType.m_positions}, m_currentPosition{offsetType.m_currentPosition} {
                }

                bool operator==(const OffsetType &another) const {
                    if (m_positions != another.m_positions) {
                        throw std::runtime_error("Comparing offsets with different positions");
                    }
                    return m_currentPosition == another.m_currentPosition;
                }

                bool operator!=(const OffsetType &another) const {
                    return !(*this == another);
                }

                bool operator<(const OffsetType &another) const {
                    if (m_positions != another.m_positions) {
                        throw std::runtime_error("Comparing offsets with different positions");
                    }
                    return m_currentPosition < another.m_currentPosition;
                }

                bool operator<=(const OffsetType &another) const {
                    if (m_positions != another.m_positions) {
                        throw std::runtime_error("Comparing offsets with different positions");
                    }
                    return m_currentPosition <= another.m_currentPosition;
                }

                bool operator>(const OffsetType &another) const {
                    if (m_positions != another.m_positions) {
                        throw std::runtime_error("Comparing offsets with different positions");
                    }
                    return m_currentPosition > another.m_currentPosition;
                }

                bool operator>=(const OffsetType &another) const {
                    if (m_positions != another.m_positions) {
                        throw std::runtime_error("Comparing offsets with different positions");
                    }
                    return m_currentPosition >= another.m_currentPosition;
                }

                OffsetType operator++() {
                    if (m_currentPosition == m_positions.size()) {
                        throw std::runtime_error("Going beyond last position");
                    }
                    OffsetType offset{m_positions, ++m_currentPosition};
                    return offset;
                }

                OffsetType operator++(int) {
                    if (m_currentPosition == m_positions.size()) {
                        throw std::runtime_error("Going beyond last position");
                    }
                    OffsetType offset{m_positions, m_currentPosition++};
                    return offset;
                }

                OffsetType operator+(difference_type diff) {
                    OffsetType offsetType{*this};
                    for (int i = 0; i < diff; ++i) {
                        offsetType++;
                    }
                    return offsetType;
                }

                OffsetType operator+=(difference_type diff) {
                    for (int i = 0; i < diff; ++i) {
                        (*this)++;
                    }
                    return *this;
                }

                OffsetType operator--() {
                    if (m_currentPosition == 0) {
                        throw std::runtime_error("Going before zero position");
                    }
                    OffsetType offset{m_positions, --m_currentPosition};
                    return offset;
                }

                OffsetType operator--(int) {
                    if (m_currentPosition == 0) {
                        throw std::runtime_error("Going before zero position");
                    }
                    OffsetType offset{m_positions, m_currentPosition--};
                    return offset;
                }

                OffsetType operator-(difference_type diff) {
                    OffsetType offsetType{*this};
                    for (int i = 0; i < diff; ++i) {
                        offsetType--;
                    }
                    return offsetType;
                }

                OffsetType operator-=(difference_type diff) {
                    for (int i = 0; i < diff; ++i) {
                        (*this)--;
                    }
                    return *this;
                }

                difference_type operator-(const OffsetType &another) const {
                    OffsetType offsetType(*this);
                    difference_type res = 0;
                    while (offsetType != another) {
                        offsetType--;
                        ++res;
                    }
                    return res;
                }

                explicit operator Size() const {
                    return m_currentPosition;
                }

            private:
                Positions m_positions;
                Size m_currentPosition;
            };

        }// namespace internal
    }    // namespace ndarray
}// namespace np
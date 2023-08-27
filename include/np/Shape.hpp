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
#include <cctype>
#include <cstddef>
#include <numeric>
#include <vector>

#include <np/Constants.hpp>
#include <np/Exception.hpp>

namespace np {
    //////////////////////////////////////////////////////////////
    /// \brief Shape of an array
    ///
    /// Each dimension is represented by a vector element.
    ///
    //////////////////////////////////////////////////////////////
    class Shape {
    public:
        Shape() = default;

        using Storage = std::vector<Size>;

        //////////////////////////////////////////////////////////////
        /// \brief Initializer constructor
        ///
        /// Initialize shape with a list of dimensions.
        ///
        /// \param initList list of dimensions
        ///
        //////////////////////////////////////////////////////////////
        Shape(std::initializer_list<Size> initList)
            : m_sizes{initList} {
        }

        //////////////////////////////////////////////////////////////
        /// \brief Initializer constructor
        ///
        /// Initialize shape with a vector of dimensions.
        ///
        /// \param storage vector of dimensions
        ///
        //////////////////////////////////////////////////////////////
        explicit Shape(const Storage &storage)
            : m_sizes{storage} {
        }

        //////////////////////////////////////////////////////////////
        /// \brief Move constructor
        ///
        /// Initialize shape with a vector of dimensions.
        ///
        /// \param storage vector of dimensions
        ///
        //////////////////////////////////////////////////////////////
        explicit Shape(Storage &&storage)
            : m_sizes{std::move(storage)} {
        }

        Shape(const Shape &another) = default;
        Shape(Shape &&another) = default;

        Shape &operator=(const Shape &another) {
            if (this != &another) {
                m_sizes = another.m_sizes;
            }
            return *this;
        }

        Shape &operator=(const Storage &storage) {
            m_sizes = storage;
            return *this;
        }

        Shape &operator=(Storage &&storage) noexcept {
            m_sizes = std::move(storage);
            return *this;
        }

        //////////////////////////////////////////////////////////////
        /// \brief Initializer constructor
        ///
        /// Initialize shape with a string with dimensions.
        /// 0,
        //  1,
        //  1, 2
        //  1, 2, 3
        //
        /// \param shapeTupleStr string with dimensions
        ///
        //////////////////////////////////////////////////////////////
        explicit Shape(const std::string &shapeTupleStr) {
            // 0,
            // 1,
            // 1, 2
            // 1, 2, 3
            if (shapeTupleStr == "0,")
                return;// empty
            Size prevCommaIndex = 0;
            auto push_number = [this, &shapeTupleStr](Size commaIndex, Size prevCommaIndex) {
                if (commaIndex == prevCommaIndex)
                    return;
                auto number{shapeTupleStr.substr(prevCommaIndex, commaIndex - prevCommaIndex)};
                std::size_t i = 0;
                while (!std::isdigit(number[i]) && i < number.length()) {
                    ++i;
                }
                auto numberStr{number.substr(i, number.length() - i)};
                m_sizes.push_back(std::stoul(numberStr));
            };
            while (true) {
                std::size_t commaIndex = shapeTupleStr.find(',', prevCommaIndex);
                if (commaIndex == std::string::npos) {
                    push_number(static_cast<Size>(shapeTupleStr.length()), prevCommaIndex);
                    break;
                }
                push_number(static_cast<Size>(commaIndex), prevCommaIndex);
                prevCommaIndex = static_cast<Size>(commaIndex) + 1;
            }
            NP_THROW_UNLESS_WITH_ARG(!m_sizes.empty(), "Incorrect shape string format", shapeTupleStr);
        }

        //////////////////////////////////////////////////////////////
        /// \brief Conversion to string operator
        ///
        /// Represents a shape as a string with dimensions.
        ///
        /// \return A string with dimensions
        ///
        //////////////////////////////////////////////////////////////
        explicit operator std::string() const {
            if (m_sizes.empty()) {
                return "0,";
            } else if (m_sizes.size() == 1) {
                return std::to_string(m_sizes.at(0)) + ",";
            }

            std::string shape;
            for (std::size_t dim = 0; dim < m_sizes.size(); ++dim) {
                if (dim > 0)
                    shape += ", ";
                shape += std::to_string(m_sizes.at(dim));
            }
            return shape;
        }

        //////////////////////////////////////////////////////////////
        /// \brief Flatten the dimensions
        ///
        /// (2, 3, 4) -> (24,).
        ///
        //////////////////////////////////////////////////////////////
        void flatten() {
            if (m_sizes.empty()) {
                return;
            }
            auto product = std::accumulate(m_sizes.begin(), m_sizes.end(), static_cast<Size>(1), std::multiplies<>());
            m_sizes = {product};
        }

        //////////////////////////////////////////////////////////////
        /// \brief Reverses the dimensions
        ///
        /// (2, 3, 4) -> (4, 3, 2).
        ///
        //////////////////////////////////////////////////////////////
        void transpose() {
            std::reverse(m_sizes.begin(), m_sizes.end());
        }

        [[nodiscard]] bool empty() const {
            return m_sizes.empty();
        }

        void clear() {
            m_sizes.clear();
        }

        [[nodiscard]] std::size_t size() const {
            return m_sizes.size();
        }

        [[nodiscard]] Size calcSizeByShape() const {
            if (empty())
                return 0;

            return std::accumulate(m_sizes.cbegin(), m_sizes.cend(), static_cast<Size>(1), std::multiplies<>());
        }

        void singleElement() {
            m_sizes = {1};
        }

        void addDim(Size size) {
            m_sizes.push_back(size);
        }

        void expandDims(Size axis) {
            m_sizes.insert(std::next(m_sizes.cbegin(), axis), 1);
        }

        void removeFirstDim() {
            if (empty()) {
                throw std::runtime_error("Empty shape, cannot remove first dim");
            }
            m_sizes.erase(m_sizes.begin());
        }

        const Size &operator[](const std::size_t index) const {
            return m_sizes[index];
        }

        Size &operator[](const std::size_t index) {
            return m_sizes[index];
        }

        [[nodiscard]] Storage::iterator begin() {
            return m_sizes.begin();
        }

        [[nodiscard]] Storage::const_iterator cbegin() const {
            return m_sizes.cbegin();
        }

        [[nodiscard]] Storage::iterator end() {
            return m_sizes.end();
        }

        [[nodiscard]] Storage::const_iterator cend() const {
            return m_sizes.cend();
        }

        [[nodiscard]] Size &back() {
            return m_sizes.back();
        }

        [[nodiscard]] const Size &back() const {
            return m_sizes.back();
        }

        [[nodiscard]] Shape broadcast(const Shape &another) const {
            auto size1 = size();
            auto size2 = another.size();
            Shape shape{};
            if (size1 == 0 || size2 == 0) {
                return shape;
            }
            int32_t i1 = static_cast<int32_t>(size1) - 1;
            int32_t i2 = static_cast<int32_t>(size2) - 1;
            while (i1 >= 0 || i2 >= 0) {
                auto s1 = i1 < 0 ? 1 : m_sizes[i1];
                auto s2 = i2 < 0 ? 1 : another.m_sizes[i2];
                if (s1 != s2 && s1 != 1 && s2 != 1) {
                    throw std::runtime_error("Arrays cannot be broadcast together");
                }
                std::size_t out;
                if (s1 == s2 || s2 == 1) {
                    out = s1;
                } else {
                    out = s2;
                }
                shape.addDim(out);
                --i1;
                --i2;
            }
            shape.transpose();
            return shape;
        }

        inline friend bool operator==(const Shape &left, const Shape &right) {
            return left.m_sizes == right.m_sizes;
        }

        inline friend bool operator!=(const Shape &left, const Shape &right) {
            return !operator==(left, right);
        }

        inline friend std::ostream &operator<<(std::ostream &stream, const Shape &shape) {
            stream << "{";
            for (std::size_t index = 0; index < shape.m_sizes.size(); ++index) {
                if (index > 0)
                    stream << " ";
                stream << shape.m_sizes[index];
            }
            stream << "}";
            return stream;
        }

    private:
        Storage m_sizes;
    };
}// namespace np

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
    class Shape : public std::vector<Size> {
    public:
        inline Shape() = default;

        //////////////////////////////////////////////////////////////
        /// \brief Initializer constructor
        ///
        /// Initialize shape with a list of dimensions.
        ///
        /// \param initList list of dimensions
        ///
        //////////////////////////////////////////////////////////////
        inline Shape(std::initializer_list<Size> initList)
            : std::vector<Size>{initList} {
        }

        //////////////////////////////////////////////////////////////
        /// \brief Initializer constructor
        ///
        /// Initialize shape with a vector of dimensions.
        ///
        /// \param initList vector of dimensions
        ///
        //////////////////////////////////////////////////////////////
        inline explicit Shape(const std::vector<Size> &v)
            : std::vector<Size>{v} {
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
        inline explicit Shape(const std::string &shapeTupleStr) {
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
                push_back(std::stoul(numberStr));
            };
            while (true) {
                std::size_t commaIndex = shapeTupleStr.find(',', prevCommaIndex);
                if (commaIndex == std::string::npos) {
                    push_number(shapeTupleStr.length(), prevCommaIndex);
                    break;
                }
                push_number(commaIndex, prevCommaIndex);
                prevCommaIndex = commaIndex + 1;
            }
            NP_THROW_UNLESS_WITH_ARG(!empty(), "Incorrect shape string format", shapeTupleStr);
        }

        //////////////////////////////////////////////////////////////
        /// \brief Conversion to string operator
        ///
        /// Represents a shape as a string with dimensions.
        ///
        /// \return A string with dimensions
        ///
        //////////////////////////////////////////////////////////////
        inline explicit operator std::string() const {
            if (empty()) {
                return "0,";
            } else if (size() == 1) {
                return std::to_string(at(0)) + ",";
            }

            std::string shape;
            for (std::size_t dim = 0; dim < size(); ++dim) {
                if (dim > 0)
                    shape += ", ";
                shape += std::to_string(at(dim));
            }
            return shape;
        }

        //////////////////////////////////////////////////////////////
        /// \brief Flatten the dimensions
        ///
        /// (2, 3, 4) -> (24,).
        ///
        //////////////////////////////////////////////////////////////
        inline void flatten() {
            if (empty()) {
                return;
            }
            auto product = std::accumulate(begin(), end(), 1, std::multiplies<Size>());
            *this = {product};
        }

        //////////////////////////////////////////////////////////////
        /// \brief Reverses the dimensions
        ///
        /// (2, 3, 4) -> (4, 3, 2).
        ///
        //////////////////////////////////////////////////////////////
        inline void transpose() {
            std::reverse(begin(), end());
        }
    };
}// namespace np

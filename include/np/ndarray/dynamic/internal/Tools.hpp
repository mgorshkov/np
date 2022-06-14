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

#include <np/Shape.hpp>

namespace np {
    namespace ndarray {
        namespace array_dynamic {
            namespace internal {
                enum class Operator {
                    More,
                    MoreOrEqual,
                    Equal,
                    LessOrEqual,
                    Less,
                    None
                };
                using OperatorWithArg = std::pair<Operator, double>;

                inline static OperatorWithArg getOperatorWithArg(const std::string &cond) {
                    struct Pattern {
                        Operator op;
                        std::string str;
                    };
                    Pattern patterns[] = {
                            {Operator::MoreOrEqual, ">="},// important! first go longer patterns!!
                            {Operator::LessOrEqual, "<="},
                            {Operator::More, ">"},
                            {Operator::Equal, "="},
                            {Operator::Less, "<"}};
                    for (const auto &pattern: patterns) {
                        auto pos = cond.find(pattern.str);
                        if (pos != std::string::npos) {
                            auto arg = cond.substr(pos + std::size(pattern.str),
                                                   cond.length() - pos - std::size(pattern.str));
                            return {pattern.op, std::stod(arg)};
                        }
                    }
                    throw std::runtime_error("Invalid condition: " + cond);
                }
            }// namespace internal
        }    // namespace array_dynamic
    }        // namespace ndarray
}// namespace np

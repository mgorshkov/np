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

#include <cctype>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <np/ndarray/internal/AllToNumberConvertor.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            enum class Operator {
                None,
                More,
                MoreOrEqual,
                Equal,
                LessOrEqual,
                NotEqual,
                Less
            };

            template<typename DType>
            struct OperatorWithArg {
                Operator m_operator;
                DType m_arg;
            };

            using SubsettingIndexType = Size;

            using SlicingIndexType = std::tuple<Size, Size, SignedSize>;

            template<typename DType>
            using BooleanIndexType = OperatorWithArg<DType>;

            template<typename DType>
            using IndexType = std::variant<std::monostate, SubsettingIndexType, SlicingIndexType, BooleanIndexType<DType>>;

            template<typename DType>
            using IndicesType = std::vector<IndexType<DType>>;

            enum class IndexingMode {
                None,
                Subsetting,
                Slicing,
                BooleanIndexing,
                Size// this must be the last one
            };

            using IndexingChecker = std::function<bool(const std::string &cond)>;

            template<typename DType>
            using IndexingWorker = std::function<IndexType<DType>(Size index, const std::string &cond)>;

            template<typename DType>
            struct IndexingHandler {
                IndexingMode mode{IndexingMode::None};
                IndexingChecker checker;
                IndexingWorker<DType> worker;
            };

            struct Pattern {
                Operator op{Operator::None};
                const char *str{nullptr};
            };

            template<typename DType>
            inline OperatorWithArg<DType> getOperatorWithArg(const std::string &cond) {
                static const Pattern kPatterns[] = {
                        {Operator::MoreOrEqual, ">="},// important! first go longer patterns!!
                        {Operator::LessOrEqual, "<="},
                        {Operator::NotEqual, "!="},
                        {Operator::NotEqual, "<>"},
                        {Operator::More, ">"},
                        {Operator::Equal, "="},
                        {Operator::Less, "<"}};

                for (const auto &pattern: kPatterns) {
                    auto patternStartPos = cond.find(pattern.str);
                    if (patternStartPos != std::string::npos) {
                        const auto patternEndPos = patternStartPos + strlen(pattern.str);
                        auto argStartPos = patternEndPos;
                        while (!std::isalnum(cond[argStartPos]) && argStartPos < cond.length()) {
                            ++argStartPos;
                        }
                        auto arg = cond.substr(argStartPos, cond.length() - argStartPos);
                        return {pattern.op, AllToNumberConvertor<DType>()(arg)};
                    }
                }
                return {Operator::None, DType{}};
            }

            static inline bool isNone(const std::string &cond) {
                return cond.empty();
            }

            static inline bool isSubsetting(const std::string &cond) {
                return std::all_of(cond.begin(), cond.end(), [](const auto &c) {
                    return std::isdigit(c);
                });
            }

            static inline bool isSlicing(const std::string &cond) {
                return std::all_of(cond.begin(), cond.end(), [](const auto &c) {
                    return c == ':' || std::isdigit(c);
                });
            }

            template<typename DType>
            static inline bool isBooleanIndexing(const std::string &cond) {
                return internal::getOperatorWithArg<DType>(cond).m_operator != internal::Operator::None;
            }
        }// namespace internal
    }    // namespace ndarray
}// namespace np
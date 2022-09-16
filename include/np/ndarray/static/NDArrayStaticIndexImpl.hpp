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

#include <type_traits>

#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/static/NDArrayStaticDecl.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            // Subsetting
            template<typename DType, Size SizeT, Size... SizeTs>
            inline void
            set(NDArrayStatic<DType, SizeT, SizeTs...> &array, Size i,
                const typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType &data) {
                array.m_ArrayImpl.set(i, data.m_ArrayImpl);
            }

            // Select an element at an index
            // a[2]
            template<typename DType, Size SizeT, Size... SizeTs>
            inline typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::operator[](Size i) const {
                return ReducedType{m_ArrayImpl[i]};
            }

            // Subsetting
            // a[2] Select the element at the 2nd index
            // b[1,2] Select the element at row 1 column 2 (equivalent to b[1][2])
            // Boolean indexing
            // a[a < 2] Select elements from a less than 2
            // Slicing
            // a[0:2] Select items at index 0 and 1
            // b[0:2,1] Select items at rows 0 and 1 in column 1
            template<typename DType, Size SizeT, Size... SizeTs>
            inline auto NDArrayStatic<DType, SizeT, SizeTs...>::operator[](const std::string &cond) const {
                for (auto indexing: m_IndexingHandlers) {
                    if (indexing.checker(cond)) {
                        return indexing.worker(cond);
                    }
                }
                throw std::runtime_error("Invalid operator[] argument");
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::booleanIndexing(const std::string &cond) const {
                auto operatorWithArg = ndarray::internal::getOperatorWithArg<DType>(cond);
                auto pred = [&operatorWithArg, &cond](DType value) {
                    switch (operatorWithArg.first) {
                        case ndarray::internal::Operator::More:
                            return value > operatorWithArg.second;
                        case ndarray::internal::Operator::MoreOrEqual:
                            return value >= operatorWithArg.second;
                        case ndarray::internal::Operator::Equal:
                            return value == operatorWithArg.second;
                        case ndarray::internal::Operator::LessOrEqual:
                            return value <= operatorWithArg.second;
                        case ndarray::internal::Operator::NotEqual:
                            return value != operatorWithArg.second;
                        case ndarray::internal::Operator::Less:
                            return value < operatorWithArg.second;
                        default:
                            throw std::runtime_error("Invalid condition: " + cond);
                    }
                    return false;
                };
                std::vector<DType> result;
                std::copy_if(m_ArrayImpl.cbegin(),
                             m_ArrayImpl.cend(),
                             std::back_inserter(result),
                             pred);

                Shape shape{static_cast<Size>(result.size())};
                return NDArrayDynamic<DType>{result, shape};
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline NDArrayDynamic<DType> NDArrayStatic<DType, SizeT, SizeTs...>::slicing(const std::string &cond) const {
                std::size_t startPos = 0;
                auto commaPos = cond.find(',', 0);
                Size dim = 0;
                auto sh{shape()};
                auto sizeDim = size();
                std::vector<DType> result{};
                for (Size i = 0; i < size(); ++i) {
                    result.push_back(get(i));
                }
                do {
                    auto nextCommaPos = cond.find(',', commaPos + 1);
                    auto dimCond = cond.substr(startPos,
                                               nextCommaPos != std::string::npos ? nextCommaPos - startPos - 1 : cond.size() - startPos);
                    auto colonPos = dimCond.find(':');
                    if (colonPos == std::string::npos) {
                        auto index = std::stoi(cond.substr(commaPos + 1, cond.size() - commaPos - 1));
                        return static_cast<NDArrayDynamic<DType>>(operator[](index));
                    }
                    auto first = dimCond.substr(0, colonPos);
                    int firstIndex = 0;
                    try {
                        firstIndex = std::stoi(first);
                    } catch (std::invalid_argument const &ex) {
                    } catch (std::out_of_range const &ex) {
                    }
                    auto last = dimCond.substr(colonPos + 1, dimCond.size() - colonPos - 1);
                    int lastIndex = sh[0];
                    try {
                        lastIndex = std::stoi(last);
                    } catch (std::invalid_argument const &ex) {
                    } catch (std::out_of_range const &ex) {
                    }
                    auto length = lastIndex - firstIndex;
                    if (length > sh[dim]) {
                        throw std::runtime_error("Incorrect range");
                    }
                    sizeDim /= sh[dim];
                    sh[dim] = length;
                    std::vector<DType> resultNew{};
                    for (Size i = firstIndex * sizeDim; i < lastIndex * sizeDim; ++i) {
                        resultNew.push_back(result[i]);
                    }
                    result = resultNew;
                    startPos = commaPos + 1;
                    commaPos = nextCommaPos;
                    ++dim;
                } while (commaPos != std::string::npos);

                return NDArrayDynamic<DType>{result, sh};
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline typename NDArrayStatic<DType, SizeT, SizeTs...>::ReducedType
            NDArrayStatic<DType, SizeT, SizeTs...>::at(Size i) const {
                return ReducedType(m_ArrayImpl[i]);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline DType NDArrayStatic<DType, SizeT, SizeTs...>::get(std::size_t i) const {
                return m_ArrayImpl.get(i);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::set(std::size_t i, const DType &value) {
                return m_ArrayImpl.set(i, value);
            }
        }// namespace array_static
    }    // namespace ndarray
}// namespace np

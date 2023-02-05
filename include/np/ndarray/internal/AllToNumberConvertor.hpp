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

#include <np/Constants.hpp>
#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType>
            class AllToNumberConvertor;

            template<>
            class AllToNumberConvertor<std::string> {
            public:
                std::string operator()(const std::string &val) {
                    return val;
                }
            };

            template<>
            class AllToNumberConvertor<float_> {
            public:
                float_ operator()(const std::string &val) {
                    return std::stod(val);
                }
            };

            template<>
            class AllToNumberConvertor<short_> {
            public:
                short_ operator()(const std::string &val) {
                    return static_cast<short_>(std::stoi(val));
                }
            };

            template<>
            class AllToNumberConvertor<intc> {
            public:
                intc operator()(const std::string &val) {
                    return std::stoi(val);
                }
            };

            template<>
            class AllToNumberConvertor<int_> {
            public:
                int_ operator()(const std::string &val) {
                    return std::stol(val);
                }
            };

            template<>
            class AllToNumberConvertor<longlong> {
            public:
                longlong operator()(const std::string &val) {
                    return std::stoll(val);
                }
            };

            template<>
            class AllToNumberConvertor<bool_> {
            public:
                bool_ operator()(const std::string &val) {
                    return val == "1";
                }
            };
        }// namespace internal
    }    // namespace ndarray
}// namespace np
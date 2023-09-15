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

#include <cstddef>
#include <istream>
#include <ostream>
#include <string>

#include <np/Constants.hpp>
#include <np/Shape.hpp>

namespace np {
    namespace ndarray {
        namespace internal {

            constexpr std::size_t kMaxArrayDims = 10;

            template<typename Class>
            inline void dumpObject(std::ostream &stream, const Class &object) {
                stream.write(reinterpret_cast<const char *>(&object), sizeof(object));
            }

            inline void dumpObject(std::ostream &stream, const std::string &object) {
                for (uint8_t symbol: object) {
                    stream << symbol;
                }
            }

            inline void dumpObject(std::ostream &stream, const std::wstring &object) {
                for (uint32_t symbol: object) {
                    stream.write(reinterpret_cast<char *>(&symbol), sizeof(symbol));
                }
            }

            template<typename Class>
            inline Class readObject(std::istream &stream) {
                Class object{};
                stream.read(reinterpret_cast<char *>(&object), sizeof(object));
                return object;
            }

            inline std::string readStr(std::istream &stream, std::size_t size) {
                std::string object{};
                uint8_t value;
                for (std::size_t i = 0; i < size; ++i) {
                    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
                    object += static_cast<char>(value);
                }
                return object;
            }

            inline std::wstring readUnicode(std::istream &stream, std::size_t size) {
                std::wstring object{};
                uint32_t value;
                for (std::size_t i = 0; i < size; ++i) {
                    stream.read((char *) &value, sizeof(value));
                    object += static_cast<wchar_t>(value);
                }
                return object;
            }

            constexpr const char *kWhitespace = " \n\r\t\f\v";

            inline std::string ltrim(const std::string &s) {
                auto start = s.find_first_not_of(kWhitespace);
                return start == std::string::npos ? "" : s.substr(start);
            }

            inline std::string rtrim(const std::string &s) {
                auto end = s.find_last_not_of(kWhitespace);
                return end == std::string::npos ? "" : s.substr(0, end + 1);
            }

            inline std::string trim(const std::string &s) {
                return rtrim(ltrim(s));
            }

            template<typename Stream>
            class SquareBracketsInserter {
            public:
                explicit SquareBracketsInserter(Stream &stream) noexcept
                    : m_stream{stream} {
                    m_stream << "[";
                }

                ~SquareBracketsInserter() noexcept {
                    m_stream << "]";
                }

            private:
                Stream &m_stream;
            };
        }// namespace internal
    }    // namespace ndarray
}// namespace np
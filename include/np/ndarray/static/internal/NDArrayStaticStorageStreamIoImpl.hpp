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
#include <iomanip>
#include <iostream>

#include <np/ndarray/static/internal/NDArrayStaticStorage.hpp>

namespace np {
    namespace ndarray {
        namespace array_static {
            namespace internal {
                template<typename DType, Size SizeT>
                std::ostream &
                operator<<(std::ostream &stream, const NDArrayStaticStorage<DType, SizeT> &array) {
                    stream << "[";
                    if constexpr (SizeT > 100) {
                        for (Size index = 0; index < 2; ++index) {
                            if (index > 0)
                                stream << " ";
                            if constexpr (std::is_floating_point<DType>::value) {
                                stream << std::setprecision(8);
                            }
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                            stream << array.get(index);
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                        }
                        stream << std::endl
                               << "...";
                        for (Size index = SizeT - 3; index < SizeT; ++index) {
                            if (index > 0)
                                stream << " ";
                            if constexpr (std::is_floating_point<DType>::value) {
                                stream << std::setprecision(8);
                            }
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                            stream << array.get(index);
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                        }
                    } else {
                        for (Size index = 0; index < SizeT; ++index) {
                            if (index > 0)
                                stream << " ";
                            if constexpr (std::is_floating_point<DType>::value) {
                                stream << std::setprecision(8);
                            }
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                            stream << array.get(index);
                            if constexpr (std::is_same<DType, std::string>::value) {
                                stream << "\"";
                            }
                        }
                    }
                    stream << "]";
                    return stream;
                }

                template<Size SizeT>
                std::ostream &
                operator<<(std::ostream &stream, const NDArrayStaticStorage<std::wstring, SizeT> &array) {
                    stream << "[";
                    for (Size index = 0; index < SizeT; ++index) {
                        if (index > 0) {
                            stream << " ";
                        }
                        const auto &wstr = array.get(index);
                        std::string str(wstr.length(), 0);
                        std::transform(wstr.begin(), wstr.end(), str.begin(), [](wchar_t c) {
                            return static_cast<char>(c);
                        });
                        stream << "\"" << str << "\"";
                    }
                    stream << "]";
                    return stream;
                }

                template<Size SizeT>
                std::wostream &
                operator<<(std::wostream &stream, const NDArrayStaticStorage<std::wstring, SizeT> &array) {
                    stream << "[";
                    for (Size index = 0; index < SizeT; ++index) {
                        if (index > 0) {
                            stream << " ";
                        }
                        stream << "\"" << array.get(index) << "\"";
                    }
                    stream << "]";
                    return stream;
                }
            }// namespace internal
        }    // namespace array_static
    }        // namespace ndarray
}// namespace np
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

#include <np/ndarray/internal/Tools.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage>
            std::ostream &
            operator<<(std::ostream &stream, const NDArrayBase<DType, Derived, Storage> &array) {
                ndarray::internal::SquareBracketsInserter squareBracketsInserter(stream);

                if (!array.shape().empty()) {
                    if (array.shape().size() == 1) {
                        if (array.shape()[0] > 100) {
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
                            for (Size index = array.shape()[0] - 3; index < array.shape()[0]; ++index) {
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
                            for (Size index = 0; index < array.shape()[0]; ++index) {
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
                    } else {
                        if (array.shape()[0] > 100) {
                            for (Size index = 0; index < 2; ++index) {
                                if (index > 0) {
                                    stream << std::endl
                                           << " ";
                                }
                                if constexpr (Storage::kDepth < kMaxArrayDims) {
                                    stream << array[index];
                                } else {
                                    assert(false);
                                }
                            }
                            stream << std::endl
                                   << "...";
                            for (Size index = array.shape()[0] - 3; index < array.shape()[0]; ++index) {
                                if (index > 0) {
                                    stream << std::endl
                                           << " ";
                                }
                                if constexpr (Storage::kDepth < kMaxArrayDims) {
                                    stream << array[index];
                                } else {
                                    assert(false);
                                }
                            }
                        } else {
                            for (Size index = 0; index < array.shape()[0]; ++index) {
                                if (index > 0) {
                                    stream << std::endl
                                           << " ";
                                }
                                if constexpr (Storage::kDepth < kMaxArrayDims) {
                                    stream << array[index];
                                } else {
                                    assert(false);
                                }
                            }
                        }
                    }
                }
                return stream;
            }

            template<typename Derived, typename Storage>
            inline std::ostream &
            operator<<(std::ostream &stream, const NDArrayBase<std::wstring, Derived, Storage> &array) {
                ndarray::internal::SquareBracketsInserter squareBracketsInserter(stream);
                if (!array.shape().empty()) {
                    if (array.shape().size() == 1) {
                        for (Size index = 0; index < array.shape()[0]; ++index) {
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
                    } else {
                        for (Size index = 0; index < array.shape()[0]; ++index) {
                            if (index > 0) {
                                stream << std::endl
                                       << " ";
                            }
                            if constexpr (Storage::kDepth < kMaxArrayDims) {
                                stream << array[index];
                            } else {
                                assert(false);
                            }
                        }
                    }
                }
                return stream;
            }

            template<typename Derived, typename Storage>
            inline std::wostream &
            operator<<(std::wostream &stream, const NDArrayBase<std::wstring, Derived, Storage> &array) {
                ndarray::internal::SquareBracketsInserter squareBracketsInserter(stream);
                if (!array.shape().empty()) {
                    if (array.shape().size() == 1) {
                        for (Size index = 0; index < array.shape()[0]; ++index) {
                            if (index > 0) {
                                stream << " ";
                            }
                            stream << "\"" << array.get(index) << "\"";
                        }
                    } else {
                        for (Size index = 0; index < array.shape()[0]; ++index) {
                            if (index > 0) {
                                stream << std::endl
                                       << " ";
                            }
                            if constexpr (Storage::kDepth < kMaxArrayDims) {
                                stream << array[index];
                            } else {
                                assert(false);
                            }
                        }
                    }
                }
                return stream;
            }
        }// namespace internal
    }    // namespace ndarray
}// namespace np
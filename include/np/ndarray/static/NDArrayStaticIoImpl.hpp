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

#include <filesystem>
#include <fstream>
#include <string>

#include <np/Exception.hpp>
#include <np/ndarray/internal/Nep1.hpp>
#include <np/ndarray/static/NDArrayStaticDecl.hpp>

// For static arrays, only save is implemented
// They are loaded as dynamic arrays

namespace np {
    namespace ndarray {
        namespace array_static {
            using namespace np::ndarray::internal;
            using std::filesystem::path;

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::save(const char *filename) {
                path path = adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
                save(output);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::save(std::ostream &stream) {
                DTypeToDescrConvertor<DType> convertor{getMaxElementSize()};
                auto descr = convertor.DTypeToDescr();
                writeNep1Header(stream, descr, static_cast<std::string>(shape()));
                dumpToStreamAsBinary(stream, m_ArrayImpl);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::savez(const char *filename) {
                path path = adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
                save(output);
            }

            template<typename DType, Size SizeT, Size... SizeTs>
            inline void NDArrayStatic<DType, SizeT, SizeTs...>::savetxt(const char *filename, const char *delimiter) {
                path path = adjustNep1Path(filename);
                std::ofstream output(path);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
                save(output, delimiter);
            }
        }// namespace array_static
    }    // namespace ndarray
}// namespace np

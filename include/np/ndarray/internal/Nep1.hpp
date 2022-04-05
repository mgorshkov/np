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

#include <iostream>
#include <cstddef>
#include <cstring>
#include <filesystem>

#include <np/ndarray/internal/Tools.hpp>

namespace np::ndarray::internal {
    inline static std::filesystem::path adjustNep1Path(const char* filename) {
        std::filesystem::path path(filename);
        if (!path.has_extension()) {
            path.replace_extension(".npy");
        }
        return path;
    }

    inline static void writeNep1Header(std::ostream& stream, const std::string& dType, const std::string& shape) {
        // The first 6 bytes are a magic string: exactly “x93NUMPY”.
        static const char* magic = "\x93NUMPY";
        static const uint32_t magicLen = sizeof(magic) - 1;
        stream << magic;
        // The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
        uint8_t major = 1;
        // The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00.
        // Note: the version of the file format is not tied to the version of the numpy package.
        uint8_t minor = 0;
        stream << major << minor;
        // “descr” dtype.descr
        // An object that can be passed as an argument to the numpy.dtype() constructor to create the array’s dtype.
        // “fortran_order” bool
        // Whether the array data is Fortran-contiguous or not. Since Fortran-contiguous arrays are a common form of non-C-contiguity, we allow them to be written directly to disk for efficiency.
        // “shape” tuple of int
        // The shape of the array.
        std::string dTypeStr = "{\'descr\': \'" + dType + "\', ";

        dTypeStr += R"('fortran_order': False, 'shape': ()" + shape + "), }";
        // The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.
        uint16_t headerLen = 0x80 - magicLen - sizeof(major) - sizeof(minor) - sizeof(uint16_t) + 1; // version 2 format if needed
        stream.write((char*)&headerLen, sizeof(headerLen));
        // The next HEADER_LEN bytes form the header data describing the array’s format.
        // It is an ASCII string which contains a Python literal expression of a dictionary.
        // It is terminated by a newline (’n’) and padded with spaces (’x20’) to make the total length of the magic
        // string + 4 + HEADER_LEN be evenly divisible by 16 for alignment purposes.
        stream << dTypeStr;
        // padding
        uint32_t paddingLen = headerLen - dTypeStr.length() - 1;
        for (std::size_t i = 0; i < paddingLen; ++i) {
            stream << " ";
        }
        stream << std::endl;
    }

    inline static std::string parseDescr(const std::string& header) {
        // “descr” dtype.descr
        // An object that can be passed as an argument to the numpy.dtype() constructor to create the array’s dtype.
        // “fortran_order” bool
        // Whether the array data is Fortran-contiguous or not. Since Fortran-contiguous arrays are a common form of non-C-contiguity, we allow them to be written directly to disk for efficiency.
        static constexpr char descrPattern[] = "\'descr\': \'";
        std::size_t descrStart = header.find(descrPattern);
        if (descrStart == std::string::npos) {
            throw std::runtime_error("Array DType description is not found");
        }
        descrStart += sizeof(descrPattern) - 1;
        std::size_t descrEnd = header.find('\'', descrStart);
        if (descrEnd == std::string::npos) {
            throw std::runtime_error("Array DType description has incorrect format");
        }
        std::string descr = header.substr(descrStart, descrEnd - descrStart);
        return descr;
    }

    inline static Shape parseShape(const std::string& header) {
        // “shape” tuple of int
        // The shape of the array.

        static constexpr char shapePattern[] = "\'shape\': (";
        std::size_t shapeStart = header.find(shapePattern);
        if (shapeStart == std::string::npos) {
            throw std::runtime_error("Array shape is not found");
        }
        shapeStart += sizeof(shapePattern) - 1;
        std::size_t shapeEnd = header.find(')', shapeStart);
        if (shapeEnd == std::string::npos) {
            throw std::runtime_error("Array DType description has incorrect format");
        }
        std::string shapeStr{header.substr(shapeStart, shapeEnd - shapeStart)};
        return Shape{shapeStr};
    }

    inline static std::tuple<std::string, Shape> readNep1Header(std::istream& stream) {
        // The first 6 bytes are a magic string: exactly “x93NUMPY”.
        static const char magic[] = "\x93NUMPY";
        static constexpr uint32_t magicLen = std::size(magic) - 1;
        char magicRead[magicLen];
        stream.read(magicRead, magicLen);
        NP_THROW_UNLESS(std::memcmp(magic, magicRead, magicLen) == 0, "Invalid magic");
        // The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.
        static constexpr uint8_t major = 1;
        uint8_t majorRead;
        stream.read((char*)&majorRead, 1);
        if (majorRead != major) {
            throw std::runtime_error("Invalid major");
        }
        // The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00.
        // Note: the version of the file format is not tied to the version of the numpy package.
        static constexpr uint8_t minor = 0;
        uint8_t minorRead;
        stream.read((char*)&minorRead, 1);
        if (minorRead != minor) {
            throw std::runtime_error("Invalid minor");
        }

        uint16_t headerSize;
        stream.read((char*)&headerSize, sizeof(headerSize));
        std::string header;
        header.resize(headerSize);
        stream.read(header.data(), headerSize);

        auto descr = parseDescr(header);
        auto shape = parseShape(header);

        return std::make_tuple(descr, shape);
    }
}

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

#include <fstream>

#include <np/ndarray/dynamic/NDArrayDynamicDecl.hpp>

#include <np/ndarray/internal/Nep1.hpp>

/* Saving & Loading On Disk
 
 >>> np.save( , a)

 >>> np.savez( ', a, b)

 >>> np.load( )
 'my_Array'
 'Array.npz
 'my_Array.npy'
 >>> np.loadtxt( )

 >>> np.genfromtxt( , delimiter= )

 >>> np.savetxt( , a, delimiter= )
 "myfile.txt"
 "my_file.csv" ','
 "myArray.txt" " "
 */

namespace np::ndarray::array_dynamic {
    
    template<typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::save(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ofstream output(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
        save(output);
    }

    template <typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::save(std::ostream& stream) {
        DTypeToCharCodeConvertor<DType> convertor{};
        np::ndarray::internal::writeNep1Header(stream, convertor.DTypeToCharCode(), static_cast<std::string>(shape()));

        m_ArrayImpl.dumpToStreamAsBinary(stream);
    }

    template<typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::savez(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ofstream output(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
    }

    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::load(std::istream& stream) {
        std::string dType;
        Shape shape;
        std::tie(dType, shape) = np::ndarray::internal::readNep1Header(stream);

        std::size_t size = np::ndarray::internal::calcSizeByShape(shape);
        DTypeToCharCodeConvertor<DType> convertorByte{};
        NP_THROW_UNLESS_WITH_ARG(convertorByte.DTypeToCharCode() == dType, "Cannot open file for writing: ", dType);

        std::vector<DType> data;
        data.resize(size);
        stream.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(DType));
        internal::NDArrayDynamicInternal<DType> arrayInternal(data.begin(), shape);
        return NDArrayDynamic<DType>{arrayInternal};
    }

    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage>
            NDArrayDynamic<DType, Storage>::load(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ifstream input(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(input.is_open(), "Cannot open file for reading: ", filename);
        return load(input);
    }

    // Saving & Loading Text Files
    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::loadtxt(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ofstream output(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
    }

    template<typename DType, typename Storage>
    inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::genfromtxt(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ofstream output(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
    }

    template<typename DType, typename Storage>
    inline void NDArrayDynamic<DType, Storage>::savetxt(const char* filename) {
        std::filesystem::path path = np::ndarray::internal::adjustNep1Path(filename);
        std::ofstream output(path, std::ios::binary);
        NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
    }
} // namespace np::ndarray::array_dynamic


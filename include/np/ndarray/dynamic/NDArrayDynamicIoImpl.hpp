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

namespace np {
    namespace ndarray {
        namespace array_dynamic {

            template<typename DType, typename Storage>
            inline void NDArrayDynamic<DType, Storage>::save(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
                save(output);
            }

            template<typename DType, typename Storage>
            inline void NDArrayDynamic<DType, Storage>::save(std::ostream &stream) {
                ndarray::internal::DTypeToDescrConvertor<DType> convertor{getMaxElementSize()};
                ndarray::internal::writeNep1Header(stream, convertor.DTypeToDescr(),
                                                   static_cast<std::string>(shape()));
                m_ArrayImpl.dumpToStreamAsBinary(stream);
            }

            template<typename DType, typename Storage>
            inline void NDArrayDynamic<DType, Storage>::savez(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::load(std::istream &stream) {
                ndarray::internal::Descr descr;
                Shape shape;
                std::tie(descr, shape) = ndarray::internal::readNep1Header(stream);

                ndarray::internal::DTypeToDescrConvertor<DType> convertorByte{descr.size};
                NP_THROW_UNLESS_WITH_ARG(convertorByte.DTypeToChar() == descr.name, "Incorrect DType in input file: ",
                                         std::to_string(descr.name));
                std::size_t size = ndarray::internal::calcSizeByShape(shape);
                std::vector<DType> data{};
                for (std::size_t i = 0; i < size; ++i) {
                    DType element{};
                    if constexpr (std::is_same<DType, std::string>::value) {
                        element = ndarray::internal::readStr(stream, descr.size);
                    } else if constexpr (std::is_same<DType, std::wstring>::value) {
                        element = ndarray::internal::readUnicode(stream, descr.size);
                    } else {
                        element = ndarray::internal::readObject<DType>(stream);
                    }
                    data.push_back(element);
                }
                return NDArrayDynamic<DType>{std::move(data), std::move(shape)};
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage>
            NDArrayDynamic<DType, Storage>::load(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ifstream input(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(input.is_open(), "Cannot open file for reading: ", filename);
                return load(input);
            }

            // Saving & Loading Text Files
            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::loadtxt(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }

            template<typename DType, typename Storage>
            inline NDArrayDynamic<DType, Storage> NDArrayDynamic<DType, Storage>::genfromtxt(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }

            template<typename DType, typename Storage>
            inline void NDArrayDynamic<DType, Storage>::savetxt(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }
        }// namespace array_dynamic
    }// namespace ndarray
}// namespace np

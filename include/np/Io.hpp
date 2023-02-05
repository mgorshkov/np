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

#include <np/Constants.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

/* Saving & Loading On Disk
 >>> np.save('my_array' , a)
 >>> np.savez('array.npz', a, b)
 >>> np.load('my_array.npy')
 >>> np.loadtxt("myfile.txt")
 >>> np.genfromtxt("my_file.csv", delimiter=',')
 >>> np.savetxt("myArray.txt", a, delimiter=" ")
 */

namespace np {
    using ndarray::array_dynamic::NDArrayDynamic;
    //////////////////////////////////////////////////////////////
    /// \brief Save the array to a file
    ///
    /// Save an array to a binary file in NumPy .npy format.
    ///
    /// \param filename Filename to save the array to
    /// \param array Array to save
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline void save(const char *filename, const Array<DType, SizeT> &array) {
        array.save(filename);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Save the array to a file
    ///
    /// Save an array to a binary file in uncompressed .npz format.
    ///
    /// \warning This method is not currently implemented
    ///
    /// \param filename Filename to save the array to
    /// \param array Array to save
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline void savez(const char *filename, const Array<DType, SizeT> &array) {
        array.savez(filename);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Load array from a file
    ///
    /// Load an array from .npy format.
    ///
    /// \param filename Filename to load the array from
    ///
    /// \return Loaded dynamic array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    inline auto load(const char *filename) {
        NDArrayDynamic<DType> array{};
        return array.load(filename);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Load an array from a file
    ///
    /// Load an array from a text file.
    ///
    /// \warning This method is not currently implemented
    ///
    /// \param filename Filename to load the array from
    ///
    /// \return Loaded dynamic array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    inline auto loadtxt(const char *filename) {
        NDArrayDynamic<DType> array{};
        return array.loadtxt(filename);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Load an array from a file
    ///
    /// Load an array from a text file with a delimiter.
    ///
    /// \warning This method is not currently implemented
    ///
    /// \param filename Filename to load the array from
    /// \param delimiter String that delimits fields
    ///
    /// \return Loaded dynamic array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    inline Array<DType> genfromtxt(const char *filename, const char *delimiter) {
        return NDArrayDynamic<DType>::genfromtxt(filename, delimiter);
    }

    //////////////////////////////////////////////////////////////
    /// \brief Save the array to a file
    ///
    /// Save an array to a text file.
    ///
    /// \warning This method is not currently implemented
    ///
    /// \param filename Filename to save the array to
    /// \param array Array to save
    /// \param delimiter String that delimits fields
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size SizeT = SIZE_DEFAULT>
    inline void savetxt(const char *filename, const Array<DType, SizeT> &array, const char *delimiter) {
        //"Expected 1D or 2D array, got %dD array instead" % X.ndim)
        //ValueError: Expected 1D or 2D array, got 3D array instead
        array.savetxt(filename, delimiter);
    }
}// namespace np

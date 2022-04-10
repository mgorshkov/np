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

#include <np/DType.hpp>
#include <np/Constants.hpp>
#include <np/Shape.hpp>
#include <np/Exception.hpp>

#include <np/ndarray/dynamic/NDArrayDynamic.hpp>
#include <np/ndarray/static/NDArrayStatic.hpp>

namespace np {
    using ndarray::array_static::NDArrayStatic;
    using ndarray::array_dynamic::NDArrayDynamic;

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic array of zeros
    ///
    /// Create a dynamic array of zeros of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto zeros(Shape shape) {
        return NDArrayDynamic<DType>{shape, 0};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static array of zeros
    ///
    /// Create a static array of zeros of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimentions of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, Size SizeT, Size ...SizeTs>
    auto zeros() {
        return NDArrayStatic<DType, SizeT, SizeTs...>{ 0 };
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic array of ones
    ///
    /// Create a dynamic array of ones of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto ones(Shape shape) {
        return NDArrayDynamic<DType>{shape, 1};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static array of ones
    ///
    /// Create a static array of ones of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimensions of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, Size SizeT, Size ...SizeTs>
    auto ones() {
        return NDArrayStatic<DType, SizeT, SizeTs...>{ 1 };
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic range of values
    ///
    /// Create a dynamic 1D array with regularly incrementing values starting with zero.
    ///
    /// \param DType Type of array elements
    /// \param stop end value of the range (non-inclusive)
    ///
    /// \return A dynamic array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto arange(DType stop) {
        std::array<DType, stop> array;
        for (DType i = 0; i < stop; ++i) {
            array[i] = i;
        }
        return NDArrayDynamic<DType>{array};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static range of values
    ///
    /// Create a static 1D array with regularly incrementing values starting with zero.
    ///
    /// \param DType Type of array elements
    /// \param stop end value of the range (non-inclusive)
    ///
    /// \return A static array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, DType stop>
    auto arange() {
        std::array<DType, stop> array;
        for (DType i = 0; i < stop; ++i) {
            array[i] = i;
        }
        return NDArrayStatic<DType, stop>{array};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic range of values
    ///
    /// Create a dynamic 1D array of regularly incrementing values given the step.
    ///
    /// \param DType Type of array elements
    /// \param start end value of the range (inclusive)
    /// \param stop end value of the range (non-inclusive)
    /// \param step increment of the values (default 1)
    ///
    /// \return A dynamic array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto arange(DType start, DType stop, DType step) {
        NP_THROW_UNLESS(step != 0, "Step must not be zero.");

        DType const size = (stop - start) / step;
        std::array<DType, size> array;
        Size i{0};
        for (DType value = start; value < stop; value += step) {
            array[i++] = value;
        }
        return NDArrayDynamic<DType>{array};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static range of values
    ///
    /// Create a static 1D array with regularly incrementing values given the step.
    ///
    /// \param DType Type of array elements
    /// \param start end value of the range (inclusive)
    /// \param stop end value of the range (non-inclusive)
    /// \param step increment of the values (default 1)
    ///
    /// \return A static array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, DType start, DType stop, DType step = 1>
    auto arange() {
        NP_THROW_CONSTEXPR_UNLESS(step != 0, "Step must not be zero.");

        static DType const constexpr size = (stop - start) / step;
        NDArrayStatic<DType, size> array;
        Size i{0};
        for (DType value = start; value < stop; value += step) {
            set(array, i++, value);
        }
        return array;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic range of values
    ///
    /// Create a dynamic 1D array of regularly incrementing values given the number of samples.
    ///
    /// \param DType Type of array elements
    /// \param start end value of the range (inclusive)
    /// \param stop end value of the range (non-inclusive)
    /// \param num number of samples (default 50)
    ///
    /// \return A dynamic array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto linspace(DType start, DType stop, Size num = 50) {
        NP_THROW_UNLESS(num > 0, "Number of samples must be non-negative.");

        std::vector<DType> vector;
        vector.reserve(num);
        Size i{0};
        const DType delta = (stop - start) / (num - 1);
        for (DType value = start; value <= stop; value += delta) {
            vector.set(i++, value);
        }
        return NDArrayDynamic<DType>{vector};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static range of values
    ///
    /// Create a static 1D array of regularly incrementing values given the number of samples.
    ///
    /// \param DType Type of array elements
    /// \param start end value of the range (inclusive)
    /// \param stop end value of the range (non-inclusive)
    /// \param num number of samples (default 50)
    ///
    /// \return A static array of values
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault, Size num = 50>
    auto linspace(DType start, DType stop) {
        NP_THROW_CONSTEXPR_UNLESS(num > 0, "Number of samples must be non-negative.");

        NDArrayStatic<DType, num> array;
        Size i{0};
        const DType delta = (stop - start) / (num - 1);
        for (DType value = start; value <= stop; value += delta) {
            set(array, i++, value);
        }
        return array;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic array filled with a value
    ///
    /// Create a dynamic array filled with a fillValue.
    ///
    /// \param DType Type of array elements
    /// \param fillValue Value to fill the array
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto full(DType fillValue, Shape shape) {
        auto internal = ndarray::array_dynamic::internal::NDArrayDynamicInternal{shape, fillValue};
        return NDArrayDynamic<DType>{internal};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a static array filled with a value
    ///
    /// Create a static array filled with a fillValue.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimensions of the array
    /// \param fillValue Value to fill the array
    /// \param shape Shape of the array
    ///
    /// \return A static array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, Size SizeT, Size ...SizeTs>
    auto full(DType fillValue) {
        return NDArrayStatic<DType, SizeT, SizeTs...>{ fillValue };
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create an identity matrix
    ///
    /// Create an identity matrix.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimensions of the array
    ///
    /// \return A static array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto eye(Size size) {
        std::vector<std::vector<DType>> vector;
        vector.resize(size);
        for (Size i = 0; i < size; ++i) {
            vector[i].resize(size);
            for (Size j = 0; j < size; ++j) {
                vector[i][j] = i == j;
            }
        }
        return NDArrayDynamic<DType>{vector};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create an identity matrix
    ///
    /// Create an identity matrix.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimensions of the array
    ///
    /// \return A static array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, Size SizeT>
    auto eye() {
        std::array<std::array<DType, SizeT>, SizeT> array;
        for (Size i = 0; i < SizeT; ++i) {
            for (Size j = 0; j < SizeT; ++j) {
                array[i][j] = i == j;
            }
        }
        return NDArrayStatic<DType, SizeT, SizeT> {array};
    }

    namespace random {
        //////////////////////////////////////////////////////////////
        /// \brief Create a random dynamic array of values
        ///
        /// Create a random array with uniform distribution.
        ///
        /// \param DType Type of array elements
        /// \param shape shape of the array
        ///
        /// \return A dynamic array of random values
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType = DTypeDefault>
        auto rand(Shape shape) {
            std::random_device device;
            std::mt19937 generator(device());
            std::uniform_real_distribution<DType> distribution;
            std::vector<DType> vector;
            auto size = ndarray::internal::calcSizeByShape(shape);
            vector.resize(size);
            std::generate(vector.begin(), vector.end(), [&]{ return distribution(generator); });

            auto internal = ndarray::array_dynamic::internal::NDArrayDynamicInternal{vector, shape};
            return NDArrayDynamic<DType>{internal};
        }

        //////////////////////////////////////////////////////////////
        /// \brief Create a random dynamic array of values
        ///
        /// Create a random dynamic array with uniform distribution.
        ///
        /// \param DType Type of array elements
        /// \param size size of the array
        ///
        /// \return A dynamic array of random values
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType = DTypeDefault>
        auto rand(Size size) {
            Shape shape{size};
            return rand<DType>(shape);
        }

        template<typename DType, Size... SizeTs>
        struct rand_helper;

        template<typename DType, Size SizeT>
        struct rand_helper<DType, SizeT> {
            rand_helper() {
                std::random_device device;
                std::mt19937 generator(device());
                std::uniform_real_distribution<DType> distribution;
                std::vector<DType> vector;
                vector.resize(SizeT);
                std::generate(vector.begin(), vector.end(), [&]{ return distribution(generator); });

                m_Array = vector;
            }

            explicit operator NDArrayStatic<DType, SizeT>() {
                return static_cast<NDArrayStatic<DType, SizeT>>(m_Array);
            }

            NDArrayStatic<DType, SizeT> m_Array;
        };

        template<typename DType, Size SizeT, Size... SizeTs>
        struct rand_helper<DType, SizeT, SizeTs...> {
            rand_helper() {
                for (Size i = 0; i < SizeT; ++i) {
                    set(m_Array, i, static_cast<NDArrayStatic<DType, SizeTs...>>(rand_helper<DType, SizeTs...>()));
                }
            }

            explicit operator NDArrayStatic<DType, SizeT, SizeTs...>() {
                return m_Array;
            }

            NDArrayStatic<DType, SizeT, SizeTs...> m_Array;
        };

        //////////////////////////////////////////////////////////////
        /// \brief Create a random static array of values
        ///
        /// Create a random static array with uniform distribution.
        ///
        /// \param DType Type of array elements
        /// \param SizeT 1st dimension of the array
        /// \param SizeTs The rest dimensions of the array
        ///
        /// \return A static array of zeros
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType, Size SizeT, Size... SizeTs>
        auto rand() {
            return rand_helper<DType, SizeT, SizeTs...>{};
        }
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create an empty dynamic array of values
    ///
    /// Create an empty dynamic array.
    ///
    /// \param DType Type of array elements
    /// \param shape shape of the array
    ///
    /// \return An empty dynamic array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto empty(Shape shape) {
        return NDArrayDynamic<DType>{shape};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create an empty static array of values
    ///
    /// Create an empty static array.
    ///
    /// \param DType Type of array elements
    /// \param SizeT 1st dimension of the array
    /// \param SizeTs The rest dimensions of the array
    ///
    /// \return An empty static array
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, Size SizeT, Size... SizeTs>
    auto empty() {
        return NDArrayStatic<DType, SizeT, SizeTs...>{};
    }
}

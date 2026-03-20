/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <new>
#include <np/Constants.hpp>
#include <np/DType.hpp>
#include <np/Exception.hpp>
#include <np/Shape.hpp>
#include <random>

#ifdef USE_CUDA
#include <np/internal/cuda/Random.hpp>
#endif

#include <np/ndarray/constant/NDArrayConstant.hpp>
#include <np/ndarray/diagonal/NDArrayDiagonal.hpp>
#include <np/ndarray/diagonal/NDArrayIdentity.hpp>
#include <np/ndarray/dynamic/NDArrayDynamicCreatorsImpl.hpp>
#include <np/ndarray/static/NDArrayStaticCreatorsImpl.hpp>

namespace np {
    using ndarray::array_constant::NDArrayConstant;
    using ndarray::array_diagonal::NDArrayDiagonal;
    using ndarray::array_diagonal::NDArrayIdentity;
    using ndarray::array_dynamic::NDArrayDynamic;
    using ndarray::array_static::NDArrayStatic;

    //////////////////////////////////////////////////////////////
    /// \brief Create a dynamic array of zeros
    ///
    /// Create a constant array of zeros of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto zeros(Shape shape) {
        return NDArrayConstant<DType>{std::move(shape), 0};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a constant array of ones
    ///
    /// Create a constant array of ones of a given shape.
    ///
    /// \param DType Type of array elements
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType = DTypeDefault>
    auto ones(Shape shape) {
        return NDArrayConstant<DType>{std::move(shape), 1};
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
        std::vector<DType> vector;
        vector.resize(stop);
        std::iota(vector.begin(), vector.end(), 0);
        return NDArrayDynamic<DType>{vector};
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
        std::iota(array.begin(), array.end(), 0);
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
    auto arange(DType start, DType stop, DType step = 1) {
        NP_THROW_UNLESS(step != 0, "Step must not be zero.");

        Size size = static_cast<Size>((stop - start) / step);
        Shape shape{size};
        NDArrayDynamic<DType> result{shape};
        Size i{};
        if (step > 0) {
            for (DType value = start; value < stop; value += step) {
                result.set(i++, value);
            }
        } else {
            for (DType value = start; value > stop; value += step) {
                result.set(i++, value);
            }
        }
        return result;
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

        static Size const constexpr size = (stop - start) / step;
        NDArrayStatic<DType, size> array{};
        Size i{0};
        if constexpr (step > 0) {
            for (DType value = start; value < stop; value += step) {
                array.set(i++, value);
            }
        } else {
            for (DType value = start; value > stop; value += step) {
                array.set(i++, value);
            }
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
        vector.resize(num);
        const DType delta = (stop - start) / (static_cast<DType>(num) - 1);
        if (delta == 0) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid parameters, delta == 0");
        }
        size_t i = 0;
        for (DType value = start; value <= stop; value += delta) {
            vector[i++] = value;
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

        NDArrayStatic<DType, num> array{};
        const DType delta = (stop - start) / (num - 1);
        if (delta == 0) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "Invalid parameters, delta == 0");
        }
        Size i{0};
        for (DType value = start; value <= stop; value += delta) {
            array.set(i++, value);
        }
        return array;
    }

    //////////////////////////////////////////////////////////////
    /// \brief Create a constant array filled with a value
    ///
    /// Create a constant array filled with a fillValue.
    ///
    /// \param DType Type of array elements
    /// \param fillValue Value to fill the array
    /// \param shape Shape of the array
    ///
    /// \return A dynamic array of zeros
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType>
    auto full(const DType &fillValue, const Shape &shape) {
        return NDArrayConstant{shape, fillValue};
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
        return NDArrayIdentity<DType>{size};
    }

    namespace random {
        //////////////////////////////////////////////////////////////
        /// \brief Thread-local random generator for thread-safe random number generation.
        ///
        /// Each thread gets its own std::mt19937_64 instance to avoid:
        /// 1. Data races (std::mt19937_64 is not thread-safe)
        /// 2. Cache line contention from multiple threads writing to shared state
        /// 3. Performance degradation at small array sizes
        ///
        /// Uses mt19937_64 (64-bit) instead of mt19937 (32-bit) because:
        /// - Generating double-precision [0,1) values needs 53 bits of mantissa
        /// - mt19937_64 provides 64 bits per call vs 32 bits for mt19937
        /// - std::uniform_real_distribution<double> with mt19937 requires 2 calls
        /// - With mt19937_64, we use a fast inline conversion: bits >> 11 * 0x1.0p-53
        ///
        //////////////////////////////////////////////////////////////
        inline std::mt19937_64 &getGenerator() {
            static thread_local std::mt19937_64 generator{std::random_device{}()};
            return generator;
        }

        //////////////////////////////////////////////////////////////
        /// \brief Seeds the random data generator (current thread only)
        ///
        /// Seeds the random data generator for the calling thread.
        ///
        //////////////////////////////////////////////////////////////
        inline void seed(unsigned int sd) {
            getGenerator().seed(sd);
        }

        //////////////////////////////////////////////////////////////
        /// \brief Minimum array size to use CUDA for random generation.
        ///
        /// CPU OpenMP with mt19937_64 is faster than the CUDA path for sizes
        /// below this threshold due to:
        /// 1. cudaMalloc/cudaFree per-call overhead
        /// 2. cudaMemcpy round-trip latency
        /// 3. cudaDeviceSynchronize() blocking
        /// 4. CPU OpenMP already highly optimized for random generation
        ///
        /// Set to 500M elements (~4GB of doubles) where GPU bandwidth may
        /// start to compensate for the overhead.
        //////////////////////////////////////////////////////////////
        static constexpr Size kCudaRandMinSize = 500'000'000;

        // Fast conversion of a 64-bit random value to double in [0,1).
        // Uses the top 53 bits of the 64-bit value (the mantissa precision of double)
        // and multiplies by 2^-53. This is significantly faster than
        // std::uniform_real_distribution<double> which has branching and
        // rejection sampling overhead.
        //
        // Result is in [0, 1) with uniform distribution across all 2^53 possible
        // double values in that range (same as numpy's uniform distribution).
        inline double fast_rand_double(std::uint64_t bits) {
            return (bits >> 11) * 0x1.0p-53;
        }

        // Fast conversion of a 64-bit random value to float in [0,1).
        // Uses the top 24 bits (the mantissa precision of float).
        inline float fast_rand_float(std::uint64_t bits) {
            return (bits >> 40) * 0x1.0p-24f;
        }

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
        auto rand(const Shape &shape, DType minValue = 0.0, DType maxValue = 1.0) {
            auto size = shape.calcSizeByShape();
            auto *data = new DType[size];

#ifdef USE_CUDA
            // Use GPU for very large arrays only (see kCudaRandMinSize)
            if (size > kCudaRandMinSize) {
                // Generate a random seed from the random device
                unsigned long long seed = std::random_device{}();
                internal::cuda::randUniform(data, size, minValue, maxValue, seed);
            } else
#endif
            {
                // Each thread uses its own thread_local generator to avoid:
                // 1. Data races (std::mt19937_64 is not thread-safe)
                // 2. Cache line contention from shared mutable state
                // 3. Performance degradation at small array sizes
#ifdef USE_OPENMP
#pragma omp parallel default(none) shared(data, size, minValue, maxValue)
#endif
                {
                    auto &local_gen = getGenerator();
                    if constexpr (std::is_same_v<DType, double>) {
                        // Fast path for double: use inline bit conversion
                        // Avoids std::uniform_real_distribution overhead (branching, rejection sampling)
                        const double scale = (maxValue - minValue);
                        // index variable in OpenMP 'for' statement must have signed integral type
#ifdef USE_OPENMP
#pragma omp for
#endif
                        for (std::int32_t offset = 0; offset < static_cast<std::int32_t>(size); ++offset) {
                            data[offset] = minValue + fast_rand_double(local_gen()) * scale;
                        }
                    } else if constexpr (std::is_same_v<DType, float>) {
                        // Fast path for float: use inline bit conversion
                        const float scale = static_cast<float>(maxValue - minValue);
#ifdef USE_OPENMP
#pragma omp for
#endif
                        for (std::int32_t offset = 0; offset < static_cast<std::int32_t>(size); ++offset) {
                            data[offset] = static_cast<float>(minValue) + fast_rand_float(local_gen()) * scale;
                        }
                    } else {
                        // Generic path for other types (e.g., int)
                        std::uniform_real_distribution<DType> distribution(minValue, maxValue);
#ifdef USE_OPENMP
#pragma omp for
#endif
                        for (std::int32_t offset = 0; offset < static_cast<std::int32_t>(size); ++offset) {
                            data[offset] = distribution(local_gen);
                        }
                    }
                }
            }

            return NDArrayDynamic<DType>{data, shape};
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
            const Shape shape{size};
            return rand<DType>(shape);
        }

        template<typename DType, Size SizeT, Size... Sizes>
        struct rand_helper {
            static Size constexpr m_size = (SizeT * ... * Sizes);

            rand_helper() {
                auto &local_gen = getGenerator();
                std::vector<DType> vector;
                vector.resize(m_size);
                if constexpr (std::is_same_v<DType, double>) {
                    std::generate(vector.begin(), vector.end(), [&] { return fast_rand_double(local_gen()); });
                } else if constexpr (std::is_same_v<DType, float>) {
                    std::generate(vector.begin(), vector.end(), [&] { return fast_rand_float(local_gen()); });
                } else {
                    std::uniform_real_distribution<DType> distribution;
                    std::generate(vector.begin(), vector.end(), [&] { return distribution(local_gen); });
                }
                const Shape shape{SizeT, Sizes...};
                m_array = NDArrayStatic<DType, m_size>(vector, shape);
            }

            explicit operator NDArrayStatic<DType, m_size>() {
                return m_array;
            }

            NDArrayStatic<DType, m_size> m_array;
        };

        //////////////////////////////////////////////////////////////
        /// \brief Create a random static array of values
        ///
        /// Create a random static array with uniform distribution.
        ///
        /// \param DType Type of array elements
        /// \param SizeT First dim of the array
        /// \param Sizes Rest dims of the array
        ///
        /// \return A static array of zeros
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType, Size SizeT, Size... Sizes>
        NDArrayStatic<DType, (SizeT * ... * Sizes)> rand() {
            return static_cast<NDArrayStatic<DType, (SizeT * ... * Sizes)>>(rand_helper<DType, SizeT, Sizes...>());
        }

        //////////////////////////////////////////////////////////////
        /// \brief Create a random dynamic array of values
        ///
        /// Return a sample (or samples) from the “standard normal” distribution.
        ///
        /// \param DType Type of array elements
        /// \param shape shape of the array
        ///
        /// \return A dynamic array of random values
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType = DTypeDefault>
        auto randn(const Shape &shape) {
            auto size = shape.calcSizeByShape();
            auto *data = new DType[size];
#ifdef USE_OPENMP
#pragma omp parallel default(none) shared(data, size)
#endif
            {
                auto &local_gen = getGenerator();
                std::normal_distribution<DType> distribution;
#ifdef USE_OPENMP
#pragma omp for
#endif
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t offset = 0; offset < static_cast<std::int32_t>(size); ++offset) {
                    data[offset] = distribution(local_gen);
                }
            }

            return NDArrayDynamic<DType>{data, shape};
        }

        //////////////////////////////////////////////////////////////
        /// \brief Create a random dynamic array of values
        ///
        /// Return a sample (or samples) from the “standard normal” distribution.
        ///
        /// \param DType Type of array elements
        /// \param size size of the array
        ///
        /// \return A dynamic array of random values
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType = DTypeDefault>
        auto randn(Size size) {
            const Shape shape{size};
            return randn<DType>(shape);
        }

        template<typename DType, Size SizeT, Size... Sizes>
        struct randn_helper {
            static Size constexpr m_size = (SizeT * ... * Sizes);

            randn_helper() {
                std::normal_distribution<DType> distribution;
                auto &local_gen = getGenerator();
                std::vector<DType> vector;
                vector.resize(m_size);
                std::generate(vector.begin(), vector.end(), [&] { return distribution(local_gen); });
                const Shape shape{SizeT, Sizes...};
                m_array = NDArrayStatic<DType, m_size>(vector, shape);
            }

            explicit operator NDArrayStatic<DType, m_size>() {
                return m_array;
            }

            NDArrayStatic<DType, m_size> m_array;
        };

        //////////////////////////////////////////////////////////////
        /// \brief Create a random static array of values
        ///
        /// Create a random static array with uniform distribution.
        ///
        /// \param DType Type of array elements
        /// \param SizeT First dim of the array
        /// \param Sizes Rest dims of the array
        ///
        /// \return A static array of zeros
        ///
        //////////////////////////////////////////////////////////////
        template<typename DType, Size SizeT, Size... Sizes>
        NDArrayStatic<DType, (SizeT * ... * Sizes)> randn() {
            return static_cast<NDArrayStatic<DType, (SizeT * ... * Sizes)>>(randn_helper<DType, SizeT, Sizes...>());
        }
    }// namespace random

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
        return NDArrayConstant<DType>{std::move(shape), 0};
    }

    //////////////////////////////////////////////////////////////
    /// \brief Extract a diagonal or construct a diagonal array.
    ///
    /// Return the extracted diagonal or constructed diagonal array.
    ///
    /// \param DType Type of array elements
    /// \param v if v is a 2-D array, return a copy of its k-th diagonal. If v is a 1-D array, return a 2-D array with v on the k-th diagonal.
    /// \param k Diagonal in question. The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
    ///
    /// \return the extracted diagonal or constructed diagonal array.
    ///
    //////////////////////////////////////////////////////////////
    template<typename DType, typename Derived, typename Storage>
    auto diag0(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &v, int k = 0) {
        if (!v.empty()) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "diag0 supports empty arrays");
        }
        return NDArrayDiagonal<DType, Derived, Storage, 0>(v, k);
    }

    template<typename DType, typename Derived, typename Storage>
    auto diag1(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &v, int k = 0) {
        if (v.ndim() != 1) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "diag1 supports 1D arrays");
        }
        return NDArrayDiagonal<DType, Derived, Storage, 1>(v, k);
    }

    template<typename DType, typename Derived, typename Storage>
    auto diag2(const ndarray::internal::NDArrayBase<DType, Derived, Storage> &v, int k = 0) {
        if (v.ndim() != 2) {
            NP_THROW_WITH_STACKTRACE(std::invalid_argument, "diag2 supports 2D arrays");
        }
        return NDArrayDiagonal<DType, Derived, Storage, 2>(v, k);
    }
}// namespace np

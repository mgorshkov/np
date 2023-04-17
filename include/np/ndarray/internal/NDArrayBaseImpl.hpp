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
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <unordered_set>
#include <vector>

#include <np/Axis.hpp>
#include <np/Shape.hpp>
#include <np/ndarray/internal/Agg.hpp>
#include <np/ndarray/internal/Index.hpp>
#include <np/ndarray/internal/Indexing.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>
#include <np/ndarray/internal/Nep1.hpp>

namespace np {
    namespace ndarray {
        namespace internal {
            template<typename DType, typename Derived, typename Storage>
            template<typename... Args>
            NDArrayBase<DType, Derived, Storage>::NDArrayBase(Args &&...args)
                : m_storage(std::forward<Args>(args)...) {
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayBase<DType, Derived, Storage>::NDArrayBase(const NDArrayBase &another)
                : m_storage{another.m_storage} {
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayBase<DType, Derived, Storage>::NDArrayBase(NDArrayBase &&another) noexcept
                : m_storage{std::move(another.m_storage)} {
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayBase<DType, Derived, Storage> &NDArrayBase<DType, Derived, Storage>::operator=(const NDArrayBase &another) {
                if (this != &another) {
                    m_storage = another.m_storage;
                }
                return *this;
            }

            template<typename DType, typename Derived, typename Storage>
            NDArrayBase<DType, Derived, Storage> &NDArrayBase<DType, Derived, Storage>::operator=(NDArrayBase &&another) noexcept {
                if (this != &another) {
                    m_storage = std::move(another.m_storage);
                }
                return *this;
            }

            template<typename DType, typename Derived, typename Storage>
            constexpr DType NDArrayBase<DType, Derived, Storage>::dtype() {
                return DType{};
            }

            // Save and load
            template<typename DType, typename Derived, typename Storage>
            void NDArrayBase<DType, Derived, Storage>::save(std::ostream &stream) {
                ndarray::internal::DTypeToDescrConvertor<DType> convertor{NDArrayBase::getMaxElementSize()};
                ndarray::internal::writeNep1Header(stream, convertor.DTypeToDescr(),
                                                   static_cast<std::string>(shape()));
                NDArrayBase::dumpToStreamAsBinary(stream);
            }

            template<typename DType, typename Derived, typename Storage>
            void NDArrayBase<DType, Derived, Storage>::save(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
                save(output);
            }

            template<typename DType, typename Derived, typename Storage>
            void NDArrayBase<DType, Derived, Storage>::savez(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::load(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ifstream input(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(input.is_open(), "Cannot open file for reading: ", filename);
                return load(input);
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::loadtxt(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ifstream input(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(input.is_open(), "Cannot open file for reading: ", filename);
                return load(input);
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::genfromtxt(const char *filename) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ifstream input(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(input.is_open(), "Cannot open file for reading: ", filename);
                return load(input);
            }

            template<typename DType, typename Derived, typename Storage>
            void NDArrayBase<DType, Derived, Storage>::savetxt(const char *filename, const char *) {
                std::filesystem::path path = ndarray::internal::adjustNep1Path(filename);
                std::ofstream output(path, std::ios::binary);
                NP_THROW_UNLESS_WITH_ARG(output.is_open(), "Cannot open file for writing: ", filename);
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator+(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                return add(array);
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::add(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape().broadcast(array.shape())};
                auto size1 = size();
                auto size2 = array.size();
                auto maxSize = std::max(size1, size2);
#pragma omp parallel for default(none) shared(array, maxSize, size1, size2, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(maxSize); ++i) {
                    DType plusResult;
                    ndarray::internal::add(get(i % size1), array.get(i % size2), plusResult);
                    result.set(i, plusResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator-(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                return subtract(array);
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::subtract(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape().broadcast(array.shape())};
                auto size1 = size();
                auto size2 = array.size();
                auto maxSize = std::max(size1, size2);
#pragma omp parallel for default(none) shared(array, maxSize, size1, size2, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(maxSize); ++i) {
                    DType minusResult;
                    ndarray::internal::subtract(get(i % size1), array.get(i % size2), minusResult);
                    result.set(i, minusResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator*(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                return multiply(array);
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::multiply(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape().broadcast(array.shape())};
                auto size1 = size();
                auto size2 = array.size();
                auto maxSize = std::max(size1, size2);
#pragma omp parallel for default(none) shared(array, maxSize, size1, size2, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(maxSize); ++i) {
                    DType multiplyResult;
                    ndarray::internal::multiply(get(i % size1), array.get(i % size2), multiplyResult);
                    result.set(i, multiplyResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator/(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                return divide(array);
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::divide(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape().broadcast(array.shape())};
                auto size1 = size();
                auto size2 = array.size();
                auto maxSize = std::max(size1, size2);
#pragma omp parallel for default(none) shared(array, maxSize, size1, size2, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(maxSize); ++i) {
                    DType divideResult;
                    ndarray::internal::divide(get(i % size1), array.get(i % size2), divideResult);
                    result.set(i, divideResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::exp() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType expResult{};
                    ndarray::internal::exp(get(i), expResult);
                    result.set(i, expResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::sqrt() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType sqrtResult{};
                    ndarray::internal::sqrt(get(i), sqrtResult);
                    result.set(i, sqrtResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::sin() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType sinResult{};
                    ndarray::internal::sin(get(i), sinResult);
                    result->set(i, sinResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::cos() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType cosResult{};
                    ndarray::internal::cos(get(i), cosResult);
                    result->set(i, cosResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::log() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType logResult{};
                    ndarray::internal::log(get(i), logResult);
                    result.set(i, logResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::abs() const {
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shape()};
#pragma omp parallel for default(none) shared(result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType absResult{};
                    ndarray::internal::abs(get(i), absResult);
                    result.set(i, absResult);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            DType NDArrayBase<DType, Derived, Storage>::dot(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                if (shape().size() != 1 || array.shape().size() != 1 || shape() != array.shape()) {
                    throw std::runtime_error("Shapes are different or arguments are not 1D arrays");
                }
                DType result{0};
#pragma omp parallel for default(none) shared(array) reduction(+ \
                                                               : result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType multipleResult{};
                    ndarray::internal::multiply(get(i), array.get(i), multipleResult);
                    result += multipleResult;
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator==(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                array_dynamic::NDArrayDynamicBool result{shape()};
#pragma omp parallel for default(none) shared(array, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    const auto equals = get(i) == array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator<(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                array_dynamic::NDArrayDynamicBool result{shape()};
#pragma omp parallel for default(none) shared(array, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    auto equals = get(i) < array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::operator>(const NDArrayBase<DType, Derived2, Storage2> &array) const {
                array_dynamic::NDArrayDynamicBool result{shape()};
#pragma omp parallel for default(none) shared(array, result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    auto equals = get(i) > array.get(i);
                    result.set(i, equals);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            DType NDArrayBase<DType, Derived, Storage>::sum() const {
                DType result{};
#pragma omp parallel for default(none) reduction(+ \
                                                 : result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    const auto &element = get(i);
                    result += element;
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            DType NDArrayBase<DType, Derived, Storage>::nansum() const {
                DType result{};
#pragma omp parallel for default(none) reduction(+ \
                                                 : result)
                // index variable in OpenMP 'for' statement must have signed integral type
                for (std::int32_t i = 0; i < static_cast<std::int32_t>(size()); ++i) {
                    DType nanToZeroResult{};
                    ndarray::internal::nanToZero(get(i), nanToZeroResult);
                    result += nanToZeroResult;
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            DType NDArrayBase<DType, Derived, Storage>::min() const {
                DType result{};
                bool inited{false};
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    if (!inited || element < result) {
                        result = element;
                        inited = true;
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            DType NDArrayBase<DType, Derived, Storage>::max() const {
                DType result{};
                bool inited{false};
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    if (!inited || element > result) {
                        result = element;
                        inited = true;
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::cumsum() const {
                Shape sh = shape();
                sh.flatten();
                ndarray::array_dynamic::NDArrayDynamic<DType> result{sh};
                DType sum = 0;
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    ndarray::internal::add(sum, element, sum);
                    result.set(i, sum);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::nancumsum() const {
                Shape sh = shape();
                sh.flatten();
                ndarray::array_dynamic::NDArrayDynamic<DType> result{sh};
                DType sum = 0;
                for (Size i = 0; i < size(); ++i) {
                    DType nanToZeroResult{};
                    ndarray::internal::nanToZero(get(i), nanToZeroResult);
                    ndarray::internal::add(sum, nanToZeroResult, sum);

                    result.set(i, sum);
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::mean() const {
                auto s = size();
                if (s == 0)
                    return 0;
                float_ result{};
                for (Size i = 0; i < s; ++i) {
                    const auto &element = get(i);
                    ndarray::internal::inc(result, element);
                }
                float_ resultDiv{};
                ndarray::internal::divideBySize(result, s, resultDiv);
                return resultDiv;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::nanmean() const {
                auto s = size();
                if (s == 0)
                    return 0;
                float_ result{};
                Size count = 0;
                for (Size i = 0; i < s; ++i) {
                    const auto &element = get(i);
                    bool isNaNResult{};
                    ndarray::internal::isNaN(element, isNaNResult);
                    if (isNaNResult)
                        continue;
                    ndarray::internal::inc(result, element);
                    ++count;
                }
                float_ resultDiv{};
                ndarray::internal::divideBySize(result, count, resultDiv);
                return resultDiv;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::median() const {
                auto s = size();
                if (s == 0)
                    return 0;
                ndarray::array_dynamic::NDArrayDynamic<float_> array{shape()};
                for (Size i = 0; i < s; ++i) {
                    array.set(i, get(i));
                }
                auto middle1 = array.getStorage().begin();
                std::advance(middle1, s / 2);

                const auto begin = array.getStorage().begin();
                const auto end = array.getStorage().end();

                if (s % 2 == 0) {
                    auto middle2 = array.getStorage().begin();
                    std::advance(middle2, (s - 1) / 2);

                    std::nth_element(begin,
                                     middle1,
                                     end);

                    std::nth_element(begin,
                                     middle2,
                                     end);

                    // Find the average of values at indices size / 2 and (size - 1) / 2
                    float_ addResult;
                    ndarray::internal::add(array.get((s - 1) / 2), array.get(s / 2), addResult);
                    float_ result;
                    ndarray::internal::divideByDouble(addResult, 2.0, result);
                    return result;
                }
                std::nth_element(begin,
                                 middle1,
                                 end);
                return array.get(s / 2);
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::nanmedian() const {
                auto s = size();
                if (s == 0)
                    return 0;
                ndarray::array_dynamic::NDArrayDynamic<float_> array{};
                for (Size i = 0; i < s; ++i) {
                    bool isNaNResult{};
                    ndarray::internal::isNaN(get(i), isNaNResult);
                    if (!isNaNResult) {
                        array.push_back(get(i));
                    }
                }
                Size count = array.size();
                auto middle1 = array.getStorage().begin();
                std::advance(middle1, count / 2);

                auto begin = array.getStorage().begin();
                auto end = begin;
                std::advance(end, count);

                if (s % 2 == 0) {
                    auto middle2 = array.getStorage().begin();
                    std::advance(middle2, (count - 1) / 2);

                    std::nth_element(begin,
                                     middle1,
                                     end);

                    std::nth_element(begin,
                                     middle2,
                                     end);

                    // Find the average of values at indices size / 2 and (size - 1) / 2
                    float_ addResult;
                    ndarray::internal::add(array.get((count - 1) / 2), array.get(count / 2), addResult);
                    float_ result;
                    ndarray::internal::divideByDouble(addResult, 2.0, result);
                    return result;
                }
                std::nth_element(begin,
                                 middle1,
                                 end);
                return array.get(count / 2);
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::cov() const {
                auto sh = shape();
                if (sh.size() != 1 && sh.size() != 2)
                    throw std::runtime_error("Only 1D and 2D arrays supported");

                if (sh.size() == 1) {
                    float_ res;
                    np::ndarray::internal::setDouble(1.0, res);
                    return ndarray::array_dynamic::NDArrayDynamic<float_>{res};
                }

                Shape resultShape({len(), len()});
                ndarray::array_dynamic::NDArrayDynamic<float_> result{resultShape};
                for (Size i = 0; i < len(); ++i) {
                    for (Size j = 0; j < len(); ++j) {
                        auto subArray1 = (*this)[i];
                        auto subArray2 = (*this)[j];

                        float_ c = vectorCov(subArray1, subArray2);
                        result.set(i + j * len(), c);
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::corrcoef() const {
                auto sh = shape();
                auto s = sh.size();
                if (s != 1 && s != 2)
                    throw std::runtime_error("Only 1D and 2D arrays supported");

                if (s == 1) {
                    float_ res{};
                    ndarray::internal::setDouble(1, res);
                    return ndarray::array_dynamic::NDArrayDynamic<float_>{res};
                }

                Shape resultShape({len(), len()});
                ndarray::array_dynamic::NDArrayDynamic<float_> result{resultShape};
                for (Size i = 0; i < len(); ++i) {
                    for (Size j = 0; j < len(); ++j) {
                        auto subArray1 = (*this)[i];
                        auto subArray2 = (*this)[j];

                        float_ c = vectorCorr(subArray1, subArray2);
                        result.set(i + j * len(), c);
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::std_() const {
                ndarray::array_dynamic::NDArrayDynamic<float_> x{shape()};
                float_ m = mean();
                for (Size i = 0; i < size(); ++i) {
                    float_ result;
                    ndarray::internal::subtract<float_>(get(i), m, result);
                    float_ resultAbs;
                    ndarray::internal::abs(result, resultAbs);
                    auto a = resultAbs;
                    float_ resultMul;
                    ndarray::internal::multiply(a, a, resultMul);
                    x.set(i, resultMul);
                }
                float_ resultSqrt;
                ndarray::internal::sqrt(x.mean(), resultSqrt);
                return resultSqrt;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::nanstd() const {
                ndarray::array_dynamic::NDArrayDynamic<float_> x;
                float_ m = nanmean();
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    bool isNaNResult{};
                    ndarray::internal::isNaN(element, isNaNResult);
                    if (isNaNResult)
                        continue;
                    float_ result;
                    ndarray::internal::subtract<float_>(element, m, result);
                    float_ resultAbs;
                    ndarray::internal::abs(result, resultAbs);
                    auto a = resultAbs;
                    float_ resultMul;
                    ndarray::internal::multiply(a, a, resultMul);
                    x.push_back(resultMul);
                }
                float_ resultSqrt;
                ndarray::internal::sqrt(x.nanmean(), resultSqrt);
                return resultSqrt;
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::var() const {
                ndarray::array_dynamic::NDArrayDynamic<float_> x{shape()};
                float_ m = mean();
                for (Size i = 0; i < size(); ++i) {
                    float_ result;
                    ndarray::internal::subtract<float_>(get(i), m, result);
                    float_ resultAbs;
                    ndarray::internal::abs(result, resultAbs);
                    auto a = resultAbs;
                    float_ resultMul;
                    ndarray::internal::multiply(a, a, resultMul);
                    x.set(i, resultMul);
                }
                return x.mean();
            }

            template<typename DType, typename Derived, typename Storage>
            float_ NDArrayBase<DType, Derived, Storage>::nanvar() const {
                ndarray::array_dynamic::NDArrayDynamic<float_> x;
                float_ m = nanmean();
                for (Size i = 0; i < size(); ++i) {
                    const auto &element = get(i);
                    bool isNaNResult{};
                    ndarray::internal::isNaN(element, isNaNResult);
                    if (isNaNResult)
                        continue;
                    float_ result;
                    ndarray::internal::subtract<float_>(element, m, result);
                    float_ resultAbs;
                    ndarray::internal::abs(result, resultAbs);
                    auto a = resultAbs;
                    float_ resultMul;
                    ndarray::internal::multiply(a, a, resultMul);
                    x.push_back(resultMul);
                }
                return x.nanmean();
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::view() const {
                return copy();
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::copy() const {
                array_dynamic::NDArrayDynamic<DType> result{shape()};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i));
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            void NDArrayBase<DType, Derived, Storage>::sort() {
                std::sort(m_storage.begin(), m_storage.end());
                auto sh = shape();
                sh.flatten();
                setShape(sh);
            }

            // template<std::size_t N = -1>
            // inline void sort(std::optional<Axis<N>> axis=std::optional<Axis<N>>{});

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::transpose() const {
                auto sh = shape();
                if (sh.empty()) {
                    return ndarray::array_dynamic::NDArrayDynamic<DType>{};
                }
                //auto s = sh.size();
                //if (s != 1 && s != 2)
                //    throw std::runtime_error("Only 1D and 2D arrays are currently supported");
                auto shapeNew{sh};
                shapeNew.transpose();
                ndarray::array_dynamic::NDArrayDynamic<DType> result{shapeNew};
                if (sh.size() == 1) {
                    for (Size i = 0; i < size(); ++i) {
                        result.set(i, this->get(i));
                    }
                } else {
                    Size dim1 = sh[0];
                    std::vector<ndarray::array_dynamic::NDArrayDynamic<DType>> subarrays;
                    for (Size i = 0; i < dim1; ++i) {
                        ndarray::array_dynamic::NDArrayDynamic<DType> subarray;
                        auto shape = this->shape();
                        if (shape.empty()) {
                            throw std::runtime_error("Index " + std::to_string(i) + " of an empty array requested");
                        }
                        if (shape.size() == 1) {
                            if (i >= shape[0]) {
                                throw std::runtime_error("Index " + std::to_string(i) + " out of bounds");
                            }
                            subarray = ndarray::array_dynamic::NDArrayDynamic<DType>{this->get(i)};
                        } else {
                            auto layerSize = shape[0] == 0 ? 0 : this->size() / shape[0];
                            shape.removeFirstDim();
                            subarray = ndarray::array_dynamic::NDArrayDynamic<DType>{shape};
                            for (std::size_t j = 0; j < shape.calcSizeByShape(); ++j) {
                                subarray.set(j, get(i * layerSize + j));
                            }
                        }
                        subarrays.push_back(subarray.transpose());
                    }
                    std::size_t index = 0;
                    for (Size i = 0; i < subarrays[0].size(); ++i) {
                        for (Size d = 0; d < dim1; ++d) {
                            result.set(index++, subarrays[d].get(i));
                        }
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::ravel() const {
                auto sh = shape();
                sh.flatten();
                return reshape(sh);
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::reshape(const Shape &shape) const {
                if (size() != shape.calcSizeByShape())
                    throw std::runtime_error("Sizes of new and current arrays must be equal");
                array_dynamic::NDArrayDynamic<DType> result{shape};
                for (Size i = 0; i < shape.calcSizeByShape(); ++i) {
                    result.set(i, get(i));
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::resize(const Shape &shape) const {
                Size newSize = shape.calcSizeByShape();
                Size copySize = std::min(size(), newSize);
                array_dynamic::NDArrayDynamic<DType> result{shape};
                for (Size offset = 0; offset < newSize; offset += copySize) {
                    for (Size i = 0; i < copySize && i + offset < newSize; ++i) {
                        result.set(i + offset, get(i));
                    }
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::append(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                if (array.size() == 0)
                    return copy();
                auto size1 = shape().calcSizeByShape();
                auto size2 = array.shape().calcSizeByShape();
                Shape sh{size1 + size2};
                array_dynamic::NDArrayDynamic<DType> result{sh};
                for (Size i = 0; i < size(); ++i) {
                    result.set(i, get(i));
                }
                for (Size i = 0; i < array.size(); ++i) {
                    result.set(i + size(), array.get(i));
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType, Derived, Storage>::insert(Size index, const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                auto size1 = size();
                if (index > size1) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                if (array.size() == 0)
                    return copy();
                auto size2 = array.size();
                Shape sh{size1 + size2};
                array_dynamic::NDArrayDynamic<DType> result{sh};
                for (Size i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                for (Size i = 0; i < array.size(); ++i) {
                    result.set(i + index, array.get(i));
                }
                for (Size i = index; i < size(); ++i) {
                    result.set(i + array.size(), get(i));
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::del(Size index) const {
                auto sz = size();
                if (sz == 0)
                    throw std::runtime_error("Cannot del from an empty array");
                if (index > sz) {
                    throw std::runtime_error("Index exceeds array bounds");
                }
                Shape sh{sz - 1};
                array_dynamic::NDArrayDynamic<DType> result{sh};
                for (Size i = 0; i < index; ++i) {
                    result.set(i, get(i));
                }
                if (index < sz - 1) {
                    for (Size i = index + 1; i < size(); ++i) {
                        result.set(i - 1, get(i));
                    }
                }
                return result;
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::concatenate(const NDArrayBase<DType2, Derived2, Storage2> &array, std::optional<std::size_t> axis) const {
                if (array.size() == 0)
                    return copy();
                if (axis == std::nullopt) {
                    return concatenate(array, 0);
                }
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                // All the dims except the 'axis' should be equal
                for (std::size_t i = 0; i < sh1.size(); ++i) {
                    if (i != *axis && sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the first should be equal");
                }
                if (*axis >= sh1.size())
                    throw std::runtime_error("axis : " + std::to_string(*axis) + " is out of bounds for array of dimension " + std::to_string(sh1.size()));
                Shape sh{sh1};
                auto size1 = sh1.calcSizeByShape();
                auto size2 = sh2.calcSizeByShape();
                if (*axis == 0) {
                    sh[*axis] = sh1[*axis] + sh2[*axis];
                    array_dynamic::NDArrayDynamic<DType1> result{sh};
                    for (Size i = 0; i < size1; ++i) {
                        result.set(i, get(i));
                    }
                    for (Size i = 0; i < size2; ++i) {
                        result.set(i + size1, array.get(i));
                    }
                    return result;
                }
                if (*axis == 1) {
                    Size SizeT2 = std::accumulate(sh.begin() + 1, sh.end(), static_cast<Size>(1), std::multiplies<>());
                    sh[*axis] = sh1[*axis] + sh2[*axis];
                    ndarray::array_dynamic::NDArrayDynamic<DType1> result{sh};
                    Size destOffset = 0;
                    Size src1Offset = 0;
                    Size src2Offset = 0;
                    for (Size x = 0; x < sh1[0]; ++x) {
                        for (Size y = 0; y < SizeT2; ++y) {
                            result.set(destOffset++, get(src1Offset++));
                        }
                        for (Size y = 0; y < SizeT2; ++y) {
                            result.set(destOffset++, array.get(src2Offset++));
                        }
                    }
                    return result;
                }

                throw std::runtime_error("axis > 1 are not supported");
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::vstack(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                return concatenate(array);
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::r_(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                return concatenate(array);
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::hstack(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                if (size() == 0) {
                    return array.copy();
                }
                if (array.size() == 0) {
                    return copy();
                }
                // Both are not empty
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                return sh1.size() == 1 ?
                                       //concatenation along 1st axis
                               concatenate(array)
                                       :
                                       //concatenation along 2nd axis
                               concatenate(array, 1);
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::column_stack(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                if (sh1.empty() && sh2.empty()) {
                    return ndarray::array_dynamic::NDArrayDynamic<DType1>{};
                }
                if (sh1.size() == 1) {
                    Size length = sh1[0];
                    Shape sh{length, 2};
                    ndarray::array_dynamic::NDArrayDynamic<DType1> result{sh};
                    std::size_t index1 = 0;
                    std::size_t index2 = 0;
                    Size i = 0;
                    while (i < size() + array.size()) {
                        result.set(i++, get(index1++));
                        result.set(i++, array.get(index2++));
                    }
                    return result;
                }
                //concatenation along 2nd axis
                return concatenate(array, 1);
            }

            template<typename DType1, typename Derived1, typename Storage1>
            template<typename DType2, typename Derived2, typename Storage2>
            auto NDArrayBase<DType1, Derived1, Storage1>::c_(const NDArrayBase<DType2, Derived2, Storage2> &array) const {
                if (size() == 0) {
                    return array.copy();
                }
                if (array.size() == 0) {
                    return copy();
                }
                // Both are not empty
                Shape sh1 = shape();
                Shape sh2 = array.shape();
                if (sh1.size() != sh2.size())
                    throw std::runtime_error("Number of dims should be equal");
                if (sh1.size() == 1) {
                    //concatenation along 1st axis
                    return concatenate(array);
                }
                // All the dims except the last should be equal
                Size last = sh1.size() - 1;
                Size sizes = 1;
                for (Size i = 0; i < last; ++i) {
                    if (sh1[i] != sh2[i])
                        throw std::runtime_error("All the dims except the last should be equal");
                    sizes *= sh1[i];
                }
                Shape sh = shape();
                sh[last] = sh1[last] + sh2[last];
                ndarray::array_dynamic::NDArrayDynamic<DType1> result{sh};
                Size destOffset = 0;
                for (Size y = 0; y < sizes; ++y) {
                    for (Size i = 0; i < sh1[last]; ++i) {
                        result.set(i + destOffset, get(i + y * sh1[last]));
                    }
                    destOffset += sh1[last];
                    for (Size i = 0; i < sh2[last]; ++i) {
                        result.set(i + destOffset, array.get(i + y * sh2[last]));
                    }
                    destOffset += sh2[last];
                }
                return result;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::hsplit(std::size_t sections) const {
                if (sections == 0) {
                    throw std::runtime_error("Sections must not be 0");
                }
                Shape sh = shape();
                if (sh.empty()) {
                    std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.push_back(array_dynamic::NDArrayDynamic<DType>{});
                    }
                    return results;
                } else if (sh.size() == 1) {
                    Shape sh1{sh};
                    if (sh[0] % sections != 0) {
                        throw std::runtime_error("Array split does not result in an equal division");
                    }
                    Size sectionSize = sh[0] / sections;
                    sh1[0] = sectionSize;
                    std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        ndarray::array_dynamic::NDArrayDynamic<DType> result{sh1};
                        results.emplace_back(std::move(result));
                    }
                    std::vector<Size> sectionIndexes(sections);
                    for (Size i = 0; i < size(); ++i) {
                        Size section = i / sectionSize;
                        results[section].set(sectionIndexes[section]++, get(i));
                    }
                    return results;
                }
                Shape sh1{sh};
                if (sh[1] % sections != 0) {
                    throw std::runtime_error("Array split does not result in an equal division");
                }
                Size rest = 1;
                for (std::size_t i = 2; i < sh1.size(); ++i) {
                    rest *= sh1[i];
                }
                Size sectionSize = sh[1] / sections;
                sh1[1] = sectionSize;
                std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                for (std::size_t section = 0; section < sections; ++section) {
                    ndarray::array_dynamic::NDArrayDynamic<DType> result{sh1};
                    results.emplace_back(std::move(result));
                }
                std::vector<Size> sectionIndexes(sections);
                Size i = 0;
                std::size_t section = 0;
                while (i < size()) {
                    for (Size j = 0; j < rest; ++j) {
                        results[section].set(sectionIndexes[section]++, get(i++));
                    }
                    ++section;
                    if (section >= sections) {
                        section = 0;
                    }
                }
                return results;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::vsplit(std::size_t sections) const {
                if (sections == 0) {
                    throw std::runtime_error("Sections must not be 0");
                }
                Shape sh = shape();
                if (sh.empty()) {
                    std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        results.emplace_back(std::move(ndarray::array_dynamic::NDArrayDynamic<DType>{}));
                    }
                    return results;
                } else if (sh.size() == 1) {
                    Shape sh1{sh};
                    if (sh[0] % sections != 0) {
                        throw std::runtime_error("Array split does not result in an equal division");
                    }
                    Size sectionSize = sh[0] / sections;
                    sh1[0] = sectionSize;
                    std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                    for (std::size_t section = 0; section < sections; ++section) {
                        ndarray::array_dynamic::NDArrayDynamic<DType> result{sh1};
                        results.emplace_back(result);
                    }
                    std::vector<Size> sectionIndexes(sections);
                    for (Size i = 0; i < size(); ++i) {
                        Size section = i / sectionSize;
                        results[section].set(sectionIndexes[section]++, get(i));
                    }
                    return results;
                }
                Shape sh0{sh};
                if (sh[0] % sections != 0) {
                    throw std::runtime_error("Array split does not result in an equal division");
                }
                Size rest = 1;
                for (std::size_t i = 1; i < sh0.size(); ++i) {
                    rest *= sh0[i];
                }
                Size sectionSize = sh[0] / sections;
                sh0[0] = sectionSize;
                std::vector<array_dynamic::NDArrayDynamic<DType>> results;
                for (std::size_t section = 0; section < sections; ++section) {
                    ndarray::array_dynamic::NDArrayDynamic<DType> result{sh0};
                    results.emplace_back(std::move(result));
                }
                std::vector<Size> sectionIndexes(sections);
                Size i = 0;
                std::size_t section = 0;
                while (i < size()) {
                    for (Size j = 0; j < rest; ++j) {
                        results[section].set(sectionIndexes[section]++, get(i++));
                    }
                    ++section;
                    if (section >= sections) {
                        section = 0;
                    }
                }
                return results;
            }

            template<typename DType, typename Derived, typename Storage>
            auto NDArrayBase<DType, Derived, Storage>::load(std::istream &stream) {
                ndarray::internal::Descr descr;
                Shape shape;
                std::tie(descr, shape) = ndarray::internal::readNep1Header(stream);

                ndarray::internal::DTypeToDescrConvertor<DType> convertorByte{descr.size};
                NP_THROW_UNLESS_WITH_ARG(convertorByte.DTypeToChar() == descr.name, "Incorrect DType in input file: ",
                                         std::to_string(descr.name));
                std::size_t size = shape.calcSizeByShape();
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
                return ndarray::array_dynamic::NDArrayDynamic<DType>{std::move(data), std::move(shape)};
            }

            template<typename DType, typename Derived, typename Storage>
            typename NDArrayBase<DType, Derived, Storage>::IndexParentConstType NDArrayBase<DType, Derived, Storage>::operator[](Size i) const {
                auto shape = this->shape();
                if (shape.empty()) {
                    throw std::runtime_error("Index " + std::to_string(i) + " of an empty array requested");
                }
                if (shape.size() == 1 && i >= shape[0]) {
                    throw std::runtime_error("Index " + std::to_string(i) + " out of bounds");
                }

                IndexType<DType> index{SubsettingIndexType{i}};
                return {this, {index}};
            }

            template<typename DType, typename Derived, typename Storage>
            typename NDArrayBase<DType, Derived, Storage>::IndexParentType NDArrayBase<DType, Derived, Storage>::operator[](Size i) {
                auto shape = this->shape();
                if (shape.empty()) {
                    throw std::runtime_error("Index " + std::to_string(i) + " of an empty array requested");
                }
                if (shape.size() == 1 && i >= shape[0]) {
                    throw std::runtime_error("Index " + std::to_string(i) + " out of bounds");
                }

                IndexType<DType> index{SubsettingIndexType{i}};
                return {this, {index}};
            }

            // Subsetting
            // a[2] Select the element at the 2nd index
            // b[1,2] Select the element at row 1 column 2 (equivalent to b[1][2])
            // Boolean indexing
            // a[a < 2] Select elements from a less than 2
            // Slicing
            // a[0:2] Select items at index 0 and 1
            // b[0:2,1] Select items at rows 0 and 1 in column 1
            template<typename DType, typename Derived, typename Storage>
            typename NDArrayBase<DType, Derived, Storage>::IndexParentConstType NDArrayBase<DType, Derived, Storage>::operator[](const std::string &cond) const {
                std::size_t prevCommaPos{0};
                Size dimIndex{0};
                IndicesType<DType> indices;
                while (true) {
                    std::size_t commaPos = cond.find(',', prevCommaPos);
                    auto dimCond = cond.substr(prevCommaPos, commaPos == std::string::npos ? cond.size() : commaPos - prevCommaPos);
                    dimCond.erase(dimCond.find_last_not_of(" \n\r\t") + 1);
                    prevCommaPos = commaPos + 1;

                    auto index = runHandlers(dimIndex, dimCond);
                    indices.emplace_back(index);

                    if (commaPos == std::string::npos) {
                        break;
                    }

                    ++dimIndex;
                    if (dimIndex >= shape().size()) {
                        throw std::runtime_error("Too many indices for array: array is " +
                                                 std::to_string(shape().size()) +
                                                 "-dimensional, but " + std::to_string(dimIndex + 1) + " were indexed");
                    }
                }
                return {this, indices};
            }

            template<typename DType, typename Derived, typename Storage>
            typename NDArrayBase<DType, Derived, Storage>::IndexParentType NDArrayBase<DType, Derived, Storage>::operator[](const std::string &cond) {
                std::size_t prevCommaPos{0};
                Size dimIndex{0};
                IndicesType<DType> indices;
                while (true) {
                    std::size_t commaPos = cond.find(',', prevCommaPos);
                    auto dimCond = cond.substr(prevCommaPos, commaPos == std::string::npos ? cond.size() : commaPos - prevCommaPos);
                    dimCond.erase(dimCond.find_last_not_of(" \n\r\t") + 1);
                    prevCommaPos = commaPos + 1;

                    auto index = runHandlers(dimIndex, dimCond);
                    if (index.index() != 0) {
                        indices.emplace_back(index);
                    }

                    if (commaPos == std::string::npos) {
                        break;
                    }

                    ++dimIndex;
                    if (dimIndex >= shape().size()) {
                        throw std::runtime_error("Too many indices for array: array is " +
                                                 std::to_string(shape().size()) +
                                                 "-dimensional, but " + std::to_string(dimIndex + 1) + " were indexed");
                    }
                }
                return {this, indices};
            }

            template<typename DType, typename Derived, typename Storage>
            IndexType<DType> NDArrayBase<DType, Derived, Storage>::runHandlers(Size dimIndex, const std::string &dimCond) const {
                static constexpr std::size_t kIndexingHandlersSize{static_cast<std::size_t>(IndexingMode::Size)};

                const IndexingHandler<DType> indexingHandlers[kIndexingHandlersSize] = {
                        {IndexingMode::None,
                         isNone,
                         std::bind(&NDArrayBase::none, this, std::placeholders::_1, std::placeholders::_2)},
                        {IndexingMode::Subsetting,
                         isSubsetting,
                         std::bind(&NDArrayBase::subsetting, this, std::placeholders::_1, std::placeholders::_2)},
                        {IndexingMode::Slicing,
                         isSlicing,
                         std::bind(&NDArrayBase::slicing, this, std::placeholders::_1, std::placeholders::_2)},
                        {IndexingMode::BooleanIndexing,
                         isBooleanIndexing<DType>,
                         std::bind(&NDArrayBase::booleanIndexing, this, std::placeholders::_1, std::placeholders::_2)}};

                for (const auto &indexingHandler: indexingHandlers) {
                    if (indexingHandler.checker(dimCond)) {
                        return indexingHandler.worker(dimIndex, dimCond);
                    }
                }
                throw std::runtime_error("Invalid operator[] argument");
            }

            template<typename DType, typename Derived, typename Storage>
            IndexType<DType> NDArrayBase<DType, Derived, Storage>::none(Size, const std::string &) const {
                return IndexType<DType>{};
            }

            template<typename DType, typename Derived, typename Storage>
            IndexType<DType> NDArrayBase<DType, Derived, Storage>::subsetting(Size dimIndex, const std::string &dimCond) const {
                auto shape{this->shape()};
                Size index = std::stoul(dimCond);
                if (index > shape[dimIndex]) {
                    throw std::runtime_error("Index " + std::to_string(index) +
                                             " out of bounds for axis " + std::to_string(dimIndex) +
                                             " with size " + std::to_string(shape[dimIndex]));
                }
                return IndexType<DType>{SubsettingIndexType{index}};
            }

            template<typename DType, typename Derived, typename Storage>
            IndexType<DType> NDArrayBase<DType, Derived, Storage>::slicing(Size dimIndex, const std::string &dimCond) const {
                auto colonPos = dimCond.find(':');
                auto shape{this->shape()};
                Size first = 0;
                Size last = shape[dimIndex];
                Size step = 1;
                if (colonPos == std::string::npos) {
                    throw std::runtime_error("Invalid format");
                } else {
                    auto firstStr = dimCond.substr(0, colonPos);
                    if (!firstStr.empty()) {
                        first = std::stoul(firstStr);
                    }
                    auto lastStr = dimCond.substr(colonPos + 1, dimCond.size() - colonPos - 1);
                    if (!lastStr.empty()) {
                        last = std::stoul(lastStr);
                    }
                    auto colonPosStep = dimCond.find(':', colonPos + 1);
                    if (colonPosStep != std::string::npos) {
                        auto stepStr = dimCond.substr(colonPosStep + 1, dimCond.size() - colonPosStep - 1);
                        if (!stepStr.empty()) {
                            step = std::stoul(stepStr);
                        }
                    }
                }
                if (last > shape[dimIndex]) {
                    throw std::runtime_error("Index " + std::to_string(last) +
                                             " out of bounds for axis " + std::to_string(dimIndex) +
                                             " with size " + std::to_string(shape[dimIndex]));
                }
                if (first >= last) {
                    last = first;
                }

                return IndexType<DType>{SlicingIndexType{first, last, step}};
            }

            template<typename DType, typename Derived, typename Storage>
            IndexType<DType> NDArrayBase<DType, Derived, Storage>::booleanIndexing(Size, const std::string &dimCond) const {
                return IndexType<DType>{getOperatorWithArg<DType>(dimCond)};
            }

        }// namespace internal
    }    // namespace ndarray
}// namespace np

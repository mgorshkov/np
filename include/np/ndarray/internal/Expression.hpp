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

#include <immintrin.h>
#include <np/internal/CpuDispatch.hpp>
#include <np/ndarray/internal/NDArrayBase.hpp>
#include <type_traits>

// AMX tile configuration
// AMX uses 8 tile registers (tmm0-tmm7), each is a 2D matrix.
// Palette 1: max 16 rows, max 64 bytes per row = 1024 bytes per tile
// For double: 8 doubles/row × 16 rows = 128 doubles per tile
// For float: 16 floats/row × 16 rows = 256 floats per tile
//
// AMX doesn't have element-wise SIMD operations like AVX.
// The strategy is to use AMX tiles as "wide load/store" units:
// - Load 128 doubles (or 256 floats) at once into a tile
// - Process each row using AVX512 intrinsics
// - Store results back
// This gives 16x wider loads than AVX512 (1024 bytes vs 64 bytes for zmm).

#include <cstdint>
#include <cstring>

namespace np {
    namespace ndarray {
        namespace internal {

#ifdef ENABLE_AMX
            // AMX tile config structure (matches hardware layout)
            struct AmxTileConfig {
                uint8_t palette_id;
                uint8_t start_row;
                uint8_t reserved[14];
                uint16_t colsb[8];
                uint8_t rows[8];
            };

            // Initialize AMX tiles for double-precision processing.
            // Configures tiles with 16 rows × 64 bytes (8 doubles per row).
            // Must be called before any tile operations.
            // Returns the number of elements per tile (128 for double).
            inline void amx_init_tiles() {
                AmxTileConfig cfg;
                std::memset(&cfg, 0, sizeof(cfg));
                cfg.palette_id = 1;
                for (int i = 0; i < 8; ++i) {
                    cfg.rows[i] = 16;
                    cfg.colsb[i] = 64;// 64 bytes per row
                }
                _tile_loadconfig(&cfg);
            }

            // Release AMX tile state.
            inline void amx_release_tiles() {
                _tile_release();
            }

            // AMX tile constants
            static constexpr Size kAmxTileRows = 16;
            static constexpr Size kAmxTileColBytes = 64;
            static constexpr Size kAmxDoublesPerTile = kAmxTileRows * (kAmxTileColBytes / 8);// 128
            static constexpr Size kAmxFloatsPerTile = kAmxTileRows * (kAmxTileColBytes / 4); // 256
#endif

        }// namespace internal
    }// namespace ndarray
}// namespace np


#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <np/internal/cuda/Add.hpp>
#include <np/internal/cuda/Divide.hpp>
#include <np/internal/cuda/Multiply.hpp>
#include <np/internal/cuda/Subtract.hpp>
#endif

namespace np {
    namespace ndarray {
        namespace internal {

            // Forward declarations
            template<typename Op, typename Left, typename Right>
            class BinaryExpression;

            // Operation tags
            struct AddOp {};
            struct MultiplyOp {};
            struct SubtractOp {};
            struct DivideOp {};
            struct LessOp {};

            // Scalar wrapper
            template<typename Scalar>
            struct ScalarWrapper {
                using DType = Scalar;
                Scalar value;

                Scalar get(Size) const { return value; }
                Size size() const { return 1; }
                Shape shape() const { return Shape{1}; }
            };

            // Empty storage for expression types (doesn't store actual data)
            template<typename DType>
            struct EmptyStorage {
                static constexpr bool is_contiguous = false;
                static constexpr Size kDepth = 0;
                EmptyStorage() = default;

                explicit EmptyStorage(const Shape &shape) : m_shape(shape) {}

                [[nodiscard]] const Shape &shape() const { return m_shape; }
                void setShape(const Shape &shape) { m_shape = shape; }

                [[nodiscard]] Size size() const { return m_shape.calcSizeByShape(); }

                // Return reference to a static default value to avoid returning reference to temporary
                // This is used by NDArrayBase::get() which returns const DType&
                // The actual get() for expressions is overridden in BinaryExpression
                const DType &get(Size) const {
                    static DType defaultValue{};
                    return defaultValue;
                }
                void set(Size, const DType &) {}

            private:
                Shape m_shape;
            };

            // Expression base class
            template<typename Derived, typename DType>
            class ExpressionBase {
            public:
                // Support indexing with string (evaluates first)
                auto operator[](const std::string &cond) const {
                    return eval()[cond];
                }

                // Support size() etc.
                Size size() const {
                    return static_cast<const Derived &>(*this).size();
                }

                // Support shape()
                Shape shape() const {
                    return static_cast<const Derived &>(*this).shape();
                }

                // Conversion to array (implicit)
                operator Array<DType>() const {
                    return eval();
                }

                // Evaluate expression to an array
                auto eval() const {
                    return static_cast<const Derived &>(*this).eval();
                }

                // Count true values for boolean expressions
                Size count_if() const {
                    return static_cast<const Derived &>(*this).count_if();
                }

                // Fused count of elements less than a scalar threshold.
                // For expression trees like (a*b + c*d) < threshold, this performs
                // the entire computation in a single fused AVX2 pass with no
                // intermediate array allocations.
                // This is the key optimization for Monte Carlo PI estimation.
                template<Arithmetic Scalar>
                Size count_less_than(const Scalar &threshold) const {
                    auto less_expr = BinaryExpression<LessOp, Derived, ScalarWrapper<Scalar>>(
                            static_cast<const Derived &>(*this),
                            ScalarWrapper<Scalar>{static_cast<Scalar>(threshold)});
                    return less_expr.count_if();
                }

                // Sum of elements matching a boolean condition string.
                // For expression trees like (a*b + c*d) with condition "dist<1",
                // this uses the fused AVX2 count_if() path with no intermediate
                // array allocations.
                //
                // \param cond A condition string (e.g., "dist<1", "x>0.5")
                // \return Count of elements matching the condition as float_
                float_ sum(const std::string &cond) const {
                    auto opWithArg = ndarray::internal::getOperatorWithArg<typename Derived::DType>(cond);
                    if (opWithArg.m_operator == ndarray::internal::Operator::Less) {
                        return static_cast<float_>(count_less_than(opWithArg.m_arg));
                    }
                    // For other operators, fall back to eval + boolean indexing
                    auto selected = static_cast<const Derived &>(*this)[cond];
                    return static_cast<float_>(selected.size());
                }

                // Lazy comparison: returns BinaryExpression<LessOp, ...> instead of materializing
                // This enables fused evaluation paths like count_if() for (a*b + c*d) < scalar
                template<typename OtherDerived, typename OtherDType>
                auto operator<(const ExpressionBase<OtherDerived, OtherDType> &other) const {
                    return BinaryExpression<LessOp, Derived, OtherDerived>(
                            static_cast<const Derived &>(*this),
                            static_cast<const OtherDerived &>(other));
                }

                // Lazy comparison with scalar
                template<Arithmetic Scalar>
                auto operator<(const Scalar &value) const {
                    return BinaryExpression<LessOp, Derived, ScalarWrapper<Scalar>>(
                            static_cast<const Derived &>(*this),
                            ScalarWrapper<Scalar>{value});
                }
            };

            // Helper: check if a type is a BinaryExpression with a specific operation
            template<typename T, typename Op>
            struct is_binary_op : std::false_type {};

            template<typename Op1, typename Left, typename Right, typename Op2>
            struct is_binary_op<BinaryExpression<Op1, Left, Right>, Op2> : std::is_same<Op1, Op2> {};

            // Helper: check if a type is a BinaryExpression with any arithmetic operation
            template<typename T>
            struct is_arithmetic_binary_expr : std::false_type {};

            template<typename Op, typename Left, typename Right>
            struct is_arithmetic_binary_expr<BinaryExpression<Op, Left, Right>> {
                static constexpr bool value = std::is_same_v<Op, AddOp> || std::is_same_v<Op, SubtractOp> ||
                                              std::is_same_v<Op, MultiplyOp> || std::is_same_v<Op, DivideOp>;
            };

            // Forward declaration of ArrayExpression (defined later in this file)
            template<typename T, typename Derived, typename Storage>
            class ArrayExpression;

            // Helper: check if a type is an ArrayExpression (leaf array)
            template<typename T>
            struct is_array_expression : std::false_type {};

            template<typename T, typename Derived, typename Storage>
            struct is_array_expression<ArrayExpression<T, Derived, Storage>> : std::true_type {};

            // Binary expression template
            // Inherits from both ExpressionBase and NDArrayBase to be compatible with compare()
            //
            // SIMD register-based evaluation:
            // Each BinaryExpression node computes its operation using SIMD registers.
            // The compute_reg256(i) method returns a __m256d register for chunk i by:
            //   - Getting registers from children (which recursively compute from their children)
            //   - Applying the operation using SIMD intrinsics
            //   - Storing the result in m_reg256
            //
            // For the Monte Carlo PI estimation (a*b + c*d) < 1.0:
            //   rx*rx: load rx, mul → __m256d (in m_reg256)
            //   ry*ry: load ry, mul → __m256d (in m_reg256)
            //   (rx*rx)+(ry*ry): get children's __m256d, add → __m256d (in m_reg256)
            //   dist < 1.0: get left's __m256d, cmp, movemask, popcount → count (no store!)
            //
            // Total: 4 loads, 0 stores for count_if(), 1 store for eval()
            template<typename Op, typename Left, typename Right>
            class BinaryExpression : public ExpressionBase<BinaryExpression<Op, Left, Right>, typename std::conditional_t<std::is_same_v<Op, LessOp>, bool, typename std::common_type_t<typename Left::DType, typename Right::DType>>>,
                                     public ndarray::internal::NDArrayBase<typename std::conditional_t<std::is_same_v<Op, LessOp>, bool, typename std::common_type_t<typename Left::DType, typename Right::DType>>, BinaryExpression<Op, Left, Right>, EmptyStorage<typename std::conditional_t<std::is_same_v<Op, LessOp>, bool, typename std::common_type_t<typename Left::DType, typename Right::DType>>>> {
            public:
                using DType = typename std::conditional_t<std::is_same_v<Op, LessOp>, bool, typename std::common_type_t<typename Left::DType, typename Right::DType>>;
                using OpType = Op;
                using LeftType = Left;
                using RightType = Right;

                BinaryExpression(Left left, Right right)
                    : ndarray::internal::NDArrayBase<DType, BinaryExpression<Op, Left, Right>, EmptyStorage<DType>>(left.shape().broadcast(right.shape())),
                      m_left(std::move(left)), m_right(std::move(right)) {}

                // Override get() to return computed value
                // Returns by value since each index computes a different value
                // Handles broadcasting by wrapping indices for operands with different sizes
                DType get(Size i) const override {
                    const Size leftSize = m_left.size();
                    const Size rightSize = m_right.size();
                    if (leftSize == rightSize) {
                        return applyOp(m_left.get(i), m_right.get(i));
                    }
                    // Broadcasting: wrap indices for the smaller operand
                    return applyOp(m_left.get(i % leftSize), m_right.get(i % rightSize));
                }

                // Explicit operator[] overloads to resolve ambiguity between
                // ExpressionBase::operator[](const std::string&) and NDArrayBase::operator[](SignedSize)
                // when a string literal is passed (e.g., dist["dist<1"])
                // Evaluates the expression first and caches the result to avoid dangling pointer
                // in the returned IndexParentConstType (which holds a raw pointer to the parent array).
                // Without caching, eval() returns a temporary Array, and operator[] on the temporary
                // returns an IndexParentConstType with a dangling pointer.
                auto operator[](const std::string &cond) const {
                    if (!m_evaluated) {
                        m_evaluated = true;
                        const_cast<BinaryExpression *>(this)->m_cache = eval();
                    }
                    return m_cache[cond];
                }

                auto operator[](const char *cond) const {
                    if (!m_evaluated) {
                        m_evaluated = true;
                        const_cast<BinaryExpression *>(this)->m_cache = eval();
                    }
                    return m_cache[cond];
                }

                Size size() const override {
                    return shape().calcSizeByShape();
                }

                Shape shape() const override {
                    return m_left.shape().broadcast(m_right.shape());
                }

                // Evaluate the expression to an array.
                // Uses SIMD registers flowing from child nodes - each node computes
                // its operation in registers and passes them up the chain.
                // Only the top-level eval() stores to an array.
                // Helper: evaluate with broadcasting-aware generic path (no SIMD).
                // Used as fallback when operands have different sizes (broadcasting).
                void eval_broadcast(Array<DType> &result, Size n) const {
                    const Size leftSize = m_left.size();
                    const Size rightSize = m_right.size();
                    auto result_data = result.data();
                    for (Size i = 0; i < n; ++i) {
                        result_data[i] = applyOp(m_left.get(i % leftSize), m_right.get(i % rightSize));
                    }
                }

                Array<DType> eval() const {
                    const Size n = size();
                    Array<DType> result(Shape{n});

                    constexpr bool isArithmeticOp = std::is_same_v<Op, AddOp> || std::is_same_v<Op, SubtractOp> ||
                                                    std::is_same_v<Op, MultiplyOp> || std::is_same_v<Op, DivideOp>;

                    if constexpr (isArithmeticOp) {
                        // CUDA branch for large arrays
#ifdef USE_CUDA
                        if (n > 256) {
                            const auto left_array = m_left.eval();
                            const auto right_array = m_right.eval();
                            if constexpr (std::is_same_v<Op, AddOp>) {
                                np::internal::cuda::add(left_array.data(), left_array.size(),
                                                        right_array.data(), right_array.size(),
                                                        result.data(), result.size());
                            } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                                np::internal::cuda::subtract(left_array.data(), left_array.size(),
                                                             right_array.data(), right_array.size(),
                                                             result.data(), result.size());
                            } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                                np::internal::cuda::multiply(left_array.data(), left_array.size(),
                                                             right_array.data(), right_array.size(),
                                                             result.data(), result.size());
                            } else if constexpr (std::is_same_v<Op, DivideOp>) {
                                np::internal::cuda::divide(left_array.data(), left_array.size(),
                                                           right_array.data(), right_array.size(),
                                                           result.data(), result.size());
                            }
                            return result;
                        }
#endif
                        // For double-precision, use SIMD register-based evaluation
                        // where each node computes in registers and passes them up
                        if constexpr (std::is_same_v<DType, double>) {
                            // Broadcasting check: SIMD paths access data() directly with linear indices
                            // and cannot handle operands with different sizes. Fall back to generic path
                            // which wraps indices via get(i % size) for correct broadcasting.
                            if (m_left.size() != m_right.size()) {
                                eval_broadcast(result, n);
                                return result;
                            }

                            auto result_data = result.data();

                            Size i = 0;
#ifdef ENABLE_AMX
                            // AMX: 128 doubles per tile (16 rows × 8 doubles per row)
                            // Load tiles from left and right operands, compute row-by-row with AVX512
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AMX)) {
                                amx_init_tiles();
                                for (; i + kAmxDoublesPerTile - 1 < n; i += kAmxDoublesPerTile) {
                                    // Load left and right tiles
                                    _tile_loadd(0, m_left.data() + i, kAmxTileColBytes);
                                    _tile_loadd(1, m_right.data() + i, kAmxTileColBytes);
                                    // Process each row of the tiles using AVX512
                                    for (Size row = 0; row < kAmxTileRows; ++row) {
                                        Size offset = i + row * (kAmxTileColBytes / 8);
                                        __m512d a = _mm512_loadu_pd(m_left.data() + offset);
                                        __m512d b = _mm512_loadu_pd(m_right.data() + offset);
                                        __m512d c;
                                        if constexpr (std::is_same_v<Op, AddOp>) {
                                            c = _mm512_add_pd(a, b);
                                        } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                                            c = _mm512_sub_pd(a, b);
                                        } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                                            c = _mm512_mul_pd(a, b);
                                        } else if constexpr (std::is_same_v<Op, DivideOp>) {
                                            c = _mm512_div_pd(a, b);
                                        }
                                        _mm512_storeu_pd(result_data + offset, c);
                                    }
                                }
                                amx_release_tiles();
                            }
#endif
#ifdef ENABLE_AVX512
                            // AVX512: 8 doubles per iteration
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX512)) {
                                for (; i + 7 < n; i += 8) {
                                    __m512d c = compute_reg512(i);
                                    _mm512_storeu_pd(result_data + i, c);
                                }
                            }
#endif
#ifdef ENABLE_AVX2
                            // AVX2: 4 doubles per iteration
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX2)) {
                                for (; i + 3 < n; i += 4) {
                                    __m256d c = compute_reg256(i);
                                    _mm256_storeu_pd(result_data + i, c);
                                }
                            }
#endif
                            // Scalar remainder
                            for (; i < n; ++i) {
                                result_data[i] = applyOp(m_left.get(i), m_right.get(i));
                            }
                            return result;
                        }

                        // For float, use SIMD register-based evaluation
                        if constexpr (std::is_same_v<DType, float>) {
                            // Broadcasting check: SIMD paths access data() directly with linear indices
                            // and cannot handle operands with different sizes. Fall back to generic path
                            // which wraps indices via get(i % size) for correct broadcasting.
                            if (m_left.size() != m_right.size()) {
                                eval_broadcast(result, n);
                                return result;
                            }

                            auto result_data = result.data();

                            Size i = 0;
#ifdef ENABLE_AMX
                            // AMX: 256 floats per tile (16 rows × 16 floats per row)
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AMX)) {
                                amx_init_tiles();
                                for (; i + kAmxFloatsPerTile - 1 < n; i += kAmxFloatsPerTile) {
                                    // Load left and right tiles
                                    _tile_loadd(0, m_left.data() + i, kAmxTileColBytes);
                                    _tile_loadd(1, m_right.data() + i, kAmxTileColBytes);
                                    // Process each row of the tiles using AVX512
                                    for (Size row = 0; row < kAmxTileRows; ++row) {
                                        Size offset = i + row * (kAmxTileColBytes / 4);
                                        __m512 a = _mm512_loadu_ps(m_left.data() + offset);
                                        __m512 b = _mm512_loadu_ps(m_right.data() + offset);
                                        __m512 c;
                                        if constexpr (std::is_same_v<Op, AddOp>) {
                                            c = _mm512_add_ps(a, b);
                                        } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                                            c = _mm512_sub_ps(a, b);
                                        } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                                            c = _mm512_mul_ps(a, b);
                                        } else if constexpr (std::is_same_v<Op, DivideOp>) {
                                            c = _mm512_div_ps(a, b);
                                        }
                                        _mm512_storeu_ps(result_data + offset, c);
                                    }
                                }
                                amx_release_tiles();
                            }
#endif
#ifdef ENABLE_AVX512
                            // AVX512: 16 floats per iteration
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX512)) {
                                for (; i + 15 < n; i += 16) {
                                    __m512 c = compute_reg512_ps(i);
                                    _mm512_storeu_ps(result_data + i, c);
                                }
                            }
#endif
#ifdef ENABLE_AVX2
                            // AVX2: 8 floats per iteration
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX2)) {
                                for (; i + 7 < n; i += 8) {
                                    __m256 c = compute_reg256_ps(i);
                                    _mm256_storeu_ps(result_data + i, c);
                                }
                            }
#endif
                            // Scalar remainder
                            for (; i < n; ++i) {
                                result_data[i] = applyOp(m_left.get(i), m_right.get(i));
                            }
                            return result;
                        }
                    }

                    // Generic fallback with OpenMP parallelization
                    // Handles:
                    //   - Non-floating-point arithmetic types (e.g., int)
                    //   - Non-arithmetic operations (e.g., LessOp)
                    //   - Broadcasting by wrapping indices for operands with different sizes
                    {
                        const Size leftSize = m_left.size();
                        const Size rightSize = m_right.size();
#ifdef USE_OPENMP
#pragma omp parallel for default(none) shared(result, leftSize, rightSize, n)
#endif
                        for (Size i = 0; i < n; ++i) {
                            result.set(i, applyOp(m_left.get(i % leftSize), m_right.get(i % rightSize)));
                        }
                    }
                    return result;
                }

                // Count true values for boolean expressions (only for LessOp).
                // Uses SIMD registers flowing from child nodes - each node computes
                // its operation in registers and passes them up the chain.
                // No intermediate arrays are materialized.
                Size count_if() const {
                    if constexpr (std::is_same_v<Op, LessOp>) {
                        const Size n = size();
                        Size count = 0;

                        // For double-precision comparisons, use SIMD register-based path
                        if constexpr (std::is_same_v<typename Left::DType, double>) {
                            [[maybe_unused]] double threshold = m_right.get(0);

                            Size i = 0;
#if defined(ENABLE_AMX) && defined(ENABLE_AVX512)
                            // AMX: 128 doubles per tile (16 rows × 8 doubles per row)
                            // For count_if(), the left operand is a BinaryExpression (e.g., AddOp).
                            // We load tiles for the leaf arrays and compute the expression row-by-row.
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AMX)) {
                                amx_init_tiles();
                                __m512d thresh = _mm512_set1_pd(threshold);
                                for (; i + kAmxDoublesPerTile - 1 < n; i += kAmxDoublesPerTile) {
                                    // Process each row of the tile using AVX512
                                    // The left operand's compute_reg512() recursively evaluates
                                    // the expression tree for each 8-double chunk
                                    for (Size row = 0; row < kAmxTileRows; ++row) {
                                        Size offset = i + row * 8;
                                        __m512d a = this->get_left_reg512(offset);
                                        __mmask8 mask = _mm512_cmp_pd_mask(a, thresh, _CMP_LT_OQ);
                                        count += __builtin_popcount(mask & 0xFF);
                                    }
                                }
                                amx_release_tiles();
                            }
#endif
#ifdef ENABLE_AVX512
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX512)) {
                                __m512d thresh = _mm512_set1_pd(threshold);
                                for (; i + 7 < n; i += 8) {
                                    // Get the left operand's SIMD register directly
                                    // This recursively computes the expression tree in registers
                                    __m512d a = this->get_left_reg512(i);
                                    __mmask8 mask = _mm512_cmp_pd_mask(a, thresh, _CMP_LT_OQ);
                                    count += __builtin_popcount(mask & 0xFF);
                                }
                            }
#endif
#ifdef ENABLE_AVX2
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX2)) {
                                __m256d thresh = _mm256_set1_pd(threshold);
                                for (; i + 3 < n; i += 4) {
                                    // Get the left operand's SIMD register directly
                                    // This recursively computes the expression tree in registers
                                    __m256d a = get_left_reg256(i);
                                    __m256d mask = _mm256_cmp_pd(a, thresh, _CMP_LT_OQ);
                                    int mask_bits = _mm256_movemask_pd(mask);
                                    count += __builtin_popcount(mask_bits);
                                }
                            }
#endif
                            // Scalar remainder
                            for (; i < n; ++i) {
                                if (applyOp(m_left.get(i), m_right.get(i))) ++count;
                            }
                            return count;
                        }

                        // For single-precision comparisons, use SIMD register-based path
                        if constexpr (std::is_same_v<typename Left::DType, float>) {
                            [[maybe_unused]] float threshold = static_cast<float>(m_right.get(0));

                            Size i = 0;
#if defined(ENABLE_AMX) && defined(ENABLE_AVX512)
                            // AMX: 256 floats per tile (16 rows × 16 floats per row)
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AMX)) {
                                amx_init_tiles();
                                __m512 thresh = _mm512_set1_ps(threshold);
                                for (; i + kAmxFloatsPerTile - 1 < n; i += kAmxFloatsPerTile) {
                                    // Process each row of the tile using AVX512
                                    for (Size row = 0; row < kAmxTileRows; ++row) {
                                        Size offset = i + row * 16;
                                        __m512 a = this->get_left_reg512_ps(offset);
                                        __mmask16 mask = _mm512_cmp_ps_mask(a, thresh, _CMP_LT_OQ);
                                        count += __builtin_popcount(mask & 0xFFFF);
                                    }
                                }
                                amx_release_tiles();
                            }
#endif
#ifdef ENABLE_AVX512
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX512)) {
                                __m512 thresh = _mm512_set1_ps(threshold);
                                for (; i + 15 < n; i += 16) {
                                    // Get the left operand's SIMD register directly
                                    // This recursively computes the expression tree in registers
                                    __m512 a = this->get_left_reg512_ps(i);
                                    __mmask16 mask = _mm512_cmp_ps_mask(a, thresh, _CMP_LT_OQ);
                                    count += __builtin_popcount(mask & 0xFFFF);
                                }
                            }
#endif
#ifdef ENABLE_AVX2
                            if (np::internal::simd_at_least(np::internal::SimdLevel::AVX2)) {
                                __m256 thresh = _mm256_set1_ps(threshold);
                                for (; i + 7 < n; i += 8) {
                                    // Get the left operand's SIMD register directly
                                    // This recursively computes the expression tree in registers
                                    __m256 a = get_left_reg256_ps(i);
                                    __m256 mask = _mm256_cmp_ps(a, thresh, _CMP_LT_OQ);
                                    int mask_bits = _mm256_movemask_ps(mask);
                                    count += __builtin_popcount(mask_bits);
                                }
                            }
#endif
                            // Scalar remainder
                            for (; i < n; ++i) {
                                if (applyOp(m_left.get(i), m_right.get(i))) ++count;
                            }
                            return count;
                        }

                        // Generic fallback for non-floating-point types
                        for (Size i = 0; i < n; ++i) {
                            if (applyOp(m_left.get(i), m_right.get(i))) ++count;
                        }
                        return count;
                    } else {
                        // For non-boolean expressions, count_if is not meaningful
                        static_assert(std::is_same_v<Op, LessOp>, "count_if only supported for boolean expressions");
                        return 0;
                    }
                }

            private:
                Left m_left;
                Right m_right;
                mutable bool m_evaluated{false};
                mutable Array<DType> m_cache;

                template<typename T1, typename T2>
                auto applyOp(T1 a, T2 b) const {
                    if constexpr (std::is_same_v<Op, AddOp>) {
                        return a + b;
                    } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                        return a * b;
                    } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                        return a - b;
                    } else if constexpr (std::is_same_v<Op, DivideOp>) {
                        return a / b;
                    } else if constexpr (std::is_same_v<Op, LessOp>) {
                        return a < b;
                    }
                }

#ifdef ENABLE_AVX2
                // Compute a __m256d register for chunk i by getting registers from children
                // and applying this node's operation. No memory stores involved.
                __m256d compute_reg256(Size i) const {
                    __m256d a = get_left_reg256(i);
                    __m256d b = get_right_reg256(i);
                    if constexpr (std::is_same_v<Op, AddOp>) {
                        return _mm256_add_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                        return _mm256_sub_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                        return _mm256_mul_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, DivideOp>) {
                        return _mm256_div_pd(a, b);
                    }
                }
#endif

#ifdef ENABLE_AVX512
                // Compute a __m512d register for chunk i (AVX512)
                __m512d compute_reg512(Size i) const {
                    __m512d a = get_left_reg512(i);
                    __m512d b = get_right_reg512(i);
                    if constexpr (std::is_same_v<Op, AddOp>) {
                        return _mm512_add_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                        return _mm512_sub_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                        return _mm512_mul_pd(a, b);
                    } else if constexpr (std::is_same_v<Op, DivideOp>) {
                        return _mm512_div_pd(a, b);
                    }
                }

                // Compute a __m512 register for chunk i (float, AVX512)
                __m512 compute_reg512_ps(Size i) const {
                    __m512 a = get_left_reg512_ps(i);
                    __m512 b = get_right_reg512_ps(i);
                    if constexpr (std::is_same_v<Op, AddOp>) {
                        return _mm512_add_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                        return _mm512_sub_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                        return _mm512_mul_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, DivideOp>) {
                        return _mm512_div_ps(a, b);
                    }
                }
#endif

#ifdef ENABLE_AVX2
                // Compute a __m256 register for chunk i (float, AVX2)
                __m256 compute_reg256_ps(Size i) const {
                    __m256 a = get_left_reg256_ps(i);
                    __m256 b = get_right_reg256_ps(i);
                    if constexpr (std::is_same_v<Op, AddOp>) {
                        return _mm256_add_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                        return _mm256_sub_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                        return _mm256_mul_ps(a, b);
                    } else if constexpr (std::is_same_v<Op, DivideOp>) {
                        return _mm256_div_ps(a, b);
                    }
                }
#endif

#ifdef ENABLE_AVX2
                // Get the left operand's __m256d register for chunk i.
                // If left is a BinaryExpression, recursively compute its register.
                // If left is an ArrayExpression, load from its data pointer directly.
                __m256d get_left_reg256(Size i) const {
                    if constexpr (is_binary_op<Left, AddOp>::value ||
                                  is_binary_op<Left, SubtractOp>::value ||
                                  is_binary_op<Left, MultiplyOp>::value ||
                                  is_binary_op<Left, DivideOp>::value) {
                        // Left is a BinaryExpression - recursively compute its register
                        return static_cast<const BinaryExpression<typename Left::OpType, typename Left::LeftType, typename Left::RightType> &>(m_left).compute_reg256(i);
                    } else {
                        // Left is an ArrayExpression - load from data pointer directly
                        // No eval() needed - data() returns pointer to actual storage
                        return _mm256_loadu_pd(m_left.data() + i);
                    }
                }

                // Get the right operand's __m256d register for chunk i.
                __m256d get_right_reg256(Size i) const {
                    if constexpr (is_binary_op<Right, AddOp>::value ||
                                  is_binary_op<Right, SubtractOp>::value ||
                                  is_binary_op<Right, MultiplyOp>::value ||
                                  is_binary_op<Right, DivideOp>::value) {
                        // Right is a BinaryExpression - recursively compute its register
                        return static_cast<const BinaryExpression<typename Right::OpType, typename Right::LeftType, typename Right::RightType> &>(m_right).compute_reg256(i);
                    } else {
                        // Right is an ArrayExpression - load from data pointer directly
                        return _mm256_loadu_pd(m_right.data() + i);
                    }
                }
#endif

#ifdef ENABLE_AVX512
                // Get the left operand's __m512d register for chunk i (AVX512)
                __m512d get_left_reg512(Size i) const {
                    if constexpr (is_binary_op<Left, AddOp>::value ||
                                  is_binary_op<Left, SubtractOp>::value ||
                                  is_binary_op<Left, MultiplyOp>::value ||
                                  is_binary_op<Left, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Left::OpType, typename Left::LeftType, typename Left::RightType> &>(m_left).compute_reg512(i);
                    } else {
                        return _mm512_loadu_pd(m_left.data() + i);
                    }
                }
                // Get the right operand's __m512d register for chunk i (AVX512)
                __m512d get_right_reg512(Size i) const {
                    if constexpr (is_binary_op<Right, AddOp>::value ||
                                  is_binary_op<Right, SubtractOp>::value ||
                                  is_binary_op<Right, MultiplyOp>::value ||
                                  is_binary_op<Right, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Right::OpType, typename Right::LeftType, typename Right::RightType> &>(m_right).compute_reg512(i);
                    } else {
                        return _mm512_loadu_pd(m_right.data() + i);
                    }
                }
#endif
#ifdef ENABLE_AMX
                // Compute a tile of __m512d registers for chunk i (AMX tile-based).
                // Processes kAmxDoublesPerTile (128) doubles at once using AMX tiles.
                // Loads tiles from leaf arrays and computes the expression row-by-row.
                // Returns the result via the output array.
                void compute_tile_amx(double *result_data, Size i) const {
                    // Load left and right tiles
                    _tile_loadd(0, m_left.data() + i, kAmxTileColBytes);
                    _tile_loadd(1, m_right.data() + i, kAmxTileColBytes);
                    // Process each row using AVX512
                    for (Size row = 0; row < kAmxTileRows; ++row) {
                        Size offset = row * (kAmxTileColBytes / 8);
                        __m512d a = _mm512_loadu_pd(m_left.data() + i + offset);
                        __m512d b = _mm512_loadu_pd(m_right.data() + i + offset);
                        __m512d c;
                        if constexpr (std::is_same_v<Op, AddOp>) {
                            c = _mm512_add_pd(a, b);
                        } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                            c = _mm512_sub_pd(a, b);
                        } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                            c = _mm512_mul_pd(a, b);
                        } else if constexpr (std::is_same_v<Op, DivideOp>) {
                            c = _mm512_div_pd(a, b);
                        }
                        _mm512_storeu_pd(result_data + i + offset, c);
                    }
                }

                // Compute a tile of __m512 registers for chunk i (float, AMX tile-based).
                // Processes kAmxFloatsPerTile (256) floats at once.
                void compute_tile_amx_ps(float *result_data, Size i) const {
                    // Load left and right tiles
                    _tile_loadd(0, m_left.data() + i, kAmxTileColBytes);
                    _tile_loadd(1, m_right.data() + i, kAmxTileColBytes);
                    // Process each row using AVX512
                    for (Size row = 0; row < kAmxTileRows; ++row) {
                        Size offset = row * (kAmxTileColBytes / 4);
                        __m512 a = _mm512_loadu_ps(m_left.data() + i + offset);
                        __m512 b = _mm512_loadu_ps(m_right.data() + i + offset);
                        __m512 c;
                        if constexpr (std::is_same_v<Op, AddOp>) {
                            c = _mm512_add_ps(a, b);
                        } else if constexpr (std::is_same_v<Op, SubtractOp>) {
                            c = _mm512_sub_ps(a, b);
                        } else if constexpr (std::is_same_v<Op, MultiplyOp>) {
                            c = _mm512_mul_ps(a, b);
                        } else if constexpr (std::is_same_v<Op, DivideOp>) {
                            c = _mm512_div_ps(a, b);
                        }
                        _mm512_storeu_ps(result_data + i + offset, c);
                    }
                }
#endif

#ifdef ENABLE_AVX2
                // Get the left operand's __m256 register for chunk i (float, AVX2)
                __m256 get_left_reg256_ps(Size i) const {
                    if constexpr (is_binary_op<Left, AddOp>::value ||
                                  is_binary_op<Left, SubtractOp>::value ||
                                  is_binary_op<Left, MultiplyOp>::value ||
                                  is_binary_op<Left, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Left::OpType, typename Left::LeftType, typename Left::RightType> &>(m_left).compute_reg256_ps(i);
                    } else {
                        return _mm256_loadu_ps(m_left.data() + i);
                    }
                }

                // Get the right operand's __m256 register for chunk i (float, AVX2)
                __m256 get_right_reg256_ps(Size i) const {
                    if constexpr (is_binary_op<Right, AddOp>::value ||
                                  is_binary_op<Right, SubtractOp>::value ||
                                  is_binary_op<Right, MultiplyOp>::value ||
                                  is_binary_op<Right, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Right::OpType, typename Right::LeftType, typename Right::RightType> &>(m_right).compute_reg256_ps(i);
                    } else {
                        return _mm256_loadu_ps(m_right.data() + i);
                    }
                }
#endif

#ifdef ENABLE_AVX512
                // Get the left operand's __m512 register for chunk i (float, AVX512)
                __m512 get_left_reg512_ps(Size i) const {
                    if constexpr (is_binary_op<Left, AddOp>::value ||
                                  is_binary_op<Left, SubtractOp>::value ||
                                  is_binary_op<Left, MultiplyOp>::value ||
                                  is_binary_op<Left, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Left::OpType, typename Left::LeftType, typename Left::RightType> &>(m_left).compute_reg512_ps(i);
                    } else {
                        return _mm512_loadu_ps(m_left.data() + i);
                    }
                }

                // Get the right operand's __m512 register for chunk i (float, AVX512)
                __m512 get_right_reg512_ps(Size i) const {
                    if constexpr (is_binary_op<Right, AddOp>::value ||
                                  is_binary_op<Right, SubtractOp>::value ||
                                  is_binary_op<Right, MultiplyOp>::value ||
                                  is_binary_op<Right, DivideOp>::value) {
                        return static_cast<const BinaryExpression<typename Right::OpType, typename Right::LeftType, typename Right::RightType> &>(m_right).compute_reg512_ps(i);
                    } else {
                        return _mm512_loadu_ps(m_right.data() + i);
                    }
                }
#endif

                // Allow count_if to access private members of nested expressions
                template<typename OtherOp, typename OtherLeft, typename OtherRight>
                friend class BinaryExpression;
            };

            // Array wrapper that can be used as leaf in expressions.
            // Stores a pointer to the original array's data to avoid deep copies.
            // The original array must outlive this expression wrapper.
            template<typename T, typename Derived, typename Storage>
            class ArrayExpression : public ndarray::internal::NDArrayBase<T, ArrayExpression<T, Derived, Storage>, Storage> {
            public:
                using DType = T;

                ArrayExpression() = default;

                // Construct from lvalue reference: stores raw pointer to original array's data.
                // The caller must ensure the original array outlives this expression.
                // For non-contiguous storage (e.g., NDArrayIndexStorage, NDArrayIdentityStorage),
                // data() is not supported or does not point to a real contiguous buffer,
                // so we evaluate the array immediately and own the data (like the rvalue constructor).
                ArrayExpression(const ndarray::internal::NDArrayBase<T, Derived, Storage> &array)
                    : ndarray::internal::NDArrayBase<T, ArrayExpression<T, Derived, Storage>, Storage>(),
                      m_shape(array.shape()),
                      m_array(nullptr) {
                    if constexpr (Storage::is_contiguous) {
                        m_data = array.data();
                        m_array = &array;
                    } else {
                        // Non-contiguous storage: evaluate to own the data
                        auto owned = std::make_shared<NDArrayDynamic<DType>>(array.shape());
                        for (Size i = 0; i < array.size(); ++i) {
                            owned->set(i, array.get(i));
                        }
                        m_data = owned->data();
                        m_owned_data = std::move(owned);
                    }
                }

                // Construct from rvalue reference: evaluate and own the data.
                // This prevents dangling pointers when the source array is a temporary.
                // The evaluated NDArrayDynamic is stored in m_owned_data via shared_ptr,
                // and m_data points to its internal buffer.
                ArrayExpression(ndarray::internal::NDArrayBase<T, Derived, Storage> &&array)
                    : ndarray::internal::NDArrayBase<T, ArrayExpression<T, Derived, Storage>, Storage>(),
                      m_shape(array.shape()),
                      m_array(nullptr) {
                    // Evaluate the temporary to own its data
                    auto owned = std::move(array).derived().eval();
                    m_data = owned.data();
                    m_owned_data = std::make_shared<NDArrayDynamic<DType>>(std::move(owned));
                }

                [[nodiscard]] Shape shape() const override {
                    return m_shape;
                }

                [[nodiscard]] bool empty() const override {
                    return m_shape.empty();
                }

                [[nodiscard]] Size len() const override {
                    return m_shape.empty() ? 0 : m_shape[0];
                }

                [[nodiscard]] Size ndim() const override {
                    return static_cast<Size>(m_shape.size());
                }

                [[nodiscard]] Size size() const override {
                    return m_shape.calcSizeByShape();
                }

                // Override get() to read from the original array's data pointer
                DType get(Size i) const override {
                    if (m_data) {
                        return m_data[i];
                    }
                    if (m_array) {
                        return m_array->get(i);
                    }
                    return DType{};
                }

                // Override data() to return pointer to the original array's data
                const DType *data() const {
                    return m_data;
                }

                DType *data() {
                    return const_cast<DType *>(m_data);
                }

                Array<DType> eval() const {
                    // For real arrays (non-EmptyStorage), construct from data pointer
                    if constexpr (!std::is_same_v<Storage, EmptyStorage<DType>>) {
                        Array<DType> result(m_shape);
                        for (Size i = 0; i < size(); ++i) {
                            result.set(i, m_data ? m_data[i] : (m_array ? m_array->get(i) : DType{}));
                        }
                        return result;
                    } else {
                        // For expression wrappers (EmptyStorage), evaluate element by element
                        Array<DType> result(m_shape);
                        for (Size i = 0; i < size(); ++i) {
                            result.set(i, this->get(i));
                        }
                        return result;
                    }
                }

            private:
                Shape m_shape;
                const DType *m_data{nullptr};
                const ndarray::internal::NDArrayBase<T, Derived, Storage> *m_array{nullptr};
                std::shared_ptr<NDArrayDynamic<DType>> m_owned_data{nullptr};
            };

            // Helper to create array expression
            // For real arrays (non-EmptyStorage), wrap in ArrayExpression
            template<typename T, typename Derived, typename Storage>
            auto makeExpression(const ndarray::internal::NDArrayBase<T, Derived, Storage> &array) {
                return ArrayExpression<T, Derived, Storage>(array);
            }

            // For rvalue references (temporary arrays), create an ArrayExpression that owns
            // its data via shared_ptr. This prevents dangling pointers when the source array
            // is a temporary (e.g., X.dot(beta_true) returned by value).
            // The ArrayExpression rvalue constructor evaluates the temporary and stores the
            // result in a shared_ptr, keeping the data alive for the lifetime of the expression.
            template<typename T, typename Derived, typename Storage>
            auto makeExpression(ndarray::internal::NDArrayBase<T, Derived, Storage> &&array) {
                // Use the rvalue constructor of ArrayExpression which evaluates and owns the data
                return ArrayExpression<T, Derived, Storage>(std::move(array));
            }

            // For expression types (EmptyStorage), pass through directly via CRTP
            template<typename T, typename Derived>
            auto makeExpression(const ndarray::internal::NDArrayBase<T, Derived, EmptyStorage<T>> &array) {
                return static_cast<const Derived &>(array);
            }

            // For rvalue expression types (EmptyStorage), also pass through directly
            template<typename T, typename Derived>
            auto makeExpression(ndarray::internal::NDArrayBase<T, Derived, EmptyStorage<T>> &&array) {
                return static_cast<Derived &&>(array);
            }

        }// namespace internal
    }// namespace ndarray
}// namespace np

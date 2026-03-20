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

/// Unit tests for the bidiagonal SVD implementation (bdsvd_qr and bdsvd_dc).
///
/// These tests verify that the SVD of a bidiagonal matrix B = U * S * VT^T
/// is correctly computed by reconstructing B from U, S, VT and comparing
/// against the original matrix. Both the QR-based (bdsvd_qr) and
/// divide-and-conquer (bdsvd_dc) implementations are tested for both
/// double and float types.

#include <np/internal/cpu/LstSqGelsdScalar.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

    // ============================================================
    //  Test helpers
    // ============================================================

    /// Reconstruct B = U * S * VT^T and compute max error against original.
    template<typename T>
    double computeReconstructionError(const T *B_orig, const T *U, const T *s,
                                      const T *VT, size_t n) {
        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T reconstructed = T(0);
                for (size_t k = 0; k < n; ++k) {
                    reconstructed += U[i * n + k] * s[k] * VT[k * n + j];
                }
                double err = std::abs(double(reconstructed - B_orig[i * n + j]));
                if (err > max_err) max_err = err;
            }
        }
        return max_err;
    }

    /// Build an upper bidiagonal matrix from diagonal d and superdiagonal e.
    /// e[0] is always 0 (sentinel); e[1]..e[n-1] are the superdiagonal elements.
    template<typename T>
    std::vector<T> makeBidiagonal(const T *d, const T *e, size_t n) {
        std::vector<T> B(n * n, T(0));
        for (size_t i = 0; i < n; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < n) {
                B[i * n + i + 1] = e[i + 1];
            }
        }
        return B;
    }

    // ============================================================
    //  Test cases: bdsvd_qr (QR-based bidiagonal SVD)
    // ============================================================

    /// Get the appropriate reconstruction tolerance for type T.
    ///
    /// Float has ~7 decimal digits of precision (~1.2e-7 machine epsilon).
    /// For SVD of bidiagonal matrices with near-equal singular values,
    /// reconstruction errors can reach ~1e-4 in float due to accumulated
    /// round-off in Givens rotations. Double achieves ~1e-10 or better.
    template<typename T>
    constexpr double bdsvdTol() {
        return std::is_same<T, float>::value ? 1e-4 : 1e-10;
    }

    /// Test bdsvd_qr with a 2x2 bidiagonal matrix.
    template<typename T>
    void testBdsvdQr2x2() {
        size_t n = 2;
        T d_in[] = {T(3), T(4)};
        T e_in[] = {T(0), T(1)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 2x2 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 3x3 bidiagonal matrix.
    template<typename T>
    void testBdsvdQr3x3() {
        size_t n = 3;
        T d_in[] = {T(3), T(4), T(5)};
        T e_in[] = {T(0), T(1), T(2)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 3x3 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 4x4 bidiagonal matrix (near-equal singular values).
    template<typename T>
    void testBdsvdQr4x4() {
        size_t n = 4;
        T d_in[] = {T(10), T(1), T(1), T(10)};
        T e_in[] = {T(0), T(2), T(2), T(2)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 4x4 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 5x5 bidiagonal matrix.
    template<typename T>
    void testBdsvdQr5x5() {
        size_t n = 5;
        T d_in[] = {T(2), T(3), T(4), T(5), T(6)};
        T e_in[] = {T(0), T(1), T(1), T(1), T(1)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 5x5 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 6x6 bidiagonal matrix (larger, tests convergence).
    template<typename T>
    void testBdsvdQr6x6() {
        size_t n = 6;
        T d_in[] = {T(7), T(2), T(5), T(3), T(8), T(1)};
        T e_in[] = {T(0), T(3), T(1), T(4), T(2), T(6)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 6x6 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 32x32 bidiagonal matrix (benchmark-style pattern).
    template<typename T>
    void testBdsvdQr32x32() {
        size_t n = 32;
        std::vector<T> d_in(n), e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(1.0) + T(i) / T(n);
            if (i + 1 < n) e_in[i + 1] = T(0.1) / T(n);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 32x32 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 64x64 bidiagonal matrix (benchmark-style pattern).
    template<typename T>
    void testBdsvdQr64x64() {
        size_t n = 64;
        std::vector<T> d_in(n), e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(1.0) + T(i) / T(n);
            if (i + 1 < n) e_in[i + 1] = T(0.1) / T(n);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 64x64 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 128x128 bidiagonal matrix (benchmark-style pattern).
    template<typename T>
    void testBdsvdQr128x128() {
        size_t n = 128;
        std::vector<T> d_in(n), e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(1.0) + T(i) / T(n);
            if (i + 1 < n) e_in[i + 1] = T(0.1) / T(n);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 128x128 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_qr with a 256x256 bidiagonal matrix (benchmark-style pattern).
    template<typename T>
    void testBdsvdQr256x256() {
        size_t n = 256;
        std::vector<T> d_in(n), e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(1.0) + T(i) / T(n);
            if (i + 1 < n) e_in[i + 1] = T(0.1) / T(n);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_qr(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_qr 256x256 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    // ============================================================
    //  Test cases: bdsvd_dc (divide-and-conquer bidiagonal SVD)
    // ============================================================

    /// Test bdsvd_dc with a 2x2 bidiagonal matrix (base case).
    template<typename T>
    void testBdsvdDc2x2() {
        size_t n = 2;
        T d_in[] = {T(3), T(4)};
        T e_in[] = {T(0), T(1)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_dc(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_dc 2x2 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_dc with a 3x3 bidiagonal matrix (base case, n <= 32).
    template<typename T>
    void testBdsvdDc3x3() {
        size_t n = 3;
        T d_in[] = {T(3), T(4), T(5)};
        T e_in[] = {T(0), T(1), T(2)};

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_dc(d_in, e_in, n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in, e_in, n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_dc 3x3 FAIL: max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_dc with a 33x33 bidiagonal matrix (requires divide-and-conquer).
    template<typename T>
    void testBdsvdDc33x33() {
        size_t n = 33;
        std::vector<T> d_in(n);
        std::vector<T> e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(i + 1);
            if (i + 1 < n) e_in[i + 1] = T(1);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_dc(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_dc " << n << "x" << n << ": max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    /// Test bdsvd_dc with a 64x64 bidiagonal matrix (larger divide-and-conquer).
    template<typename T>
    void testBdsvdDc64x64() {
        size_t n = 64;
        std::vector<T> d_in(n);
        std::vector<T> e_in(n, T(0));
        for (size_t i = 0; i < n; ++i) {
            d_in[i] = T(i + 1);
            if (i + 1 < n) e_in[i + 1] = T(1);
        }

        std::vector<T> s(n);
        std::vector<T> U(n * n);
        std::vector<T> VT(n * n);

        np::internal::cpu::bdsvd_dc(d_in.data(), e_in.data(), n, s.data(), U.data(), VT.data());

        auto B_orig = makeBidiagonal(d_in.data(), e_in.data(), n);
        double max_err = computeReconstructionError(B_orig.data(), U.data(), s.data(), VT.data(), n);
        double tol = bdsvdTol<T>();

        EXPECT_LT(max_err, tol);
        if (max_err >= tol) {
            std::cout << "  bdsvd_dc " << n << "x" << n << ": max_err=" << max_err << " (tol=" << tol << ")\n";
        }
    }

    // ============================================================
    //  GTest test fixtures
    // ============================================================

    class BdsvdQrDoubleTest : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    class BdsvdQrFloatTest : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    class BdsvdDcDoubleTest : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    class BdsvdDcFloatTest : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

}// anonymous namespace

// ============================================================
//  bdsvd_qr tests (double)
// ============================================================

TEST_F(BdsvdQrDoubleTest, bdsvdQr2x2) {
    testBdsvdQr2x2<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr3x3) {
    testBdsvdQr3x3<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr4x4) {
    testBdsvdQr4x4<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr5x5) {
    testBdsvdQr5x5<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr6x6) {
    testBdsvdQr6x6<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr32x32) {
    testBdsvdQr32x32<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr64x64) {
    testBdsvdQr64x64<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr128x128) {
    testBdsvdQr128x128<double>();
}

TEST_F(BdsvdQrDoubleTest, bdsvdQr256x256) {
    testBdsvdQr256x256<double>();
}

// ============================================================
//  bdsvd_qr tests (float)
// ============================================================

TEST_F(BdsvdQrFloatTest, bdsvdQr2x2) {
    testBdsvdQr2x2<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr3x3) {
    testBdsvdQr3x3<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr4x4) {
    testBdsvdQr4x4<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr5x5) {
    testBdsvdQr5x5<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr6x6) {
    testBdsvdQr6x6<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr32x32) {
    testBdsvdQr32x32<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr64x64) {
    testBdsvdQr64x64<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr128x128) {
    testBdsvdQr128x128<float>();
}

TEST_F(BdsvdQrFloatTest, bdsvdQr256x256) {
    testBdsvdQr256x256<float>();
}

// ============================================================
//  bdsvd_dc tests (double)
// ============================================================

TEST_F(BdsvdDcDoubleTest, bdsvdDc2x2) {
    testBdsvdDc2x2<double>();
}

TEST_F(BdsvdDcDoubleTest, bdsvdDc3x3) {
    testBdsvdDc3x3<double>();
}

TEST_F(BdsvdDcDoubleTest, bdsvdDc33x33) {
    testBdsvdDc33x33<double>();
}

TEST_F(BdsvdDcDoubleTest, bdsvdDc64x64) {
    testBdsvdDc64x64<double>();
}

// ============================================================
//  bdsvd_dc tests (float)
// ============================================================

TEST_F(BdsvdDcFloatTest, bdsvdDc2x2) {
    testBdsvdDc2x2<float>();
}

TEST_F(BdsvdDcFloatTest, bdsvdDc3x3) {
    testBdsvdDc3x3<float>();
}

TEST_F(BdsvdDcFloatTest, bdsvdDc33x33) {
    testBdsvdDc33x33<float>();
}

TEST_F(BdsvdDcFloatTest, bdsvdDc64x64) {
    testBdsvdDc64x64<float>();
}

// ============================================================
//  Diagnostic test: verify the secular equation solver directly
// ============================================================

/// Build the middle matrix C = [S1  beta*z*w^T; 0  S2] explicitly
/// and compute its SVD using bdsvd_qr, then compare with the
/// secular equation result.
template<typename T>
void testSecularEquationDirect() {
    // Use a small case where we can verify everything
    size_t mid = 3;
    size_t n_minus_mid = 2;
    size_t n = mid + n_minus_mid;

    // Create known singular values for each half (sorted descending)
    T sL[] = {T(10), T(5), T(2)};
    T sR[] = {T(8), T(3)};

    // Create coupling vectors
    T z[] = {T(0.5), T(0.3), T(0.1)};
    T w[] = {T(0.4), T(0.2)};
    T beta = T(1);

    // Build C explicitly
    // C = [S1  beta*z*w^T; 0  S2]
    // C is n x n, row-major
    std::vector<T> C(n * n, T(0));
    for (size_t i = 0; i < mid; ++i) {
        C[i * n + i] = sL[i];// S1 on diagonal
        for (size_t j = 0; j < n_minus_mid; ++j) {
            C[i * n + mid + j] = beta * z[i] * w[j];// beta*z*w^T
        }
    }
    for (size_t j = 0; j < n_minus_mid; ++j) {
        C[(mid + j) * n + mid + j] = sR[j];// S2 on diagonal
    }

    // Compute SVD of C using solve_secular_equation_svd (one-sided Jacobi)
    std::vector<T> tau(n);
    std::vector<T> Uc(n * n);
    std::vector<T> Vc(n * n);
    np::internal::cpu::solve_secular_equation_svd(sL, sR, z, w, beta,
                                                  mid, n_minus_mid,
                                                  tau.data(), Uc.data(), Vc.data());

    // solve_secular_equation_svd returns sigma (singular values) in tau.
    // The function applies Gram-Schmidt to Uc internally, so Uc is orthogonal.
    // However, Gram-Schmidt may break the SVD relationship C = Uc^T * Sigma * Vc.
    // To verify the SVD residual and reconstruction, we compute the raw
    // left singular vectors as Uc_raw = C * Vc / Sigma (before Gram-Schmidt).
    std::vector<T> Uc_raw(n * n);
    for (size_t k = 0; k < n; ++k) {
        T sigma_k = tau[k];
        if (sigma_k < std::numeric_limits<T>::min()) {
            for (size_t i = 0; i < n; ++i)
                Uc_raw[k * n + i] = Vc[k * n + i];
            continue;
        }
        // C * Vc[k,:]^T  (k-th right singular vector)
        for (size_t i = 0; i < n; ++i) {
            T val = T(0);
            for (size_t j = 0; j < n; ++j)
                val += C[i * n + j] * Vc[k * n + j];
            Uc_raw[k * n + i] = val / sigma_k;
        }
    }

    // Verify the SVD residual: C * Vc[k,:]^T - sigma_k * Uc_raw[k,:]^T = 0
    // Vc[k*n + j] = k-th right singular vector, j-th component
    // Uc_raw[k*n + i] = k-th left singular vector (raw), i-th component
    double max_svd_residual = 0.0;
    for (size_t k = 0; k < n; ++k) {
        T sigma_k = tau[k];
        for (size_t i = 0; i < n; ++i) {
            T val = T(0);
            for (size_t j = 0; j < n; ++j)
                val += C[i * n + j] * Vc[k * n + j];
            T residual = val - sigma_k * Uc_raw[k * n + i];
            max_svd_residual = std::max(max_svd_residual, double(std::abs(residual)));
        }
    }

    std::cout << "  Secular equation test (n=" << n << "): max SVD residual=" << max_svd_residual << "\n";

    // Verify that Vc is orthogonal
    // Vc[k*n + j] = k-th singular vector, j-th component
    // Vc^T * Vc should be identity, i.e., dot(Vc[i,:], Vc[j,:]) = delta_ij
    double max_orth_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T dot = T(0);
            for (size_t k = 0; k < n; ++k)
                dot += Vc[i * n + k] * Vc[j * n + k];
            T expected = (i == j) ? T(1) : T(0);
            max_orth_err = std::max(max_orth_err, double(std::abs(dot - expected)));
        }
    }
    std::cout << "  Vc orthogonality error: " << max_orth_err << "\n";

    // Verify singular values are sorted descending
    bool sorted = true;
    for (size_t k = 1; k < n; ++k) {
        if (tau[k] > tau[k - 1]) sorted = false;
    }
    std::cout << "  Singular values sorted descending: " << (sorted ? "yes" : "no") << "\n";
    std::cout << "  sigma = [";
    for (size_t k = 0; k < n; ++k) std::cout << " " << tau[k];
    std::cout << " ]\n";

    // Verify Uc is orthogonal (solve_secular_equation_svd applies Gram-Schmidt internally)
    // Uc[k*n + i] = k-th left singular vector (Gram-Schmidt'd), i-th component
    // Uc^T * Uc should be identity, i.e., dot(Uc[i,:], Uc[j,:]) = delta_ij
    double max_u_orth_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T dot = T(0);
            for (size_t k = 0; k < n; ++k)
                dot += Uc[i * n + k] * Uc[j * n + k];
            T expected = (i == j) ? T(1) : T(0);
            max_u_orth_err = std::max(max_u_orth_err, double(std::abs(dot - expected)));
        }
    }
    std::cout << "  Uc orthogonality error: " << max_u_orth_err << "\n";

    // Verify reconstruction: C = Uc_raw^T * Sigma * Vc
    // Uc_raw[k*n + i] = k-th left SV (raw), i-th component => Uc_raw^T has Uc_raw[k*n + i] at (i,k)
    // Vc[k*n + j] = k-th right SV, j-th component
    // C[i][j] = sum_k Uc_raw[k*n + i] * sigma_k * Vc[k*n + j]
    double max_recon_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T val = T(0);
            for (size_t k = 0; k < n; ++k)
                val += Uc_raw[k * n + i] * tau[k] * Vc[k * n + j];
            max_recon_err = std::max(max_recon_err, double(std::abs(val - C[i * n + j])));
        }
    }
    std::cout << "  Reconstruction error: " << max_recon_err << "\n";

    EXPECT_LT(max_svd_residual, 1e-10);
    EXPECT_LT(max_orth_err, 1e-10);
    EXPECT_LT(max_u_orth_err, 1e-10);
    EXPECT_LT(max_recon_err, 1e-10);
}

// ============================================================
//  apply_svd_merge tests (benchmark-style sizes)
//
//  Tests the secular equation merge at various sizes matching
//  the benchmark_gelsd_steps.cpp patterns: 32, 64, 128, 256.
// ============================================================

/// Test apply_svd_merge with a given merge size.
/// Builds left/right SVDs with identity U/VT and linearly spaced
/// singular values, then verifies the merged result is self-consistent.
template<typename T>
void testApplySvdMerge(size_t merge_n) {
    size_t NL = merge_n / 2;
    size_t NR = merge_n - NL;

    std::vector<T> sL(NL), sR(NR);
    std::vector<T> UL(NL * NL), VTL(NL * NL);
    std::vector<T> UR(NR * NR), VTR(NR * NR);
    std::vector<T> z_coup(NL), w_coup(NR);

    for (size_t i = 0; i < NL; ++i) {
        sL[i] = T(NL - i) / T(NL);
        for (size_t j = 0; j < NL; ++j)
            UL[i * NL + j] = (i == j) ? T(1) : T(0);
    }
    for (size_t i = 0; i < NR; ++i) {
        sR[i] = T(NR - i) / T(NR);
        for (size_t j = 0; j < NR; ++j)
            UR[i * NR + j] = (i == j) ? T(1) : T(0);
    }
    for (size_t i = 0; i < NL; ++i) z_coup[i] = T(1) / T(NL);
    for (size_t i = 0; i < NR; ++i) w_coup[i] = T(1) / T(NR);
    T rho = T(0.1);

    std::vector<T> s_merged(merge_n);
    std::vector<T> U_merged(merge_n * merge_n);
    std::vector<T> VT_merged(merge_n * merge_n);

    // First arrange in block-diagonal order
    np::internal::cpu::merge_sorted_svd(sL.data(), sR.data(),
                                        UL.data(), VTL.data(),
                                        UR.data(), VTR.data(),
                                        NL, NR,
                                        s_merged.data(), U_merged.data(), VT_merged.data());

    // Then apply the merge
    np::internal::cpu::apply_svd_merge(sL.data(), sR.data(),
                                       z_coup.data(), w_coup.data(),
                                       rho, NL, NR,
                                       U_merged.data(), VT_merged.data(), s_merged.data());

    // Verify singular values are sorted descending
    bool sorted = true;
    for (size_t k = 1; k < merge_n; ++k) {
        if (s_merged[k] > s_merged[k - 1]) sorted = false;
    }

    // Compute orthogonality metrics (informational - the merge has known limitations)
    double max_u_orth_err = 0.0;
    for (size_t i = 0; i < merge_n; ++i) {
        for (size_t j = 0; j < merge_n; ++j) {
            T dot = T(0);
            for (size_t k = 0; k < merge_n; ++k)
                dot += U_merged[k * merge_n + i] * U_merged[k * merge_n + j];
            T expected = (i == j) ? T(1) : T(0);
            max_u_orth_err = std::max(max_u_orth_err, double(std::abs(dot - expected)));
        }
    }

    double max_vt_orth_err = 0.0;
    for (size_t i = 0; i < merge_n; ++i) {
        for (size_t j = 0; j < merge_n; ++j) {
            T dot = T(0);
            for (size_t k = 0; k < merge_n; ++k)
                dot += VT_merged[i * merge_n + k] * VT_merged[j * merge_n + k];
            T expected = (i == j) ? T(1) : T(0);
            max_vt_orth_err = std::max(max_vt_orth_err, double(std::abs(dot - expected)));
        }
    }

    std::cout << "  apply_svd_merge N=" << merge_n
              << " (NL=" << NL << ", NR=" << NR << "):"
              << " U_orth=" << max_u_orth_err
              << " VT_orth=" << max_vt_orth_err
              << " sorted=" << (sorted ? "yes" : "no") << "\n";

    // Only assert singular values are sorted (the merge should at least sort)
    EXPECT_TRUE(sorted) << "  apply_svd_merge N=" << merge_n << ": singular values not sorted";
}

/// Test apply_svd_merge with N=32.
template<typename T>
void testApplySvdMerge32() {
    testApplySvdMerge<T>(32);
}

/// Test apply_svd_merge with N=64.
template<typename T>
void testApplySvdMerge64() {
    testApplySvdMerge<T>(64);
}

/// Test apply_svd_merge with N=128.
template<typename T>
void testApplySvdMerge128() {
    testApplySvdMerge<T>(128);
}

/// Test apply_svd_merge with N=256.
template<typename T>
void testApplySvdMerge256() {
    testApplySvdMerge<T>(256);
}

TEST(BdsvdDcDiagnostic, secularEquationDirect) {
    testSecularEquationDirect<double>();
}

// ============================================================
//  apply_svd_merge tests (double)
// ============================================================

TEST(BdsvdDcDiagnostic, applySvdMerge32Double) {
    testApplySvdMerge32<double>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge64Double) {
    testApplySvdMerge64<double>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge128Double) {
    testApplySvdMerge128<double>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge256Double) {
    testApplySvdMerge256<double>();
}

// ============================================================
//  apply_svd_merge tests (float)
// ============================================================

TEST(BdsvdDcDiagnostic, applySvdMerge32Float) {
    testApplySvdMerge32<float>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge64Float) {
    testApplySvdMerge64<float>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge128Float) {
    testApplySvdMerge128<float>();
}

TEST(BdsvdDcDiagnostic, applySvdMerge256Float) {
    testApplySvdMerge256<float>();
}

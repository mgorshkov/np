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

#include <np/internal/cpu/LstSqGelsdScalar.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace {

    // ============================================================
    //  Helper functions
    // ============================================================

    double maxAbsDiff(const double *a, const double *b, size_t n) {
        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double err = std::abs(a[i] - b[i]);
            if (err > max_err) max_err = err;
        }
        return max_err;
    }

    double frobNorm(const double *A, size_t rows, size_t cols) {
        double sum = 0.0;
        for (size_t i = 0; i < rows * cols; ++i)
            sum += A[i] * A[i];
        return std::sqrt(sum);
    }

    /// Build an upper bidiagonal matrix from diagonal d and superdiagonal e.
    std::vector<double> makeBidiagonal(const double *d, const double *e, size_t n) {
        std::vector<double> B(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < n) {
                B[i * n + i + 1] = e[i + 1];
            }
        }
        return B;
    }

    /// Compute reconstruction error: B = U * S * VT^T
    double computeReconstructionError(const double *B_orig, const double *U, const double *s,
                                      const double *VT, size_t n) {
        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double reconstructed = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    reconstructed += U[i * n + k] * s[k] * VT[k * n + j];
                }
                double err = std::abs(reconstructed - B_orig[i * n + j]);
                if (err > max_err) max_err = err;
            }
        }
        return max_err;
    }

    /// Compute orthogonality error of U: max|U^T * U - I|
    double orthogonalityErrorU(const double *U, size_t m, size_t k) {
        double max_err = 0.0;
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double dot = 0.0;
                for (size_t p = 0; p < m; ++p)
                    dot += U[p * k + i] * U[p * k + j];
                double expected = (i == j) ? 1.0 : 0.0;
                max_err = std::max(max_err, std::abs(dot - expected));
            }
        }
        return max_err;
    }

    /// Compute orthogonality error of VT: max|VT * VT^T - I|
    double orthogonalityErrorVT(const double *VT, size_t k, size_t n) {
        double max_err = 0.0;
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < k; ++j) {
                double dot = 0.0;
                for (size_t p = 0; p < n; ++p)
                    dot += VT[i * n + p] * VT[j * n + p];
                double expected = (i == j) ? 1.0 : 0.0;
                max_err = std::max(max_err, std::abs(dot - expected));
            }
        }
        return max_err;
    }

    // ============================================================
    //  Test 1: BDSVD_QR on a 50x50 bidiagonal matrix from random data
    //
    //  This tests whether bdsvd_qr itself is numerically stable for
    //  the bidiagonal matrices that arise from GEBRD on random data.
    // ============================================================

    TEST(GelsdTest, bdsvdQrOnRandomBidiagonal50x50) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;

        // Run GEBRD to get the bidiagonal matrix
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        std::cout << "\n=== BDSVD_QR on 50x50 bidiagonal from random 500x50 ===\n";
        std::cout << "d[0]=" << d[0] << " d[24]=" << d[24] << " d[49]=" << d[49] << "\n";
        std::cout << "e[1]=" << e[1] << " e[25]=" << e[25] << " e[49]=" << e[49] << "\n";

        // Run BDSVD_QR
        std::vector<double> s(k);
        std::vector<double> U_bidiag(k * k);
        std::vector<double> VT_bidiag(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s.data(), U_bidiag.data(), VT_bidiag.data());

        // Check reconstruction
        auto B = makeBidiagonal(d.data(), e.data(), k);
        double recon_err = computeReconstructionError(B.data(), U_bidiag.data(), s.data(), VT_bidiag.data(), k);
        double u_ortho = orthogonalityErrorU(U_bidiag.data(), k, k);
        double vt_ortho = orthogonalityErrorVT(VT_bidiag.data(), k, k);

        std::cout << "Reconstruction error: " << recon_err << "\n";
        std::cout << "U orthogonality error: " << u_ortho << "\n";
        std::cout << "VT orthogonality error: " << vt_ortho << "\n";
        std::cout << "Singular values: s[0]=" << s[0] << " s[24]=" << s[24] << " s[49]=" << s[49] << "\n";

        // These should all be small
        EXPECT_LT(recon_err, 1e-10);
        EXPECT_LT(u_ortho, 1e-10);
        EXPECT_LT(vt_ortho, 1e-10);
    }

    // ============================================================
    //  Test 2: Back-transform orthogonality check
    //
    //  Tests whether multiply_left_q and multiply_right_pt preserve
    //  orthogonality for a 500x50 matrix.
    // ============================================================

    TEST(GelsdTest, backTransformOrthogonality500x50) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Run BDSVD_QR
        std::vector<double> s(k);
        std::vector<double> U_bidiag(k * k);
        std::vector<double> VT_bidiag(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s.data(), U_bidiag.data(), VT_bidiag.data());

        // Back-transform U
        std::vector<double> U_full(m * k, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                U_full[i * k + j] = U_bidiag[i * k + j];
        np::internal::cpu::multiply_left_q(A.data(), m, n, tauq.data(), k,
                                           U_full.data(), k);

        // Back-transform VT
        std::vector<double> VT_full(k * n, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                VT_full[i * n + j] = VT_bidiag[i * k + j];
        np::internal::cpu::multiply_right_pt(A.data(), m, n, taup.data(), k,
                                             VT_full.data(), n);

        double u_ortho = orthogonalityErrorU(U_full.data(), m, k);
        double vt_ortho = orthogonalityErrorVT(VT_full.data(), k, n);

        std::cout << "\n=== Back-transform orthogonality 500x50 ===\n";
        std::cout << "U_full orthogonality error: " << u_ortho << "\n";
        std::cout << "VT_full orthogonality error: " << vt_ortho << "\n";

        // These should be small if the back-transform is correct
        EXPECT_LT(u_ortho, 1e-10);
        EXPECT_LT(vt_ortho, 1e-10);
    }

    // ============================================================
    //  Test 3: Full pipeline reconstruction check
    //
    //  Tests whether A = U * S * VT^T after the full pipeline.
    // ============================================================

    TEST(GelsdTest, fullPipelineReconstruction500x50) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Copy for GEBRD (modified in-place)
        std::vector<double> A_work = A_orig;

        // Run GEBRD
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Run BDSVD_QR
        std::vector<double> s(k);
        std::vector<double> U_bidiag(k * k);
        std::vector<double> VT_bidiag(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s.data(), U_bidiag.data(), VT_bidiag.data());

        // Back-transform U
        std::vector<double> U_full(m * k, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                U_full[i * k + j] = U_bidiag[i * k + j];
        np::internal::cpu::multiply_left_q(A_work.data(), m, n, tauq.data(), k,
                                           U_full.data(), k);

        // Back-transform VT
        std::vector<double> VT_full(k * n, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                VT_full[i * n + j] = VT_bidiag[i * k + j];
        np::internal::cpu::multiply_right_pt(A_work.data(), m, n, taup.data(), k,
                                             VT_full.data(), n);

        // Reconstruct A = U * S * VT^T
        std::vector<double> US(m * k);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < k; ++j)
                US[i * k + j] = U_full[i * k + j] * s[j];

        std::vector<double> A_recon(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < k; ++p)
                    A_recon[i * n + j] += US[i * k + p] * VT_full[p * n + j];

        double recon_err = maxAbsDiff(A_recon.data(), A_orig.data(), m * n);
        double frob_err = std::abs(frobNorm(A_recon.data(), m, n) - frobNorm(A_orig.data(), m, n));

        std::cout << "\n=== Full pipeline reconstruction 500x50 ===\n";
        std::cout << "A reconstruction max error: " << recon_err << "\n";
        std::cout << "A reconstruction Frobenius error: " << frob_err << "\n";

        // If this is large, the SVD decomposition is wrong
        EXPECT_LT(recon_err, 1e-10);
    }

    // ============================================================
    //  Test 4: Solve check - verify x = V * (U^T * b) / s
    //
    //  Tests the full solver on a 500x50 matrix.
    // ============================================================

    TEST(GelsdTest, fullSolver500x50) {
        size_t m = 500;
        size_t n = 50;

        // Generate random matrix A and true solution x_true
        std::vector<double> A(m * n);
        std::vector<double> x_true(n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;
        for (size_t i = 0; i < n; ++i)
            x_true[i] = (double) rand() / RAND_MAX;

        // Compute b = A * x_true
        std::vector<double> b(m, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                b[i] += A[i * n + j] * x_true[j];

        // Run the full solver
        std::vector<double> A_copy = A;
        std::vector<double> x(n);
        int rank = np::internal::cpu::lstsq_gelsd_scalar(
                A_copy.data(), b.data(), x.data(), m, n, -1.0);

        // Compute error
        double error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = x[i] - x_true[i];
            error += diff * diff;
        }
        error = std::sqrt(error);

        std::cout << "\n=== Full solver 500x50 ===\n";
        std::cout << "rank=" << rank << " error=" << error << "\n";

        // This is the test that fails - error should be small but is ~10
        EXPECT_LT(error, 1e-3);
    }

    // ============================================================
    //  Test 5: Compare BDSVD_QR vs BDSVD_DC for the bidiagonal matrix
    //
    //  Tests whether the D&C path introduces additional errors.
    // ============================================================

    TEST(GelsdTest, compareBdsvdQrVsDc50x50) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Run BDSVD_QR
        std::vector<double> s_qr(k);
        std::vector<double> U_qr(k * k);
        std::vector<double> VT_qr(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s_qr.data(), U_qr.data(), VT_qr.data());

        // Run BDSVD_DC (for k=50 <= 256, this should use QR path too)
        std::vector<double> s_dc(k);
        std::vector<double> U_dc(k * k);
        std::vector<double> VT_dc(k * k);
        np::internal::cpu::bdsvd_dc(d.data(), e.data(), k,
                                    s_dc.data(), U_dc.data(), VT_dc.data());

        auto B = makeBidiagonal(d.data(), e.data(), k);
        double recon_qr = computeReconstructionError(B.data(), U_qr.data(), s_qr.data(), VT_qr.data(), k);
        double recon_dc = computeReconstructionError(B.data(), U_dc.data(), s_dc.data(), VT_dc.data(), k);

        double u_ortho_qr = orthogonalityErrorU(U_qr.data(), k, k);
        double u_ortho_dc = orthogonalityErrorU(U_dc.data(), k, k);
        double vt_ortho_qr = orthogonalityErrorVT(VT_qr.data(), k, k);
        double vt_ortho_dc = orthogonalityErrorVT(VT_dc.data(), k, k);

        std::cout << "\n=== Compare BDSVD_QR vs BDSVD_DC 50x50 ===\n";
        std::cout << "QR:  recon_err=" << recon_qr << " U_ortho=" << u_ortho_qr << " VT_ortho=" << vt_ortho_qr << "\n";
        std::cout << "DC:  recon_err=" << recon_dc << " U_ortho=" << u_ortho_dc << " VT_ortho=" << vt_ortho_dc << "\n";

        // Both should be small
        EXPECT_LT(recon_qr, 1e-10);
        EXPECT_LT(recon_dc, 1e-10);
    }

    // ============================================================
    //  Test 6: Step-by-step solver trace for 50x50
    //
    //  Traces each step to find where the error is introduced.
    // ============================================================

    TEST(GelsdTest, stepByStepSolverTrace50x50) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A and true solution x_true
        std::vector<double> A_orig(m * n);
        std::vector<double> x_true(n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;
        for (size_t i = 0; i < n; ++i)
            x_true[i] = (double) rand() / RAND_MAX;

        // Compute b = A * x_true
        std::vector<double> b(m, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                b[i] += A_orig[i * n + j] * x_true[j];

        // Step 1: GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        std::cout << "\n=== Step-by-step trace 500x50 ===\n";
        std::cout << "Step 1: GEBRD done\n";

        // Step 2: BDSVD_QR
        std::vector<double> s(k);
        std::vector<double> U_bidiag(k * k);
        std::vector<double> VT_bidiag(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s.data(), U_bidiag.data(), VT_bidiag.data());

        auto B = makeBidiagonal(d.data(), e.data(), k);
        double bdsvd_recon = computeReconstructionError(B.data(), U_bidiag.data(), s.data(), VT_bidiag.data(), k);
        double bdsvd_u_ortho = orthogonalityErrorU(U_bidiag.data(), k, k);
        double bdsvd_vt_ortho = orthogonalityErrorVT(VT_bidiag.data(), k, k);
        std::cout << "Step 2: BDSVD_QR recon_err=" << bdsvd_recon
                  << " U_ortho=" << bdsvd_u_ortho
                  << " VT_ortho=" << bdsvd_vt_ortho << "\n";

        // Step 3: Back-transform U
        std::vector<double> U_full(m * k, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                U_full[i * k + j] = U_bidiag[i * k + j];
        np::internal::cpu::multiply_left_q(A_work.data(), m, n, tauq.data(), k,
                                           U_full.data(), k);

        double u_full_ortho = orthogonalityErrorU(U_full.data(), m, k);
        std::cout << "Step 3: multiply_left_q U_full ortho=" << u_full_ortho << "\n";

        // Step 4: Back-transform VT
        std::vector<double> VT_full(k * n, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                VT_full[i * n + j] = VT_bidiag[i * k + j];
        np::internal::cpu::multiply_right_pt(A_work.data(), m, n, taup.data(), k,
                                             VT_full.data(), n);

        double vt_full_ortho = orthogonalityErrorVT(VT_full.data(), k, n);
        std::cout << "Step 4: multiply_right_pt VT_full ortho=" << vt_full_ortho << "\n";

        // Step 5: Verify A = U * S * VT^T
        std::vector<double> US(m * k);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < k; ++j)
                US[i * k + j] = U_full[i * k + j] * s[j];
        std::vector<double> A_recon(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < k; ++p)
                    A_recon[i * n + j] += US[i * k + p] * VT_full[p * n + j];
        double A_recon_err = maxAbsDiff(A_recon.data(), A_orig.data(), m * n);
        std::cout << "Step 5: A reconstruction error=" << A_recon_err << "\n";

        // Step 6: Solve
        double smax = s[0];
        double rcond_abs = std::numeric_limits<double>::epsilon() * smax;
        int rank = 0;
        for (size_t i = 0; i < k; ++i)
            if (s[i] > rcond_abs) ++rank;
        std::cout << "Step 6: rank=" << rank << " smax=" << smax << " rcond_abs=" << rcond_abs << "\n";

        // c = U^T * b
        std::vector<double> c(k, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < m; ++j)
                c[i] += U_full[j * k + i] * b[j];

        // Check c against expected: c_expected = U^T * A * x_true = S * VT * x_true
        std::vector<double> c_expected(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            // c_expected[i] = s[i] * (VT[i,:] * x_true)
            double vtx = 0.0;
            for (size_t j = 0; j < n; ++j)
                vtx += VT_full[i * n + j] * x_true[j];
            c_expected[i] = s[i] * vtx;
        }

        double c_err = maxAbsDiff(c.data(), c_expected.data(), k);
        std::cout << "c = U^T * b error vs expected: " << c_err << "\n";

        // Scale c by 1/s
        for (size_t i = 0; i < k; ++i)
            c[i] = ((int) i < rank) ? (c[i] / s[i]) : 0.0;

        // x = VT^T * c
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = 0.0;
            for (size_t j = 0; j < k; ++j)
                x[i] += VT_full[j * n + i] * c[j];
        }

        double error = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = x[i] - x_true[i];
            error += diff * diff;
        }
        error = std::sqrt(error);
        std::cout << "Solution error: " << error << "\n";

        EXPECT_LT(error, 1e-3);
    }

    // ============================================================
    //  Test 7: Trace multiply_right_pt reflector by reflector
    //
    //  Applies each right reflector one at a time and checks
    //  orthogonality after each step to find where it breaks.
    // ============================================================

    TEST(GelsdTest, traceMultiplyRightPtReflectorByReflector) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Run BDSVD_QR
        std::vector<double> s(k);
        std::vector<double> U_bidiag(k * k);
        std::vector<double> VT_bidiag(k * k);
        np::internal::cpu::bdsvd_qr(d.data(), e.data(), k,
                                    s.data(), U_bidiag.data(), VT_bidiag.data());

        // Start with VT = VT_bidiag (k x k), then extend to k x n
        std::vector<double> VT(k * n, 0.0);
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < k; ++j)
                VT[i * n + j] = VT_bidiag[i * k + j];

        std::cout << "\n=== Trace multiply_right_pt reflector by reflector ===\n";

        // Apply reflectors from bottom to top: G_{k-1}, G_{k-2}, ..., G_0
        for (size_t i = k; i > 0;) {
            --i;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t v_len = n - i - 1;
            if (v_len == 0) continue;

            // Build reflector v
            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            if (v_len > 1) {
                std::memcpy(v + 1, A.data() + i * n + (i + 2),
                            (v_len - 1) * sizeof(double));
            }

            // Verify reflector orthogonality: H = I - tau*v*v^T should be orthogonal
            double vtv = 0.0;
            for (size_t jj = 0; jj < v_len; ++jj)
                vtv += v[jj] * v[jj];
            double expected_tau = 2.0 / vtv;
            double tau_ratio = tau / expected_tau;
            if (std::abs(tau_ratio - 1.0) > 1e-14) {
                std::cout << "  Reflector " << i << ": v^T*v=" << vtv
                          << " tau=" << tau << " expected_tau=" << expected_tau
                          << " ratio=" << tau_ratio << "\n";
                double expected_vtv = 2.0 / tau;
                std::cout << "    expected v^T*v from tau: " << expected_vtv << "\n";
                std::cout << "    actual v^T*v: " << vtv << "\n";
                std::cout << "    diff: " << (vtv - expected_vtv) << "\n";
                // Print ALL elements of A for this reflector
                std::cout << "  A[" << i << ", " << (i + 1) << ".." << (n - 1) << "]:";
                for (size_t jj = i + 1; jj < n; ++jj)
                    std::cout << " " << A[i * n + jj];
                std::cout << "\n";
                std::cout << "  v[0]=" << v[0];
                for (size_t jj = 1; jj < v_len; ++jj)
                    std::cout << " v[" << jj << "]=" << v[jj];
                std::cout << "\n";
                std::cout << "  e[" << (i + 1) << "]=" << e[i + 1] << "\n";
                // Check if the issue is that A[i, i+1] should be e[i+1] but isn't
                std::cout << "  A[" << i << ", " << (i + 1) << "]=" << A[i * n + (i + 1)]
                          << " (should be e[" << (i + 1) << "]=" << e[i + 1] << ")\n";
            }

            // Apply to VT[:, i+1:]
            np::internal::cpu::householder_apply_right(tau, v,
                                                       &VT[0 * n + (i + 1)],
                                                       k, v_len, n);

            // Check orthogonality of VT
            double vt_ortho = orthogonalityErrorVT(VT.data(), k, n);
            std::cout << "  After reflector " << i << " (v_len=" << v_len
                      << " tau=" << tau << "): VT ortho=" << vt_ortho << "\n";
            if (vt_ortho > 1e-10) {
                std::cout << "  *** ORTHOGONALITY BROKEN at reflector " << i << " ***\n";
                // Print the first few elements of v
                std::cout << "  v[0]=" << v[0];
                for (size_t jj = 1; jj < std::min(v_len, size_t(5)); ++jj)
                    std::cout << " v[" << jj << "]=" << v[jj];
                std::cout << "\n";
                break;
            }
        }

        double vt_ortho = orthogonalityErrorVT(VT.data(), k, n);
        std::cout << "Final VT orthogonality error: " << vt_ortho << "\n";
        EXPECT_LT(vt_ortho, 1e-10);
    }

    // ============================================================
    //  Test 8: Verify Q^T * A * P = B for multi-block case
    //
    //  For a 500x50 matrix (k=50, NB=32, so 2 blocks), verify that
    //  the GEBRD bidiagonal reduction is correct by checking
    //  Q^T * A * P = B where Q and P are built from the reflectors.
    // ============================================================

    TEST(GelsdTest, verifyQtransposeAPequalsB) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q from left reflectors: Q = H_0 * H_1 * ... * H_{k-1}
        // Start with identity, apply reflectors in forward order
        std::vector<double> Q(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        // Applying reflectors bottom-up (k-1 down to 0) to the left of I gives:
        //   Q = H_0 * H_1 * ... * H_{k-1}
        // Then Q^T = H_{k-1} * ... * H_0, which is the correct order for
        // the bidiagonal reduction: B = Q^T * A * P = H_{k-1} * ... * H_0 * A * P
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            // Build reflector v from A (stored in column i, rows i..m-1)
            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A_work[(i + r) * n + i];

            // Apply H_i to Q from the left: Q = H_i * Q
            // Q is m x m, H_i affects rows i..m-1
            for (size_t j = 0; j < m; ++j) {
                // dot = v^T * Q[i:, j]
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q[(i + r) * m + j];
                // Q[i:, j] -= tau * dot * v
                for (size_t r = 0; r < v_len; ++r)
                    Q[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build P^T from right reflectors: P^T = G_0 * G_1 * ... * G_{k-1}
        // where G_i = I - taup[i] * u * u^T
        // Start with identity, apply reflectors in forward order
        std::vector<double> PT(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) PT[i * n + i] = 1.0;

        // Apply right reflectors in REVERSE order (bottom-up) to build P.
        // P = G_0 * G_1 * ... * G_{k-1} where G_i = I - taup[i] * u * u^T.
        // Applying reflectors bottom-up (k-1 down to 0) to the left of I gives:
        //   PT = G_0 * G_1 * ... * G_{k-1} = P
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t u_len = n - i - 1;
            if (u_len <= 0) continue;

            // Build reflector u from A (stored in row i, columns i+1..n-1)
            double u_buf[256];
            std::vector<double> u_heap;
            double *u = u_buf;
            if (u_len > 256) {
                u_heap.resize(u_len);
                u = u_heap.data();
            }
            u[0] = 1.0;
            for (size_t c = 1; c < u_len; ++c)
                u[c] = A_work[i * n + (i + 1 + c)];

            // Apply G_i to PT from the left: PT = G_i * PT
            // PT is n x n, G_i affects columns i+1..n-1
            for (size_t j = 0; j < n; ++j) {
                // dot = u^T * PT[i+1:, j]
                double dot = 0.0;
                for (size_t c = 0; c < u_len; ++c)
                    dot += u[c] * PT[(i + 1 + c) * n + j];
                // PT[i+1:, j] -= tau * dot * u
                for (size_t c = 0; c < u_len; ++c)
                    PT[(i + 1 + c) * n + j] -= tau * dot * u[c];
            }
        }

        // Build B from d and e
        std::vector<double> B(m * n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < k)
                B[i * n + i + 1] = e[i + 1];
        }

        // Compute Q^T * A * P
        // First: AQ = A * P (m x n)
        std::vector<double> AQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < n; ++p)
                    AQ[i * n + j] += A_orig[i * n + p] * PT[p * n + j];

        // Then: QTAQ = Q^T * AQ (m x n)
        std::vector<double> QTAQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < m; ++p)
                    QTAQ[i * n + j] += Q[p * m + i] * AQ[p * n + j];

        // Compare QTAQ with B
        double max_err = 0.0;
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double err = std::abs(QTAQ[i * n + j] - B[i * n + j]);
                if (err > max_err) max_err = err;
            }
        }

        std::cout << "\n=== Verify Q^T * A * P = B for 500x50 (multi-block) ===\n";
        std::cout << "Max |Q^T * A * P - B|: " << max_err << "\n";
        std::cout << "d[0]=" << d[0] << " d[24]=" << d[24] << " d[49]=" << d[49] << "\n";
        std::cout << "e[1]=" << e[1] << " e[25]=" << e[25] << " e[49]=" << e[49] << "\n";

        // If this is large, the GEBRD bidiagonal reduction is wrong
        EXPECT_LT(max_err, 1e-10);
    }

    // ============================================================
    //  Test 9: GEBRD two-block case (100x40)
    //
    //  Tests a 100x40 matrix which requires 2 blocks (NB=32).
    //  Verifies Q^T * A * P = B.
    // ============================================================

    TEST(GelsdTest, gebrdTwoBlocks100x40) {
        size_t m = 100;
        size_t n = 40;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q from left reflectors
        std::vector<double> Q(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A_work[(i + r) * n + i];

            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q[(i + r) * m + j];
                for (size_t r = 0; r < v_len; ++r)
                    Q[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build P^T from right reflectors
        std::vector<double> PT(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) PT[i * n + i] = 1.0;

        // Apply right reflectors in REVERSE order (bottom-up) to build P.
        // P = G_0 * G_1 * ... * G_{k-1} where G_i = I - taup[i] * u * u^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t u_len = n - i - 1;
            if (u_len <= 0) continue;

            double u_buf[256];
            std::vector<double> u_heap;
            double *u = u_buf;
            if (u_len > 256) {
                u_heap.resize(u_len);
                u = u_heap.data();
            }
            u[0] = 1.0;
            for (size_t c = 1; c < u_len; ++c)
                u[c] = A_work[i * n + (i + 1 + c)];

            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t c = 0; c < u_len; ++c)
                    dot += u[c] * PT[(i + 1 + c) * n + j];
                for (size_t c = 0; c < u_len; ++c)
                    PT[(i + 1 + c) * n + j] -= tau * dot * u[c];
            }
        }

        // Build B
        std::vector<double> B(m * n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < k)
                B[i * n + i + 1] = e[i + 1];
        }

        // Compute Q^T * A * P
        std::vector<double> AQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < n; ++p)
                    AQ[i * n + j] += A_orig[i * n + p] * PT[p * n + j];

        std::vector<double> QTAQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < m; ++p)
                    QTAQ[i * n + j] += Q[p * m + i] * AQ[p * n + j];

        double max_err = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                max_err = std::max(max_err, std::abs(QTAQ[i * n + j] - B[i * n + j]));

        std::cout << "\n=== GEBRD two-block 100x40 ===\n";
        std::cout << "Max |Q^T * A * P - B|: " << max_err << "\n";

        EXPECT_LT(max_err, 1e-10);
    }

    // ============================================================
    //  Test 10: GEBRD two-block case (64x64)
    //
    //  Tests a 64x64 matrix which requires exactly 2 blocks (NB=32).
    //  Verifies Q^T * A * P = B.
    // ============================================================

    TEST(GelsdTest, gebrdTwoBlocks64x64) {
        size_t m = 64;
        size_t n = 64;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q from left reflectors
        std::vector<double> Q(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A_work[(i + r) * n + i];

            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q[(i + r) * m + j];
                for (size_t r = 0; r < v_len; ++r)
                    Q[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build P^T from right reflectors
        std::vector<double> PT(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) PT[i * n + i] = 1.0;

        // Apply right reflectors in REVERSE order (bottom-up) to build P.
        // P = G_0 * G_1 * ... * G_{k-1} where G_i = I - taup[i] * u * u^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t u_len = n - i - 1;
            if (u_len <= 0) continue;

            double u_buf[256];
            std::vector<double> u_heap;
            double *u = u_buf;
            if (u_len > 256) {
                u_heap.resize(u_len);
                u = u_heap.data();
            }
            u[0] = 1.0;
            for (size_t c = 1; c < u_len; ++c)
                u[c] = A_work[i * n + (i + 1 + c)];

            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t c = 0; c < u_len; ++c)
                    dot += u[c] * PT[(i + 1 + c) * n + j];
                for (size_t c = 0; c < u_len; ++c)
                    PT[(i + 1 + c) * n + j] -= tau * dot * u[c];
            }
        }

        // Build B
        std::vector<double> B(m * n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < k)
                B[i * n + i + 1] = e[i + 1];
        }

        // Compute Q^T * A * P
        std::vector<double> AQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < n; ++p)
                    AQ[i * n + j] += A_orig[i * n + p] * PT[p * n + j];

        std::vector<double> QTAQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < m; ++p)
                    QTAQ[i * n + j] += Q[p * m + i] * AQ[p * n + j];

        double max_err = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                max_err = std::max(max_err, std::abs(QTAQ[i * n + j] - B[i * n + j]));

        std::cout << "\n=== GEBRD two-block 64x64 ===\n";
        std::cout << "Max |Q^T * A * P - B|: " << max_err << "\n";

        EXPECT_LT(max_err, 1e-10);
    }

    // ============================================================
    //  Test 11: GEBRD one-block + one reflector (33x33)
    //
    //  Tests a 33x33 matrix which requires 1 block (32) + 1 reflector.
    //  Verifies Q^T * A * P = B.
    // ============================================================

    TEST(GelsdTest, gebrdOneBlockPlusOne33x33) {
        size_t m = 33;
        size_t n = 33;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q from left reflectors
        std::vector<double> Q(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A_work[(i + r) * n + i];

            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q[(i + r) * m + j];
                for (size_t r = 0; r < v_len; ++r)
                    Q[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build P^T from right reflectors
        std::vector<double> PT(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) PT[i * n + i] = 1.0;

        // Apply right reflectors in REVERSE order (bottom-up) to build P.
        // P = G_0 * G_1 * ... * G_{k-1} where G_i = I - taup[i] * u * u^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t u_len = n - i - 1;
            if (u_len <= 0) continue;

            double u_buf[256];
            std::vector<double> u_heap;
            double *u = u_buf;
            if (u_len > 256) {
                u_heap.resize(u_len);
                u = u_heap.data();
            }
            u[0] = 1.0;
            for (size_t c = 1; c < u_len; ++c)
                u[c] = A_work[i * n + (i + 1 + c)];

            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t c = 0; c < u_len; ++c)
                    dot += u[c] * PT[(i + 1 + c) * n + j];
                for (size_t c = 0; c < u_len; ++c)
                    PT[(i + 1 + c) * n + j] -= tau * dot * u[c];
            }
        }

        // Build B
        std::vector<double> B(m * n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < k)
                B[i * n + i + 1] = e[i + 1];
        }

        // Compute Q^T * A * P
        std::vector<double> AQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < n; ++p)
                    AQ[i * n + j] += A_orig[i * n + p] * PT[p * n + j];

        std::vector<double> QTAQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < m; ++p)
                    QTAQ[i * n + j] += Q[p * m + i] * AQ[p * n + j];

        double max_err = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                max_err = std::max(max_err, std::abs(QTAQ[i * n + j] - B[i * n + j]));

        std::cout << "\n=== GEBRD one-block + one 33x33 ===\n";
        std::cout << "Max |Q^T * A * P - B|: " << max_err << "\n";

        EXPECT_LT(max_err, 1e-10);
    }

    // ============================================================
    //  Test 12: GEBRD block-by-block verification (64x64)
    //
    //  Applies the GEBRD algorithm block by block and verifies
    //  Q^T * A * P = B after each block.
    // ============================================================

    TEST(GelsdTest, gebrdBlockByBlock64x64) {
        size_t m = 64;
        size_t n = 64;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A_orig(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A_orig[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> A_work = A_orig;
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A_work.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q from left reflectors
        std::vector<double> Q(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A_work[(i + r) * n + i];

            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q[(i + r) * m + j];
                for (size_t r = 0; r < v_len; ++r)
                    Q[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build P^T from right reflectors
        std::vector<double> PT(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) PT[i * n + i] = 1.0;

        // Apply right reflectors in REVERSE order (bottom-up) to build P.
        // P = G_0 * G_1 * ... * G_{k-1} where G_i = I - taup[i] * u * u^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = taup[i];
            if (tau == 0.0) continue;
            size_t u_len = n - i - 1;
            if (u_len <= 0) continue;

            double u_buf[256];
            std::vector<double> u_heap;
            double *u = u_buf;
            if (u_len > 256) {
                u_heap.resize(u_len);
                u = u_heap.data();
            }
            u[0] = 1.0;
            for (size_t c = 1; c < u_len; ++c)
                u[c] = A_work[i * n + (i + 1 + c)];

            for (size_t j = 0; j < n; ++j) {
                double dot = 0.0;
                for (size_t c = 0; c < u_len; ++c)
                    dot += u[c] * PT[(i + 1 + c) * n + j];
                for (size_t c = 0; c < u_len; ++c)
                    PT[(i + 1 + c) * n + j] -= tau * dot * u[c];
            }
        }

        // Build B
        std::vector<double> B(m * n, 0.0);
        for (size_t i = 0; i < k; ++i) {
            B[i * n + i] = d[i];
            if (i + 1 < k)
                B[i * n + i + 1] = e[i + 1];
        }

        // Compute Q^T * A * P
        std::vector<double> AQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < n; ++p)
                    AQ[i * n + j] += A_orig[i * n + p] * PT[p * n + j];

        std::vector<double> QTAQ(m * n, 0.0);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                for (size_t p = 0; p < m; ++p)
                    QTAQ[i * n + j] += Q[p * m + i] * AQ[p * n + j];

        double max_err = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                max_err = std::max(max_err, std::abs(QTAQ[i * n + j] - B[i * n + j]));

        std::cout << "\n=== GEBRD block-by-block 64x64 ===\n";
        std::cout << "Max |Q^T * A * P - B|: " << max_err << "\n";

        EXPECT_LT(max_err, 1e-10);
    }

    // ============================================================
    //  Test 13: Compare compact WY vs individual reflector application
    //
    //  Verifies that multiply_left_q (compact WY) produces the same
    //  result as applying left reflectors individually.
    // ============================================================

    TEST(GelsdTest, compareCompactWyVsIndividual) {
        size_t m = 500;
        size_t n = 50;
        size_t k = std::min(m, n);

        // Generate random matrix A
        std::vector<double> A(m * n);
        for (size_t i = 0; i < m * n; ++i)
            A[i] = (double) rand() / RAND_MAX;

        // Run GEBRD
        std::vector<double> d(k), e(k, 0.0);
        std::vector<double> tauq(k), taup(k);
        np::internal::cpu::gebrd(A.data(), m, n, d.data(), e.data(),
                                 tauq.data(), taup.data());

        // Build Q by applying left reflectors individually to identity
        std::vector<double> Q_individual(m * m, 0.0);
        for (size_t i = 0; i < m; ++i) Q_individual[i * m + i] = 1.0;

        // Apply left reflectors in REVERSE order (bottom-up) to build Q.
        // Q = H_0 * H_1 * ... * H_{k-1} where H_i = I - tauq[i] * v * v^T.
        for (size_t ii = k; ii > 0;) {
            size_t i = --ii;
            double tau = tauq[i];
            if (tau == 0.0) continue;
            size_t v_len = m - i;
            if (v_len <= 1) continue;

            double v_buf[256];
            std::vector<double> v_heap;
            double *v = v_buf;
            if (v_len > 256) {
                v_heap.resize(v_len);
                v = v_heap.data();
            }
            v[0] = 1.0;
            for (size_t r = 1; r < v_len; ++r)
                v[r] = A[(i + r) * n + i];

            for (size_t j = 0; j < m; ++j) {
                double dot = 0.0;
                for (size_t r = 0; r < v_len; ++r)
                    dot += v[r] * Q_individual[(i + r) * m + j];
                for (size_t r = 0; r < v_len; ++r)
                    Q_individual[(i + r) * m + j] -= tau * dot * v[r];
            }
        }

        // Build Q using compact WY (multiply_left_q applied to identity)
        std::vector<double> Q_compact(m * k, 0.0);
        for (size_t i = 0; i < k; ++i)
            Q_compact[i * k + i] = 1.0;
        np::internal::cpu::multiply_left_q(A.data(), m, n, tauq.data(), k,
                                           Q_compact.data(), k);

        // Compare: Q_individual (m x m) vs Q_compact (m x k) for first k columns
        double max_diff = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < k; ++j)
                max_diff = std::max(max_diff, std::abs(Q_individual[i * m + j] - Q_compact[i * k + j]));

        std::cout << "\n=== Compare compact WY vs individual reflector application ===\n";
        std::cout << "Max |Q_individual - Q_compact|: " << max_diff << "\n";

        // These should match to high precision
        EXPECT_LT(max_diff, 1e-12);
    }

}// namespace

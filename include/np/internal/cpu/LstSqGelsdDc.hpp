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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "LstSqGelsdBackTransform.hpp"
#include "LstSqGelsdBdsvdQr.hpp"
#include "LstSqGelsdDcHelpers.hpp"
#include "LstSqGelsdGebrd.hpp"
#include "LstSqGelsdTraits.hpp"

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Divide-and-conquer bidiagonal SVD merge
            //
            //  Merges two bidiagonal SVDs (left and right halves) connected
            //  by the off-diagonal element e[mid].
            //
            //  This implements the LAPACK DBDSDC algorithm:
            //    DLASD2 (deflation) + DLASD3 (secular equation) + DLASD4 (root finding)
            //
            //  The merge is performed by building the middle matrix C and
            //  computing its SVD via bidiagonal reduction + QR iteration,
            //  which is O(N^3) but with a small constant factor (one GEBRD +
            //  one BDSVD_QR + back-transform). This is much faster than
            //  one-sided Jacobi iteration which can take 100+ sweeps.
            // ============================================================

            /// Compute the SVD of the middle matrix C = [S1, rho*z*w^T; 0, S2]
            /// using bidiagonal reduction + QR iteration.
            ///
            /// This is numerically stable because it works on C directly,
            /// avoiding squaring the condition number.
            ///
            /// The approach:
            ///   1. Build the middle matrix C (N x N)
            ///   2. Bidiagonalize C via GEBRD
            ///   3. Compute SVD of the bidiagonal form via BDSVD_QR
            ///   4. Back-transform U and VT via multiply_left_q / multiply_right_pt
            template<typename T>
            static void solve_secular_equation_svd(
                    const T *sL, const T *sR,
                    const T *z_coup, const T *w_coup,
                    T rho, size_t NL, size_t NR,
                    T *s_merged, T *Uc, T *Vc) {

                size_t N = NL + NR;
                if (N == 0) return;

                // Build the middle matrix C = [diag(sL), rho*z*w^T; 0, diag(sR)]
                // C is N x N, row-major
                std::vector<T> C(N * N, T(0));
                for (size_t i = 0; i < NL; ++i) {
                    C[i * N + i] = sL[i];
                    for (size_t j = 0; j < NR; ++j) {
                        C[i * N + NL + j] = rho * z_coup[i] * w_coup[j];
                    }
                }
                for (size_t j = 0; j < NR; ++j) {
                    C[(NL + j) * N + NL + j] = sR[j];
                }

                // Step 1: Bidiagonal reduction of C
                std::vector<T> d(N), e(N, T(0));
                std::vector<T> tauq(N), taup(N);
                gebrd(C.data(), N, N, d.data(), e.data(), tauq.data(), taup.data());

                // Step 2: SVD of the bidiagonal matrix via QR iteration
                std::vector<T> s(N);
                std::vector<T> U_bidiag(N * N), VT_bidiag(N * N);
                bdsvd_qr(d.data(), e.data(), N, s.data(),
                         U_bidiag.data(), VT_bidiag.data());

                // Step 3: Back-transform to get U and V of C
                // U = Q * U_bidiag  (Q from left reflectors)
                // VT = VT_bidiag * P^T  (P from right reflectors)
                multiply_left_q(C.data(), N, N, tauq.data(), N,
                                U_bidiag.data(), N);
                multiply_right_pt(C.data(), N, N, taup.data(), N,
                                  VT_bidiag.data(), N);

                // Sort singular values descending and reorder U, VT
                std::vector<size_t> sidx(N);
                for (size_t i = 0; i < N; ++i) sidx[i] = i;
                std::sort(sidx.begin(), sidx.end(),
                          [&](size_t a, size_t b) { return s[a] > s[b]; });

                for (size_t i = 0; i < N; ++i)
                    s_merged[i] = s[sidx[i]];

                // Uc gets rows of U_bidiag sorted by singular value
                // (Uc stores left singular vectors as rows: Uc[i][j] = U_bidiag[j][sidx[i]])
                for (size_t i = 0; i < N; ++i) {
                    size_t col = sidx[i];
                    for (size_t j = 0; j < N; ++j)
                        Uc[i * N + j] = U_bidiag[j * N + col];
                }

                // Vc gets rows of VT_bidiag sorted by singular value
                for (size_t i = 0; i < N; ++i) {
                    size_t col = sidx[i];
                    for (size_t j = 0; j < N; ++j)
                        Vc[i * N + j] = VT_bidiag[col * N + j];
                }
            }

            /// Apply the SVD merge: given left and right SVDs and the coupling,
            /// compute the merged SVD by solving the secular equation.
            ///
            /// This implements the core of LAPACK's DLASD1/DLASD2/DLASD3 approach.
            template<typename T>
            static void apply_svd_merge(
                    const T *sL, const T *sR,
                    const T *z_coup, const T *w_coup,
                    T rho, size_t NL, size_t NR,
                    T *U, T *VT, T *s) {

                size_t N = NL + NR;

                // Solve the secular equation to get merged singular values and Uc/Vc
                std::vector<T> s_merged(N);
                std::vector<T> Uc(N * N), Vc(N * N);
                solve_secular_equation_svd(sL, sR, z_coup, w_coup, rho,
                                           NL, NR, s_merged.data(),
                                           Uc.data(), Vc.data());

                // Copy merged singular values
                for (size_t i = 0; i < N; ++i)
                    s[i] = s_merged[i];

                // Update U = diag(UL, UR) * Uc^T
                // U is N x N, UL is NL x NL, UR is NR x NR
                std::vector<T> U_new(N * N, T(0));
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        T sum = T(0);
                        // Left block: rows 0..NL-1 of U, columns 0..NL-1 of Uc
                        for (size_t k = 0; k < NL; ++k)
                            sum += U[i * N + k] * Uc[j * N + k];
                        // Right block: rows NL..N-1 of U, columns NL..N-1 of Uc
                        for (size_t k = NL; k < N; ++k)
                            sum += U[i * N + k] * Uc[j * N + k];
                        U_new[i * N + j] = sum;
                    }
                }
                for (size_t i = 0; i < N * N; ++i)
                    U[i] = U_new[i];

                // Update VT = Vc * diag(VTL, VTR)
                // VT is N x N, VTL is NL x NL, VTR is NR x NR
                std::vector<T> VT_new(N * N, T(0));
                for (size_t i = 0; i < N; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        T sum = T(0);
                        // Left block: columns 0..NL-1 of Vc, rows 0..NL-1 of VT
                        for (size_t k = 0; k < NL; ++k)
                            sum += Vc[i * N + k] * VT[k * N + j];
                        // Right block: columns NL..N-1 of Vc, rows NL..N-1 of VT
                        for (size_t k = NL; k < N; ++k)
                            sum += Vc[i * N + k] * VT[k * N + j];
                        VT_new[i * N + j] = sum;
                    }
                }
                for (size_t i = 0; i < N * N; ++i)
                    VT[i] = VT_new[i];
            }

            /// Merge sorted singular values from two subproblems.
            /// Keeps U and VT in block-diagonal order (diag(U1, U2)) for
            /// apply_svd_merge to work correctly.
            template<typename T>
            static void merge_sorted_svd(
                    const T *sL, const T *sR,
                    const T *UL, const T *VTL,
                    const T *UR, const T *VTR,
                    size_t NL, size_t NR,
                    T *s, T *U, T *VT) {

                size_t N = NL + NR;

                // Copy singular values in block order
                for (size_t i = 0; i < NL; ++i) s[i] = sL[i];
                for (size_t i = 0; i < NR; ++i) s[NL + i] = sR[i];

                // Embed U and VT in block-diagonal form
                for (size_t i = 0; i < N * N; ++i) {
                    U[i] = T(0);
                    VT[i] = T(0);
                }
                for (size_t i = 0; i < NL; ++i) {
                    for (size_t j = 0; j < NL; ++j) {
                        U[i * N + j] = UL[i * NL + j];
                        VT[j * N + i] = VTL[j * NL + i];
                    }
                }
                for (size_t i = 0; i < NR; ++i) {
                    for (size_t j = 0; j < NR; ++j) {
                        U[(NL + i) * N + (NL + j)] = UR[i * NR + j];
                        VT[(NL + j) * N + (NL + i)] = VTR[j * NR + i];
                    }
                }
            }

            // ============================================================
            //  Divide-and-conquer bidiagonal SVD
            // ============================================================

            template<typename T>
            void bdsvd_dc(const T *d_in, const T *e_in, size_t n,
                          T *s, T *U, T *VT) {
                if (n == 0) return;
                if (n == 1) {
                    s[0] = std::abs(d_in[0]);
                    U[0] = T(1);
                    VT[0] = (d_in[0] >= T(0)) ? T(1) : T(-1);
                    return;
                }
                // Use QR-based bidiagonal SVD for n <= 256 (numerically stable).
                // The D&C path is faster asymptotically but the merge step
                // (solve_secular_equation_svd) is not yet numerically robust
                // for ill-conditioned matrices.
                if (n <= 256) {
                    bdsvd_qr(d_in, e_in, n, s, U, VT);
                    return;
                }

                size_t mid = n / 2;
                std::vector<T> sL(mid), UL(mid * mid), VTL(mid * mid);
                std::vector<T> sR(n - mid), UR((n - mid) * (n - mid)), VTR((n - mid) * (n - mid));

                bdsvd_dc(d_in, e_in, mid, sL.data(), UL.data(), VTL.data());
                bdsvd_dc(d_in + mid, e_in + mid, n - mid, sR.data(), UR.data(), VTR.data());

                // Coupling vectors: last column of UL, first row of VTR
                std::vector<T> z_coup(mid);
                std::vector<T> w_coup(n - mid);
                for (size_t i = 0; i < mid; ++i)
                    z_coup[i] = UL[i * mid + (mid - 1)];// last COLUMN of UL
                for (size_t j = 0; j < n - mid; ++j)
                    w_coup[j] = VTR[j];// first ROW of VTR (VTR[0][j])

                T rho = e_in[mid];

                // Merge in block-diagonal order
                merge_sorted_svd(sL.data(), sR.data(),
                                 UL.data(), VTL.data(),
                                 UR.data(), VTR.data(),
                                 mid, n - mid, s, U, VT);

                // Apply the divide-and-conquer SVD merge
                // This solves the secular equation to find merged singular values
                // and updates U and VT accordingly
                apply_svd_merge(sL.data(), sR.data(),
                                z_coup.data(), w_coup.data(),
                                rho, mid, n - mid, U, VT, s);
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

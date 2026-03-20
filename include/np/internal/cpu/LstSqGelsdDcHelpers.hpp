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

namespace np {
    namespace internal {
        namespace cpu {

            // ============================================================
            //  Divide-and-conquer bidiagonal SVD helpers
            // ============================================================

            /// Compute Givens rotation parameters c, s such that:
            ///   [ c  s ] * [ a ] = [ r ]
            ///   [ -s c ]   [ b ]   [ 0 ]
            template<typename T>
            static void givens_rot(T a, T b, T &c, T &s, T &r) {
                if (std::abs(b) < std::numeric_limits<T>::min()) {
                    c = T(1);
                    s = T(0);
                    r = a;
                } else if (std::abs(a) < std::numeric_limits<T>::min()) {
                    c = T(0);
                    s = T(1);
                    r = b;
                } else {
                    T tau = std::sqrt(a * a + b * b);
                    c = a / tau;
                    s = -b / tau;
                    r = tau;
                }
            }

            // ============================================================
            //  DLASD5: Secular equation solver for N=2
            //  Exact translation from LAPACK DLASD5.
            // ============================================================
            template<typename T>
            static void dlasd5_solve(int I, const T *D, const T *Z, T *DELTA,
                                     T RHO, T &DSIGMA, T *WORK) {
                T ZERO = T(0), ONE = T(1), TWO = T(2), THREE = T(3), FOUR = T(4);

                T DEL = D[1] - D[0];
                T DELSQ = DEL * (D[1] + D[0]);

                if (I == 1) {
                    T W = ONE + FOUR * RHO * (Z[1] * Z[1] / (D[0] + THREE * D[1]) - Z[0] * Z[0] / (THREE * D[0] + D[1])) / DEL;
                    if (W > ZERO) {
                        T B = DELSQ + RHO * (Z[0] * Z[0] + Z[1] * Z[1]);
                        T C = RHO * Z[0] * Z[0] * DELSQ;
                        T TAU = TWO * C / (B + std::sqrt(std::abs(B * B - FOUR * C)));
                        TAU = TAU / (D[0] + std::sqrt(D[0] * D[0] + TAU));
                        DSIGMA = D[0] + TAU;
                        DELTA[0] = -TAU;
                        DELTA[1] = DEL - TAU;
                        WORK[0] = TWO * D[0] + TAU;
                        WORK[1] = (D[0] + TAU) + D[1];
                    } else {
                        T B = -DELSQ + RHO * (Z[0] * Z[0] + Z[1] * Z[1]);
                        T C = RHO * Z[1] * Z[1] * DELSQ;
                        T TAU;
                        if (B > ZERO)
                            TAU = -TWO * C / (B + std::sqrt(B * B + FOUR * C));
                        else
                            TAU = (B - std::sqrt(B * B + FOUR * C)) / TWO;
                        TAU = TAU / (D[1] + std::sqrt(std::abs(D[1] * D[1] + TAU)));
                        DSIGMA = D[1] + TAU;
                        DELTA[0] = -(DEL + TAU);
                        DELTA[1] = -TAU;
                        WORK[0] = D[0] + TAU + D[1];
                        WORK[1] = TWO * D[1] + TAU;
                    }
                } else {
                    // I = 2
                    T B = -DELSQ + RHO * (Z[0] * Z[0] + Z[1] * Z[1]);
                    T C = RHO * Z[1] * Z[1] * DELSQ;
                    T TAU;
                    if (B > ZERO)
                        TAU = (B + std::sqrt(B * B + FOUR * C)) / TWO;
                    else
                        TAU = TWO * C / (-B + std::sqrt(B * B + FOUR * C));
                    TAU = TAU / (D[1] + std::sqrt(D[1] * D[1] + TAU));
                    DSIGMA = D[1] + TAU;
                    DELTA[0] = -(DEL + TAU);
                    DELTA[1] = -TAU;
                    WORK[0] = D[0] + TAU + D[1];
                    WORK[1] = TWO * D[1] + TAU;
                }
            }

            // ============================================================
            //  DLAED6: Three-pole interpolation for secular equation
            //  Exact translation from LAPACK DLAED6.
            // ============================================================
            template<typename T>
            static void dlaed6_solve(int KNITER, bool ORGATI, T RHO,
                                     const T *D, const T *Z, T FINIT,
                                     T &TAU, int &INFO) {
                T ZERO = T(0), ONE = T(1), TWO = T(2), FOUR = T(4), EIGHT = T(8);
                const int MAXIT = 40;

                INFO = 0;

                T LBD, UBD;
                if (ORGATI) {
                    LBD = D[1];
                    UBD = D[2];
                } else {
                    LBD = D[0];
                    UBD = D[1];
                }
                if (FINIT < ZERO)
                    LBD = ZERO;
                else
                    UBD = ZERO;

                int NITER = 1;
                TAU = ZERO;

                if (KNITER == 2) {
                    T TEMP, C, A, B;
                    if (ORGATI) {
                        TEMP = (D[2] - D[1]) / TWO;
                        C = RHO + Z[0] / ((D[0] - D[1]) - TEMP);
                        A = C * (D[1] + D[2]) + Z[1] + Z[2];
                        B = C * D[1] * D[2] + Z[1] * D[2] + Z[2] * D[1];
                    } else {
                        TEMP = (D[0] - D[1]) / TWO;
                        C = RHO + Z[2] / ((D[2] - D[1]) - TEMP);
                        A = C * (D[0] + D[1]) + Z[0] + Z[1];
                        B = C * D[0] * D[1] + Z[0] * D[1] + Z[1] * D[0];
                    }
                    TEMP = std::max({std::abs(A), std::abs(B), std::abs(C)});
                    A /= TEMP;
                    B /= TEMP;
                    C /= TEMP;
                    if (C == ZERO)
                        TAU = B / A;
                    else if (A <= ZERO)
                        TAU = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                    else
                        TAU = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                    if (TAU < LBD || TAU > UBD)
                        TAU = (LBD + UBD) / TWO;
                    if (D[0] == TAU || D[1] == TAU || D[2] == TAU) {
                        TAU = ZERO;
                    } else {
                        TEMP = FINIT + TAU * Z[0] / (D[0] * (D[0] - TAU)) +
                               TAU * Z[1] / (D[1] * (D[1] - TAU)) +
                               TAU * Z[2] / (D[2] * (D[2] - TAU));
                        if (TEMP <= ZERO) LBD = TAU;
                        else
                            UBD = TAU;
                        if (std::abs(FINIT) <= std::abs(TEMP)) TAU = ZERO;
                    }
                }

                T EPS = std::numeric_limits<T>::epsilon();
                T SMALL1 = T(1e-20);
                T SMINV1 = T(1e20);
                T SMALL2 = T(1e-40);
                T SMINV2 = T(1e40);

                T TEMP;
                if (ORGATI)
                    TEMP = std::min(std::abs(D[1] - TAU), std::abs(D[2] - TAU));
                else
                    TEMP = std::min(std::abs(D[0] - TAU), std::abs(D[1] - TAU));

                bool SCALE = false;
                T SCLFAC = ONE, SCLINV = ONE;
                T DSCALE[3], ZSCALE[3];
                if (TEMP <= SMALL1) {
                    SCALE = true;
                    if (TEMP <= SMALL2) {
                        SCLFAC = SMINV2;
                        SCLINV = SMALL2;
                    } else {
                        SCLFAC = SMINV1;
                        SCLINV = SMALL1;
                    }
                    for (int i = 0; i < 3; ++i) {
                        DSCALE[i] = D[i] * SCLFAC;
                        ZSCALE[i] = Z[i] * SCLFAC;
                    }
                    TAU *= SCLFAC;
                    LBD *= SCLFAC;
                    UBD *= SCLFAC;
                } else {
                    for (int i = 0; i < 3; ++i) {
                        DSCALE[i] = D[i];
                        ZSCALE[i] = Z[i];
                    }
                }

                T FC = ZERO, DF = ZERO, DDF = ZERO;
                for (int i = 0; i < 3; ++i) {
                    TEMP = ONE / (DSCALE[i] - TAU);
                    T TEMP1 = ZSCALE[i] * TEMP;
                    T TEMP2 = TEMP1 * TEMP;
                    T TEMP3 = TEMP2 * TEMP;
                    FC += TEMP1 / DSCALE[i];
                    DF += TEMP2;
                    DDF += TEMP3;
                }
                T F = FINIT + TAU * FC;

                if (std::abs(F) > ZERO) {
                    if (F <= ZERO) LBD = TAU;
                    else
                        UBD = TAU;

                    int ITER = NITER + 1;
                    for (int NITER_LOOP = ITER; NITER_LOOP <= MAXIT; ++NITER_LOOP) {
                        T TEMP1, TEMP2;
                        if (ORGATI) {
                            TEMP1 = DSCALE[1] - TAU;
                            TEMP2 = DSCALE[2] - TAU;
                        } else {
                            TEMP1 = DSCALE[0] - TAU;
                            TEMP2 = DSCALE[1] - TAU;
                        }
                        T A = (TEMP1 + TEMP2) * F - TEMP1 * TEMP2 * DF;
                        T B = TEMP1 * TEMP2 * F;
                        T C = F - (TEMP1 + TEMP2) * DF + TEMP1 * TEMP2 * DDF;
                        TEMP = std::max({std::abs(A), std::abs(B), std::abs(C)});
                        A /= TEMP;
                        B /= TEMP;
                        C /= TEMP;
                        T ETA;
                        if (C == ZERO)
                            ETA = B / A;
                        else if (A <= ZERO)
                            ETA = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                        else
                            ETA = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                        if (F * ETA >= ZERO)
                            ETA = -F / DF;

                        TAU = TAU + ETA;
                        if (TAU < LBD || TAU > UBD)
                            TAU = (LBD + UBD) / TWO;

                        FC = ZERO;
                        T ERRETM = ZERO;
                        DF = ZERO;
                        DDF = ZERO;
                        bool pole_hit = false;
                        for (int i = 0; i < 3; ++i) {
                            if (std::abs(DSCALE[i] - TAU) > ZERO) {
                                TEMP = ONE / (DSCALE[i] - TAU);
                                T TEMP1 = ZSCALE[i] * TEMP;
                                T TEMP2 = TEMP1 * TEMP;
                                T TEMP3 = TEMP2 * TEMP;
                                T TEMP4 = TEMP1 / DSCALE[i];
                                FC += TEMP4;
                                ERRETM += std::abs(TEMP4);
                                DF += TEMP2;
                                DDF += TEMP3;
                            } else {
                                pole_hit = true;
                                break;
                            }
                        }
                        if (pole_hit) break;

                        F = FINIT + TAU * FC;
                        ERRETM = EIGHT * (std::abs(FINIT) + std::abs(TAU) * ERRETM) +
                                 std::abs(TAU) * DF;
                        if ((std::abs(F) <= FOUR * EPS * ERRETM) ||
                            ((UBD - LBD) <= FOUR * EPS * std::abs(TAU)))
                            break;
                        if (F <= ZERO) LBD = TAU;
                        else
                            UBD = TAU;
                    }
                }

                if (SCALE) TAU *= SCLINV;
            }

            // ============================================================
            //  DLASD4: Secular equation root finder
            //  Exact translation from LAPACK DLASD4.
            //
            //  Finds the square root of the I-th updated eigenvalue of
            //  diag(D)^2 + RHO * Z * Z^T where ||Z||_2 = 1.
            //
            //  Arguments:
            //    N      - size of the problem
            //    I      - index of eigenvalue to find (1-indexed, 1 <= I <= N)
            //    D      - poles (sorted ascending, 0 <= D[i] < D[j] for i < j)
            //    Z      - coupling vector (||Z||_2 = 1)
            //    DELTA  - output: D(j) - SIGMA for each j
            //    RHO    - rank-1 modification scalar (> 0)
            //    SIGMA  - output: the I-th updated eigenvalue
            //    WORK   - output: D(j) + SIGMA for each j
            //    INFO   - output: 0 = success, 1 = did not converge
            // ============================================================
            template<typename T>
            static void dlasd4_solve(int N, int I, const T *D, const T *Z,
                                     T *DELTA, T RHO, T &SIGMA, T *WORK, int &INFO) {
                T ZERO = T(0), ONE = T(1), TWO = T(2), THREE = T(3), FOUR = T(4);
                T EIGHT = T(8), TEN = T(10);
                const int MAXIT = 400;

                INFO = 0;

                // Quick return for N=1
                if (N == 1) {
                    SIGMA = std::sqrt(D[0] * D[0] + RHO * Z[0] * Z[0]);
                    DELTA[0] = ONE;
                    WORK[0] = ONE;
                    return;
                }

                // N=2 case: use DLASD5
                if (N == 2) {
                    dlasd5_solve(I, D, Z, DELTA, RHO, SIGMA, WORK);
                    return;
                }

                T EPS = std::numeric_limits<T>::epsilon();
                T RHOINV = ONE / RHO;
                T TAU2 = ZERO;

                // ============================================================
                //  Case I = N (largest eigenvalue)
                // ============================================================
                if (I == N) {
                    int II = N - 1;// 0-indexed: N-2
                    int NITER = 1;

                    // Initial guess
                    T TEMP = RHO / TWO;
                    T TEMP1 = TEMP / (D[N - 1] + std::sqrt(D[N - 1] * D[N - 1] + TEMP));
                    for (int J = 0; J < N; ++J) {
                        WORK[J] = D[J] + D[N - 1] + TEMP1;
                        DELTA[J] = (D[J] - D[N - 1]) - TEMP1;
                    }

                    // Evaluate PSI (sum over J=1..N-2)
                    T PSI = ZERO;
                    for (int J = 0; J < II; ++J) {
                        PSI = PSI + Z[J] * Z[J] / (DELTA[J] * WORK[J]);
                    }

                    T C = RHOINV + PSI;
                    T W = C + Z[II] * Z[II] / (DELTA[II] * WORK[II]) +
                          Z[N - 1] * Z[N - 1] / (DELTA[N - 1] * WORK[N - 1]);

                    T TAU;
                    if (W <= ZERO) {
                        TEMP1 = std::sqrt(D[N - 1] * D[N - 1] + RHO);
                        TEMP = Z[N - 2] * Z[N - 2] / ((D[N - 2] + TEMP1) * (D[N - 1] - D[N - 2] + RHO / (D[N - 1] + TEMP1))) +
                               Z[N - 1] * Z[N - 1] / RHO;

                        if (C <= TEMP) {
                            TAU = RHO;
                        } else {
                            T DELSQ = (D[N - 1] - D[N - 2]) * (D[N - 1] + D[N - 2]);
                            T A = -C * DELSQ + Z[N - 2] * Z[N - 2] + Z[N - 1] * Z[N - 1];
                            T B = Z[N - 1] * Z[N - 1] * DELSQ;
                            if (A < ZERO) {
                                TAU2 = TWO * B / (std::sqrt(A * A + FOUR * B * C) - A);
                            } else {
                                TAU2 = (A + std::sqrt(A * A + FOUR * B * C)) / (TWO * C);
                            }
                            TAU = TAU2 / (D[N - 1] + std::sqrt(D[N - 1] * D[N - 1] + TAU2));
                        }
                    } else {
                        T DELSQ = (D[N - 1] - D[N - 2]) * (D[N - 1] + D[N - 2]);
                        T A = -C * DELSQ + Z[N - 2] * Z[N - 2] + Z[N - 1] * Z[N - 1];
                        T B = Z[N - 1] * Z[N - 1] * DELSQ;

                        if (A < ZERO) {
                            TAU2 = TWO * B / (std::sqrt(A * A + FOUR * B * C) - A);
                        } else {
                            TAU2 = (A + std::sqrt(A * A + FOUR * B * C)) / (TWO * C);
                        }
                        TAU = TAU2 / (D[N - 1] + std::sqrt(D[N - 1] * D[N - 1] + TAU2));
                    }

                    SIGMA = D[N - 1] + TAU;
                    for (int J = 0; J < N; ++J) {
                        DELTA[J] = (D[J] - D[N - 1]) - TAU;
                        WORK[J] = D[J] + D[N - 1] + TAU;
                    }

                    // Evaluate PSI and derivative DPSI
                    T DPSI = ZERO;
                    PSI = ZERO;
                    T ERRETM = ZERO;
                    for (int J = 0; J < II; ++J) {
                        TEMP = Z[J] / (DELTA[J] * WORK[J]);
                        PSI = PSI + Z[J] * TEMP;
                        DPSI = DPSI + TEMP * TEMP;
                        ERRETM = ERRETM + PSI;
                    }
                    ERRETM = std::abs(ERRETM);

                    // Evaluate PHI and derivative DPHI
                    TEMP = Z[N - 1] / (DELTA[N - 1] * WORK[N - 1]);
                    T PHI = Z[N - 1] * TEMP;
                    T DPHI = TEMP * TEMP;
                    ERRETM = EIGHT * (-PHI - PSI) + ERRETM - PHI + RHOINV;

                    W = RHOINV + PHI + PSI;

                    // Test for convergence
                    if (std::abs(W) <= EPS * ERRETM) {
                        return;
                    }

                    // Main iteration loop
                    NITER = NITER + 1;
                    T DTNSQ1 = WORK[N - 2] * DELTA[N - 2];
                    T DTNSQ = WORK[N - 1] * DELTA[N - 1];
                    C = W - DTNSQ1 * DPSI - DTNSQ * DPHI;
                    T A = (DTNSQ + DTNSQ1) * W - DTNSQ * DTNSQ1 * (DPSI + DPHI);
                    T B = DTNSQ * DTNSQ1 * W;
                    if (C < ZERO)
                        C = std::abs(C);
                    T ETA;
                    if (C == ZERO) {
                        ETA = RHO - SIGMA * SIGMA;
                    } else if (A >= ZERO) {
                        ETA = (A + std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                    } else {
                        ETA = TWO * B / (A - std::sqrt(std::abs(A * A - FOUR * B * C)));
                    }

                    if (W * ETA > ZERO)
                        ETA = -W / (DPSI + DPHI);
                    TEMP = ETA - DTNSQ;
                    if (TEMP > RHO)
                        ETA = RHO + DTNSQ;

                    ETA = ETA / (SIGMA + std::sqrt(ETA + SIGMA * SIGMA));
                    TAU = TAU + ETA;
                    SIGMA = SIGMA + ETA;

                    for (int J = 0; J < N; ++J) {
                        DELTA[J] = DELTA[J] - ETA;
                        WORK[J] = WORK[J] + ETA;
                    }

                    // Main loop
                    int ITER = NITER + 1;
                    for (int NITER_LOOP = ITER; NITER_LOOP <= MAXIT; ++NITER_LOOP) {
                        // Evaluate PSI and DPSI
                        DPSI = ZERO;
                        PSI = ZERO;
                        ERRETM = ZERO;
                        for (int J = 0; J < II; ++J) {
                            TEMP = Z[J] / (WORK[J] * DELTA[J]);
                            PSI = PSI + Z[J] * TEMP;
                            DPSI = DPSI + TEMP * TEMP;
                            ERRETM = ERRETM + PSI;
                        }
                        ERRETM = std::abs(ERRETM);

                        // Evaluate PHI and DPHI
                        TAU2 = WORK[N - 1] * DELTA[N - 1];
                        TEMP = Z[N - 1] / TAU2;
                        PHI = Z[N - 1] * TEMP;
                        DPHI = TEMP * TEMP;
                        ERRETM = EIGHT * (-PHI - PSI) + ERRETM - PHI + RHOINV;

                        W = RHOINV + PHI + PSI;

                        // Convergence test
                        if (std::abs(W) <= EPS * ERRETM) {
                            return;
                        }

                        // Calculate new step
                        DTNSQ1 = WORK[N - 2] * DELTA[N - 2];
                        DTNSQ = WORK[N - 1] * DELTA[N - 1];
                        C = W - DTNSQ1 * DPSI - DTNSQ * DPHI;
                        A = (DTNSQ + DTNSQ1) * W - DTNSQ1 * DTNSQ * (DPSI + DPHI);
                        B = DTNSQ1 * DTNSQ * W;
                        if (A >= ZERO) {
                            ETA = (A + std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                        } else {
                            ETA = TWO * B / (A - std::sqrt(std::abs(A * A - FOUR * B * C)));
                        }

                        if (W * ETA > ZERO)
                            ETA = -W / (DPSI + DPHI);
                        TEMP = ETA - DTNSQ;
                        if (TEMP <= ZERO)
                            ETA = ETA / TWO;

                        ETA = ETA / (SIGMA + std::sqrt(ETA + SIGMA * SIGMA));
                        TAU = TAU + ETA;
                        SIGMA = SIGMA + ETA;

                        for (int J = 0; J < N; ++J) {
                            DELTA[J] = DELTA[J] - ETA;
                            WORK[J] = WORK[J] + ETA;
                        }
                    }

                    // Not converged
                    INFO = 1;
                    return;
                }

                // ============================================================
                //  Case I < N
                // ============================================================
                int IP1 = I + 1;// 1-indexed: I+1, 0-indexed: I

                // Calculate initial guess
                T DELSQ = (D[IP1] - D[I]) * (D[IP1] + D[I]);
                T DELSQ2 = DELSQ / TWO;
                T SQ2 = std::sqrt((D[I] * D[I] + D[IP1] * D[IP1]) / TWO);
                T TEMP = DELSQ2 / (D[I] + SQ2);
                for (int J = 0; J < N; ++J) {
                    WORK[J] = D[J] + D[I] + TEMP;
                    DELTA[J] = (D[J] - D[I]) - TEMP;
                }

                T PSI = ZERO;
                for (int J = 0; J < I; ++J) {
                    PSI = PSI + Z[J] * Z[J] / (WORK[J] * DELTA[J]);
                }

                T PHI = ZERO;
                for (int J = N - 1; J >= IP1 + 1; --J) {
                    PHI = PHI + Z[J] * Z[J] / (WORK[J] * DELTA[J]);
                }
                T C = RHOINV + PSI + PHI;
                T W = C + Z[I] * Z[I] / (WORK[I] * DELTA[I]) +
                      Z[IP1] * Z[IP1] / (WORK[IP1] * DELTA[IP1]);

                bool GEOMAVG = false;
                int II;
                T SGLB, SGUB;
                bool ORGATI;
                T TAU;
                T A, B;

                if (W > ZERO) {
                    // d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2
                    // Choose d(i) as origin.
                    ORGATI = true;
                    II = I;
                    SGLB = ZERO;
                    SGUB = DELSQ2 / (D[I] + SQ2);
                    A = C * DELSQ + Z[I] * Z[I] + Z[IP1] * Z[IP1];
                    B = Z[I] * Z[I] * DELSQ;
                    if (A > ZERO) {
                        TAU2 = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                    } else {
                        TAU2 = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                    }
                    TAU = TAU2 / (D[I] + std::sqrt(D[I] * D[I] + TAU2));
                    TEMP = std::sqrt(EPS);
                    if ((D[I] <= TEMP * D[IP1]) && (std::abs(Z[I]) <= TEMP) && (D[I] > ZERO)) {
                        TAU = std::min(TEN * D[I], SGUB);
                        GEOMAVG = true;
                    }
                } else {
                    // (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2
                    // Choose d(i+1) as origin.
                    ORGATI = false;
                    II = IP1;
                    SGLB = -DELSQ2 / (D[II] + SQ2);
                    SGUB = ZERO;
                    A = C * DELSQ - Z[I] * Z[I] - Z[IP1] * Z[IP1];
                    B = Z[IP1] * Z[IP1] * DELSQ;
                    if (A < ZERO) {
                        TAU2 = TWO * B / (A - std::sqrt(std::abs(A * A + FOUR * B * C)));
                    } else {
                        TAU2 = -(A + std::sqrt(std::abs(A * A + FOUR * B * C))) / (TWO * C);
                    }
                    TAU = TAU2 / (D[IP1] + std::sqrt(std::abs(D[IP1] * D[IP1] + TAU2)));
                }

                SIGMA = D[II] + TAU;
                for (int J = 0; J < N; ++J) {
                    WORK[J] = D[J] + D[II] + TAU;
                    DELTA[J] = (D[J] - D[II]) - TAU;
                }
                int IIM1 = II - 1;
                int IIP1 = II + 1;

                // Evaluate PSI and DPSI
                T DPSI = ZERO;
                PSI = ZERO;
                T ERRETM = ZERO;
                for (int J = 0; J < IIM1; ++J) {
                    TEMP = Z[J] / (WORK[J] * DELTA[J]);
                    PSI = PSI + Z[J] * TEMP;
                    DPSI = DPSI + TEMP * TEMP;
                    ERRETM = ERRETM + PSI;
                }
                ERRETM = std::abs(ERRETM);

                // Evaluate PHI and DPHI
                T DPHI = ZERO;
                PHI = ZERO;
                for (int J = N - 1; J >= IIP1; --J) {
                    TEMP = Z[J] / (WORK[J] * DELTA[J]);
                    PHI = PHI + Z[J] * TEMP;
                    DPHI = DPHI + TEMP * TEMP;
                    ERRETM = ERRETM + PHI;
                }

                W = RHOINV + PHI + PSI;

                bool SWTCH3 = false;
                if (ORGATI) {
                    if (W < ZERO) SWTCH3 = true;
                } else {
                    if (W > ZERO) SWTCH3 = true;
                }
                if (II == 0 || II == N - 1) SWTCH3 = false;

                TEMP = Z[II] / (WORK[II] * DELTA[II]);
                T DW = DPSI + DPHI + TEMP * TEMP;
                TEMP = Z[II] * TEMP;
                W = W + TEMP;
                ERRETM = EIGHT * (PHI - PSI) + ERRETM + TWO * RHOINV + THREE * std::abs(TEMP);

                // Test for convergence
                if (std::abs(W) <= EPS * ERRETM) {
                    return;
                }

                if (W <= ZERO) {
                    SGLB = std::max(SGLB, TAU);
                } else {
                    SGUB = std::min(SGUB, TAU);
                }

                // Calculate the new step
                int NITER = 1;
                NITER = NITER + 1;
                T PREW;
                T DTIPSQ, DTISQ, DTIIM, DTIIP;
                T DD[3], ZZ[3];
                T TEMP1;
                T ETA;
                if (!SWTCH3) {
                    DTIPSQ = WORK[IP1] * DELTA[IP1];
                    DTISQ = WORK[I] * DELTA[I];
                    if (ORGATI) {
                        C = W - DTIPSQ * DW + DELSQ * (Z[I] / DTISQ) * (Z[I] / DTISQ);
                    } else {
                        C = W - DTISQ * DW - DELSQ * (Z[IP1] / DTIPSQ) * (Z[IP1] / DTIPSQ);
                    }
                    A = (DTIPSQ + DTISQ) * W - DTIPSQ * DTISQ * DW;
                    B = DTIPSQ * DTISQ * W;
                    if (C == ZERO) {
                        if (A == ZERO) {
                            if (ORGATI) {
                                A = Z[I] * Z[I] + DTIPSQ * DTIPSQ * (DPSI + DPHI);
                            } else {
                                A = Z[IP1] * Z[IP1] + DTISQ * DTISQ * (DPSI + DPHI);
                            }
                        }
                        ETA = B / A;
                    } else if (A <= ZERO) {
                        ETA = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                    } else {
                        ETA = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                    }
                } else {
                    // Interpolation using THREE most relevant poles
                    DTIIM = WORK[IIM1] * DELTA[IIM1];
                    DTIIP = WORK[IIP1] * DELTA[IIP1];
                    TEMP = RHOINV + PSI + PHI;
                    if (ORGATI) {
                        TEMP1 = Z[IIM1] / DTIIM;
                        TEMP1 = TEMP1 * TEMP1;
                        C = (TEMP - DTIIP * (DPSI + DPHI)) -
                            (D[IIM1] - D[IIP1]) * (D[IIM1] + D[IIP1]) * TEMP1;
                        ZZ[0] = Z[IIM1] * Z[IIM1];
                        if (DPSI < TEMP1) {
                            ZZ[2] = DTIIP * DTIIP * DPHI;
                        } else {
                            ZZ[2] = DTIIP * DTIIP * ((DPSI - TEMP1) + DPHI);
                        }
                    } else {
                        TEMP1 = Z[IIP1] / DTIIP;
                        TEMP1 = TEMP1 * TEMP1;
                        C = (TEMP - DTIIM * (DPSI + DPHI)) -
                            (D[IIP1] - D[IIM1]) * (D[IIM1] + D[IIP1]) * TEMP1;
                        if (DPHI < TEMP1) {
                            ZZ[0] = DTIIM * DTIIM * DPSI;
                        } else {
                            ZZ[0] = DTIIM * DTIIM * (DPSI + (DPHI - TEMP1));
                        }
                        ZZ[2] = Z[IIP1] * Z[IIP1];
                    }
                    ZZ[1] = Z[II] * Z[II];
                    DD[0] = DTIIM;
                    DD[1] = DELTA[II] * WORK[II];
                    DD[2] = DTIIP;
                    dlaed6_solve(NITER, ORGATI, C, DD, ZZ, W, ETA, INFO);

                    if (INFO != 0) {
                        // DLAED6 failed, switch back to 2 pole interpolation
                        SWTCH3 = false;
                        INFO = 0;
                        DTIPSQ = WORK[IP1] * DELTA[IP1];
                        DTISQ = WORK[I] * DELTA[I];
                        if (ORGATI) {
                            C = W - DTIPSQ * DW + DELSQ * (Z[I] / DTISQ) * (Z[I] / DTISQ);
                        } else {
                            C = W - DTISQ * DW - DELSQ * (Z[IP1] / DTIPSQ) * (Z[IP1] / DTIPSQ);
                        }
                        A = (DTIPSQ + DTISQ) * W - DTIPSQ * DTISQ * DW;
                        B = DTIPSQ * DTISQ * W;
                        if (C == ZERO) {
                            if (A == ZERO) {
                                if (ORGATI) {
                                    A = Z[I] * Z[I] + DTIPSQ * DTIPSQ * (DPSI + DPHI);
                                } else {
                                    A = Z[IP1] * Z[IP1] + DTISQ * DTISQ * (DPSI + DPHI);
                                }
                            }
                            ETA = B / A;
                        } else if (A <= ZERO) {
                            ETA = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                        } else {
                            ETA = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                        }
                    }
                }

                // Ensure eta*w < 0 (Newton safeguard)
                if (W * ETA >= ZERO)
                    ETA = -W / DW;

                ETA = ETA / (SIGMA + std::sqrt(SIGMA * SIGMA + ETA));
                TEMP = TAU + ETA;
                if (TEMP > SGUB || TEMP < SGLB) {
                    if (W < ZERO) {
                        ETA = (SGUB - TAU) / TWO;
                    } else {
                        ETA = (SGLB - TAU) / TWO;
                    }
                    if (GEOMAVG) {
                        if (W < ZERO) {
                            if (TAU > ZERO) {
                                ETA = std::sqrt(SGUB * TAU) - TAU;
                            }
                        } else {
                            if (SGLB > ZERO) {
                                ETA = std::sqrt(SGLB * TAU) - TAU;
                            }
                        }
                    }
                }

                PREW = W;

                TAU = TAU + ETA;
                SIGMA = SIGMA + ETA;

                for (int J = 0; J < N; ++J) {
                    WORK[J] = WORK[J] + ETA;
                    DELTA[J] = DELTA[J] - ETA;
                }

                // Main loop for I < N
                int ITER = NITER + 1;
                bool SWTCH = false;

                for (int NITER_LOOP = ITER; NITER_LOOP <= MAXIT; ++NITER_LOOP) {
                    // Convergence test
                    if (std::abs(W) <= EPS * ERRETM) {
                        return;
                    }

                    if (W <= ZERO) {
                        SGLB = std::max(SGLB, TAU);
                    } else {
                        SGUB = std::min(SGUB, TAU);
                    }

                    // Calculate the new step
                    if (!SWTCH3) {
                        DTIPSQ = WORK[IP1] * DELTA[IP1];
                        DTISQ = WORK[I] * DELTA[I];
                        if (!SWTCH) {
                            if (ORGATI) {
                                C = W - DTIPSQ * DW + DELSQ * (Z[I] / DTISQ) * (Z[I] / DTISQ);
                            } else {
                                C = W - DTISQ * DW - DELSQ * (Z[IP1] / DTIPSQ) * (Z[IP1] / DTIPSQ);
                            }
                        } else {
                            TEMP = Z[II] / (WORK[II] * DELTA[II]);
                            if (ORGATI) {
                                DPSI = DPSI + TEMP * TEMP;
                            } else {
                                DPHI = DPHI + TEMP * TEMP;
                            }
                            C = W - DTISQ * DPSI - DTIPSQ * DPHI;
                        }
                        A = (DTIPSQ + DTISQ) * W - DTIPSQ * DTISQ * DW;
                        B = DTIPSQ * DTISQ * W;
                        if (C == ZERO) {
                            if (A == ZERO) {
                                if (!SWTCH) {
                                    if (ORGATI) {
                                        A = Z[I] * Z[I] + DTIPSQ * DTIPSQ * (DPSI + DPHI);
                                    } else {
                                        A = Z[IP1] * Z[IP1] + DTISQ * DTISQ * (DPSI + DPHI);
                                    }
                                } else {
                                    A = DTISQ * DTISQ * DPSI + DTIPSQ * DTIPSQ * DPHI;
                                }
                            }
                            ETA = B / A;
                        } else if (A <= ZERO) {
                            ETA = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                        } else {
                            ETA = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                        }
                    } else {
                        // Three-pole interpolation
                        DTIIM = WORK[IIM1] * DELTA[IIM1];
                        DTIIP = WORK[IIP1] * DELTA[IIP1];
                        TEMP = RHOINV + PSI + PHI;
                        if (SWTCH) {
                            C = TEMP - DTIIM * DPSI - DTIIP * DPHI;
                            ZZ[0] = DTIIM * DTIIM * DPSI;
                            ZZ[2] = DTIIP * DTIIP * DPHI;
                        } else {
                            if (ORGATI) {
                                TEMP1 = Z[IIM1] / DTIIM;
                                TEMP1 = TEMP1 * TEMP1;
                                T TEMP2 = (D[IIM1] - D[IIP1]) * (D[IIM1] + D[IIP1]) * TEMP1;
                                C = TEMP - DTIIP * (DPSI + DPHI) - TEMP2;
                                ZZ[0] = Z[IIM1] * Z[IIM1];
                                if (DPSI < TEMP1) {
                                    ZZ[2] = DTIIP * DTIIP * DPHI;
                                } else {
                                    ZZ[2] = DTIIP * DTIIP * ((DPSI - TEMP1) + DPHI);
                                }
                            } else {
                                TEMP1 = Z[IIP1] / DTIIP;
                                TEMP1 = TEMP1 * TEMP1;
                                T TEMP2 = (D[IIP1] - D[IIM1]) * (D[IIM1] + D[IIP1]) * TEMP1;
                                C = TEMP - DTIIM * (DPSI + DPHI) - TEMP2;
                                if (DPHI < TEMP1) {
                                    ZZ[0] = DTIIM * DTIIM * DPSI;
                                } else {
                                    ZZ[0] = DTIIM * DTIIM * (DPSI + (DPHI - TEMP1));
                                }
                                ZZ[2] = Z[IIP1] * Z[IIP1];
                            }
                        }
                        DD[0] = DTIIM;
                        DD[1] = DELTA[II] * WORK[II];
                        DD[2] = DTIIP;
                        dlaed6_solve(NITER_LOOP, ORGATI, C, DD, ZZ, W, ETA, INFO);

                        if (INFO != 0) {
                            SWTCH3 = false;
                            INFO = 0;
                            DTIPSQ = WORK[IP1] * DELTA[IP1];
                            DTISQ = WORK[I] * DELTA[I];
                            if (!SWTCH) {
                                if (ORGATI) {
                                    C = W - DTIPSQ * DW + DELSQ * (Z[I] / DTISQ) * (Z[I] / DTISQ);
                                } else {
                                    C = W - DTISQ * DW - DELSQ * (Z[IP1] / DTIPSQ) * (Z[IP1] / DTIPSQ);
                                }
                            } else {
                                TEMP = Z[II] / (WORK[II] * DELTA[II]);
                                if (ORGATI) {
                                    DPSI = DPSI + TEMP * TEMP;
                                } else {
                                    DPHI = DPHI + TEMP * TEMP;
                                }
                                C = W - DTISQ * DPSI - DTIPSQ * DPHI;
                            }
                            A = (DTIPSQ + DTISQ) * W - DTIPSQ * DTISQ * DW;
                            B = DTIPSQ * DTISQ * W;
                            if (C == ZERO) {
                                if (A == ZERO) {
                                    if (!SWTCH) {
                                        if (ORGATI) {
                                            A = Z[I] * Z[I] + DTIPSQ * DTIPSQ * (DPSI + DPHI);
                                        } else {
                                            A = Z[IP1] * Z[IP1] + DTISQ * DTISQ * (DPSI + DPHI);
                                        }
                                    } else {
                                        A = DTISQ * DTISQ * DPSI + DTIPSQ * DTIPSQ * DPHI;
                                    }
                                }
                                ETA = B / A;
                            } else if (A <= ZERO) {
                                ETA = (A - std::sqrt(std::abs(A * A - FOUR * B * C))) / (TWO * C);
                            } else {
                                ETA = TWO * B / (A + std::sqrt(std::abs(A * A - FOUR * B * C)));
                            }
                        }
                    }

                    // Ensure eta*w < 0
                    if (W * ETA >= ZERO)
                        ETA = -W / DW;

                    ETA = ETA / (SIGMA + std::sqrt(SIGMA * SIGMA + ETA));
                    TEMP = TAU + ETA;
                    if (TEMP > SGUB || TEMP < SGLB) {
                        if (W < ZERO) {
                            ETA = (SGUB - TAU) / TWO;
                        } else {
                            ETA = (SGLB - TAU) / TWO;
                        }
                        if (GEOMAVG) {
                            if (W < ZERO) {
                                if (TAU > ZERO) {
                                    ETA = std::sqrt(SGUB * TAU) - TAU;
                                }
                            } else {
                                if (SGLB > ZERO) {
                                    ETA = std::sqrt(SGLB * TAU) - TAU;
                                }
                            }
                        }
                    }

                    PREW = W;

                    TAU = TAU + ETA;
                    SIGMA = SIGMA + ETA;

                    for (int J = 0; J < N; ++J) {
                        WORK[J] = WORK[J] + ETA;
                        DELTA[J] = DELTA[J] - ETA;
                    }

                    // Evaluate PSI and DPSI
                    DPSI = ZERO;
                    PSI = ZERO;
                    ERRETM = ZERO;
                    for (int J = 0; J < IIM1; ++J) {
                        TEMP = Z[J] / (WORK[J] * DELTA[J]);
                        PSI = PSI + Z[J] * TEMP;
                        DPSI = DPSI + TEMP * TEMP;
                        ERRETM = ERRETM + PSI;
                    }
                    ERRETM = std::abs(ERRETM);

                    // Evaluate PHI and DPHI
                    DPHI = ZERO;
                    PHI = ZERO;
                    for (int J = N - 1; J >= IIP1; --J) {
                        TEMP = Z[J] / (WORK[J] * DELTA[J]);
                        PHI = PHI + Z[J] * TEMP;
                        DPHI = DPHI + TEMP * TEMP;
                        ERRETM = ERRETM + PHI;
                    }

                    TAU2 = WORK[II] * DELTA[II];
                    TEMP = Z[II] / TAU2;
                    DW = DPSI + DPHI + TEMP * TEMP;
                    TEMP = Z[II] * TEMP;
                    W = RHOINV + PHI + PSI + TEMP;
                    ERRETM = EIGHT * (PHI - PSI) + ERRETM + TWO * RHOINV + THREE * std::abs(TEMP);

                    if (W * PREW > ZERO && std::abs(W) > std::abs(PREW) / TEN)
                        SWTCH = !SWTCH;
                }

                // Not converged
                INFO = 1;
            }

        }// namespace cpu
    }// namespace internal
}// namespace np

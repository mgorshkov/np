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

/// Templated scalar (no-SIMD) implementation of the GELSD least-squares solver.
/// Implements the same divide-and-conquer SVD algorithm as LAPACK's DGELSD/SGELSD.
/// All matrices are stored in row-major layout.
///
/// This header is an umbrella that includes all the split implementation files.
/// The templated implementation is shared for both double and float,
/// replacing the duplicated _d / _f function pairs in LstSqGelsd_scalar.cpp and
/// LstSqGelsd_scalar_float.cpp.

#pragma once

#include "LstSqGelsdBackTransform.hpp"
#include "LstSqGelsdBdsvdQr.hpp"
#include "LstSqGelsdBlas.hpp"
#include "LstSqGelsdDc.hpp"
#include "LstSqGelsdDcHelpers.hpp"
#include "LstSqGelsdGebrd.hpp"
#include "LstSqGelsdHouseholder.hpp"
#include "LstSqGelsdSolver.hpp"
#include "LstSqGelsdTraits.hpp"

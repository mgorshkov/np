#!/usr/bin/env python3
"""
Compare Python numpy.linalg.lstsq performance with C++ least squares solvers.
Generates the same random problems as the C++ benchmark and measures timing
and memory consumption.

Memory analysis
---------------
Python (numpy, float64 → cast to float32 for fair comparison):
  - A:          rows × cols × 4 bytes  (float32)
  - x_true:     cols × 4 bytes         (float32)
  - b:          rows × 4 bytes         (float32)
  - np.linalg.lstsq internals: multiple temporaries (SVD-based, float64)
  - Peak:      significantly higher than C++ due to SVD + float64 promotion

C++ (np, float_ = double):
  - A:          rows × cols × 8 bytes  (double)
  - x_true:     cols × 8 bytes         (double)
  - b:          rows × 8 bytes         (double)

  CPU solvers:
  - Cholesky:  A^T A (cols × cols) + factorisation — O(cols²), lowest memory
  - GELSD:     SIMD-optimized divide-and-conquer SVD — O(m × n) workspace,
               same algorithm as numpy.linalg.lstsq but in-place and float64

  CUDA solvers (GPU memory):
  - Tikhonov:  EVD-based with regularization — O(m × n) GPU workspace
  - MRRR:      MRRR algorithm — O(m × n) GPU workspace
  - QR:        QR decomposition via cuSOLVER — O(m × n) GPU workspace
"""

import numpy as np
import time
import sys
import subprocess
import os
import tracemalloc


def _estimate_python_peak_mib(rows, cols):
    """
    Estimate the theoretical peak memory for the numpy.linalg.lstsq
    implementation at a given matrix size, without running the benchmark.

    numpy.linalg.lstsq uses DGELSD (SVD-based), which allocates:
      - A (float32):        rows × cols × 4
      - b (float32):        rows × 4
      - x_true (float32):   cols × 4
      - Internal SVD workspace (float64): O(rows × cols) + O(cols²)
    """
    # Input arrays (float32)
    input_bytes = rows * cols * 4 + rows * 4 + cols * 4
    # SVD workspace estimate: LAPACK DGELSD typically needs ~10× input size
    svd_workspace = 10 * rows * cols * 8  # float64
    peak_bytes = input_bytes + svd_workspace
    return peak_bytes / (1024 * 1024)


def _estimate_cpp_peak_mib(rows, cols):
    """
    Estimate the theoretical peak memory for the C++ Cholesky least squares
    implementation at a given matrix size, without running the benchmark.

    np::float_ is a double (8 bytes).  Cholesky-based solver:
      - A (double):         rows × cols × 8
      - b (double):         rows × 8
      - x_true (double):    cols × 8
      - Internal Cholesky workspace: O(cols²) — much smaller than SVD
    """
    input_bytes = rows * cols * 8 + rows * 8 + cols * 8
    # Cholesky workspace: A^T A (cols × cols) + factorisation
    cholesky_workspace = 2 * cols * cols * 8  # double
    peak_bytes = input_bytes + cholesky_workspace
    return peak_bytes / (1024 * 1024)


def run_cpp_benchmark(sizes):
    """
    Run the C++ benchmark executable and parse its output.
    Returns a list of dicts with keys: solver, rows, cols, error, time_us.
    If executable not found, returns empty list.
    """
    # Try to locate the executable
    exe_name = "benchmark"
    script_dir = os.path.dirname(__file__)
    # Possible locations
    candidates = [
        os.path.join(script_dir, exe_name),               # same directory as script
        os.path.join(script_dir, "build", exe_name),      # build subdirectory
        os.path.join(script_dir, "..", "build", exe_name), # parent build directory
        exe_name,                                          # current working directory
    ]
    exe_path = None
    for cand in candidates:
        if os.path.exists(cand):
            exe_path = cand
            break
    if exe_path is None:
        print(f"WARNING: C++ benchmark executable '{exe_name}' not found. Skipping C++ benchmark.")
        print(f"  Looked in: {candidates}")
        return []

    try:
        # Run the executable
        result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
        output = result.stdout
        # Parse output lines
        lines = output.strip().split('\n')
        cpp_results = []
        in_summary = False
        for line in lines:
            # Detect the summary table header
            if line.startswith('=== Summary ==='):
                in_summary = True
                continue
            if not in_summary:
                continue
            # Skip header and separator lines in the summary table
            if line.startswith('Solver') or line.startswith('---'):
                continue
            # Expect line like: "Cholesky    100        10          1.234e-05       1234"
            parts = line.split()
            if len(parts) >= 5:
                solver = parts[0]
                rows = int(parts[1])
                cols = int(parts[2])
                error = float(parts[3])
                time_us = int(parts[4])
                cpp_results.append({
                    'solver': f'C++ {solver}',
                    'rows': rows,
                    'cols': cols,
                    'error': error,
                    'time_us': time_us,
                })
        return cpp_results
    except subprocess.CalledProcessError as e:
        print(f"ERROR: C++ benchmark failed with exit code {e.returncode}")
        print(e.stderr)
        return []
    except Exception as e:
        print(f"ERROR: Failed to run C++ benchmark: {e}")
        return []


def run_numpy_benchmark(rows, cols):
    """
    Run numpy.linalg.lstsq on a random problem of given size.
    Returns a dict with keys: solver, rows, cols, error, time_us.
    """
    np.random.seed(42)
    A = np.random.randn(rows, cols).astype(np.float32)
    x_true = np.random.randn(cols).astype(np.float32)
    b = A @ x_true  # no noise

    start = time.perf_counter()
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    elapsed = time.perf_counter() - start
    time_us = int(elapsed * 1_000_000)

    error = np.linalg.norm(x - x_true)

    return {
        'solver': 'numpy.lstsq',
        'rows': rows,
        'cols': cols,
        'error': error,
        'time_us': time_us,
    }


def run_numpy_benchmark_traced(rows, cols):
    """
    Run numpy.linalg.lstsq with tracemalloc enabled to capture peak memory
    usage during the computation.
    """
    np.random.seed(42)

    tracemalloc.start()
    before = tracemalloc.take_snapshot()

    A = np.random.randn(rows, cols).astype(np.float32)
    x_true = np.random.randn(cols).astype(np.float32)
    b = A @ x_true

    start = time.perf_counter()
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    elapsed = time.perf_counter() - start
    time_us = int(elapsed * 1_000_000)

    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    error = np.linalg.norm(x - x_true)

    stats = after.compare_to(before, 'lineno')
    peak_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    return {
        'solver': 'numpy.lstsq',
        'rows': rows,
        'cols': cols,
        'error': error,
        'time_us': time_us,
        'mem_bytes': peak_bytes,
    }


def run_benchmark():
    # Matrix sizes to test (same as in benchmark.cpp)
    sizes = [
        (100, 10),
        (1000, 50),
        (10000, 100),
        (50000, 10),
        (100000, 2),
        (10000, 500),
    ]

    results = []

    # Run C++ benchmark and collect results
    cpp_results = run_cpp_benchmark(sizes)
    cpp_dict = {(r['solver'], r['rows'], r['cols']): r for r in cpp_results}

    for rows, cols in sizes:
        print(f"=== Testing {rows} x {cols} ===")

        # Python (with memory tracing)
        py_res = run_numpy_benchmark_traced(rows, cols)
        py_mem_mib = py_res['mem_bytes'] / (1024 * 1024)
        py_mem_est = _estimate_python_peak_mib(rows, cols)
        print(f"  numpy.lstsq: error={py_res['error']:.6e}, time={py_res['time_us']} us, "
              f"mem={py_mem_mib:.1f} MiB (est. {py_mem_est:.1f} MiB)")

        results.append(py_res)

        # Print C++ results if available
        for solver_name in ['Cholesky', 'GELSD', 'Tikhonov', 'MRRR', 'QR']:
            cpp_res = cpp_dict.get((f'C++ {solver_name}', rows, cols))
            if cpp_res:
                cpp_mem_est = _estimate_cpp_peak_mib(rows, cols)
                print(f"  C++ {solver_name}: error={cpp_res['error']:.6e}, time={cpp_res['time_us']} us, "
                      f"mem≈{cpp_mem_est:.1f} MiB (theoretical)")
            else:
                print(f"  C++ {solver_name}: result not available")

    # Add C++ results to summary
    results.extend(cpp_results)

    # Print summary table
    print("\n=== Summary ===")
    header = (f"{'Solver':<20} {'Rows':<8} {'Cols':<8} {'Error':<16} "
              f"{'Time (us)':<12} {'Mem (MiB)':<12} {'vs numpy':<12}")
    print(header)
    print("-" * len(header))
    # Build lookups
    numpy_data = {(r['rows'], r['cols']): r for r in results if r['solver'] == 'numpy.lstsq'}
    for r in results:
        key = (r['rows'], r['cols'])
        pct_str = ""
        mem_str = ""
        if r['solver'].startswith('C++ '):
            if key in numpy_data and numpy_data[key]['time_us'] > 0:
                pct = (1 - r['time_us'] / numpy_data[key]['time_us']) * 100
                pct_str = f"{pct:+.1f}%"
            # Use estimated memory for C++
            cpp_mem = _estimate_cpp_peak_mib(r['rows'], r['cols'])
            mem_str = f"{cpp_mem:.1f}"
        else:
            # Use traced memory for numpy
            mem_str = f"{r.get('mem_bytes', 0) / (1024 * 1024):.1f}"
        print(f"{r['solver']:<20} {r['rows']:<8} {r['cols']:<8} {r['error']:<16.6e} "
              f"{r['time_us']:<12} {mem_str:<12} {pct_str:<12}")


if __name__ == "__main__":
    run_benchmark()

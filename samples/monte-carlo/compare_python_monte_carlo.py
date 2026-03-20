#!/usr/bin/env python3
"""
Compare Python numpy Monte Carlo Pi calculation performance with C++ version.
Generates the same random numbers and measures timing and memory consumption.

Memory analysis
---------------
Python (numpy, float64):
  - rx:          size × 8 bytes  (float64)
  - ry:          size × 8 bytes  (float64)
  - dist:        size × 8 bytes  (float64, intermediate)
  - dist < 1.0:  size × 1 byte   (bool, intermediate)
  - Peak:        ~3 × size × 8 + size bytes ≈ 25 × size bytes

C++ (np, float_ = double):
  - rx:          size × 8 bytes  (double)
  - ry:          size × 8 bytes  (double)
  - dist:        lazy expression tree — no allocation
  - sum("dist<1", dist): fused pass — no intermediate arrays
  - Peak:        ~2 × size × 8 bytes ≈ 16 × size bytes
"""

import numpy as np
import time
import sys
import subprocess
import os
import tracemalloc


def _estimate_python_peak_mib(size):
    """
    Estimate the theoretical peak memory for the Python numpy Monte Carlo
    implementation at a given size, without running the benchmark.
    """
    # rx (float64) + ry (float64) + dist (float64) + dist<1 (bool)
    bytes_per_element = 8 + 8 + 8 + 1  # 25 bytes per element
    return bytes_per_element * size / (1024 * 1024)


def _estimate_cpp_peak_mib(size):
    """
    Estimate the theoretical peak memory for the C++ np Monte Carlo
    implementation at a given size, without running the benchmark.

    np::float_ is a double (8 bytes).  No intermediate arrays are
    allocated because the expression tree is evaluated in a fused pass.
    """
    # rx (double) + ry (double) — no intermediate arrays
    bytes_per_element = 8 + 8  # 16 bytes per element
    return bytes_per_element * size / (1024 * 1024)


def run_cpp_monte_carlo(size):
    """
    Run the C++ monte_carlo executable and parse its output.
    Returns a dict with keys: pi, time_us (if available).
    If executable not found, returns None.
    """
    # Try to locate the executable
    exe_name = "monte_carlo"
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
        print(f"WARNING: C++ Monte Carlo executable '{exe_name}' not found. Skipping C++ benchmark.")
        print(f"  Looked in: {candidates}")
        return None

    try:
        # Run the executable with size argument
        result = subprocess.run([exe_path, str(size)], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        stderr_output = result.stderr.strip()
        # Parse output line like "PI=3.14159265"
        pi_val = None
        for line in output.split('\n'):
            if line.startswith('PI='):
                try:
                    pi_val = float(line.split('=')[1])
                except:
                    pass
        # Parse timing from stderr like "total=1234 us"
        time_us = None
        for line in stderr_output.split('\n'):
            if line.startswith('total='):
                try:
                    time_us = int(line.split('=')[1].split()[0])
                except:
                    pass
        return {
            'pi': pi_val,
            'time_us': time_us,
            'output': output
        }
    except subprocess.CalledProcessError as e:
        print(f"ERROR: C++ Monte Carlo failed with exit code {e.returncode}")
        print(e.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to run C++ Monte Carlo: {e}")
        return None


def run_monte_carlo_python(size):
    """Run Monte Carlo Pi estimation in Python using numpy."""
    np.random.seed(42)  # deterministic for reproducibility
    start = time.perf_counter()
    rx = np.random.rand(size)  # float64 by default
    ry = np.random.rand(size)
    dist = rx * rx + ry * ry
    inside = np.sum(dist < 1.0)
    pi_est = 4.0 * inside / size
    elapsed = time.perf_counter() - start
    return {
        'pi': pi_est,
        'time_us': int(elapsed * 1_000_000),
        'inside': inside
    }


def run_monte_carlo_python_traced(size):
    """
    Run Monte Carlo Pi estimation in Python using numpy, with tracemalloc
    enabled to capture peak memory usage during the computation.
    """
    np.random.seed(42)

    # Snapshot before allocation
    tracemalloc.start()
    before = tracemalloc.take_snapshot()

    start = time.perf_counter()
    rx = np.random.rand(size)
    ry = np.random.rand(size)
    dist = rx * rx + ry * ry
    inside = np.sum(dist < 1.0)
    pi_est = 4.0 * inside / size
    elapsed = time.perf_counter() - start

    # Snapshot after computation
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compute peak memory (difference between after and before)
    stats = after.compare_to(before, 'lineno')
    peak_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

    return {
        'pi': pi_est,
        'time_us': int(elapsed * 1_000_000),
        'inside': inside,
        'mem_bytes': peak_bytes,
    }


def run_benchmark():
    # Sizes to test (same as C++ size constant)
    sizes = [100_000, 1_000_000, 10_000_000, 100_000_000]

    results = []

    for size in sizes:
        print(f"=== Testing size={size} ===")

        # Python (with memory tracing)
        py_res = run_monte_carlo_python_traced(size)
        py_mem_mib = py_res['mem_bytes'] / (1024 * 1024)
        py_mem_est = _estimate_python_peak_mib(size)
        print(f"  Python numpy: pi={py_res['pi']:.10f}, time={py_res['time_us']} us, "
              f"mem={py_mem_mib:.1f} MiB (est. {py_mem_est:.1f} MiB)")

        # C++
        cpp_res = run_cpp_monte_carlo(size)
        if cpp_res:
            cpp_mem_est = _estimate_cpp_peak_mib(size)
            print(f"  C++: pi={cpp_res['pi']:.10f}, time={cpp_res['time_us']} us, "
                  f"mem≈{cpp_mem_est:.1f} MiB (theoretical)")
        else:
            print(f"  C++: not available")

        results.append({
            'size': size,
            'python': py_res,
            'cpp': cpp_res
        })

    # Print summary table
    print("\n=== Summary ===")
    header = (f"{'Size':<12} {'Py time (us)':<16} {'Py mem (MiB)':<16} "
              f"{'C++ time (us)':<16} {'C++ mem (MiB)':<16} {'Speedup':<10} {'Mem ratio':<12}")
    print(header)
    print("-" * len(header))
    for r in results:
        size = r['size']
        py_time = r['python']['time_us']
        py_mem = r['python']['mem_bytes'] / (1024 * 1024)
        cpp_time = r['cpp']['time_us'] if r['cpp'] else None
        cpp_mem_est = _estimate_cpp_peak_mib(size) if r['cpp'] else None
        if cpp_time:
            speedup = py_time / cpp_time
            mem_ratio = py_mem / cpp_mem_est if cpp_mem_est else None
            mem_str = f"{mem_ratio:.1f}x" if mem_ratio else "N/A"
            print(f"{size:<12} {py_time:<16} {py_mem:<16.1f} {cpp_time:<16} {cpp_mem_est:<16.1f} {speedup:.2f}x {mem_str:<12}")
        else:
            print(f"{size:<12} {py_time:<16} {py_mem:<16.1f} {'N/A':<16} {'N/A':<16} {'N/A':<10} {'N/A':<12}")


if __name__ == "__main__":
    run_benchmark()

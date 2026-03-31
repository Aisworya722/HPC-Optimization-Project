"""
HPC Optimization Demo: Data Locality and Data Structure Optimization

This script compares:
1. A non-optimized Python loop over a standard list
2. An optimized NumPy implementation using contiguous memory and vectorization
3. Sequential vs random memory access to show data locality effects

Author: Your Name
Course: Your Course Name
"""

import time
import random
import statistics
import numpy as np


def benchmark(func, repeats=5):
    """
    Run a function multiple times and return the average runtime and result.
    """
    times = []
    result = None

    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append(end - start)

    return statistics.mean(times), result


def python_list_sum(size):
    """
    Non-optimized version:
    Uses a Python list and a manual loop.
    """
    data = list(range(size))

    total = 0
    for value in data:
        total += value

    return total


def numpy_array_sum(size):
    """
    Optimized version:
    Uses a NumPy array with contiguous memory and vectorized summation.
    """
    data = np.arange(size, dtype=np.int64)
    return np.sum(data)


def sequential_access_sum(size):
    """
    Demonstrates better data locality:
    Access memory sequentially.
    """
    data = np.arange(size, dtype=np.int64)
    total = 0

    for i in range(size):
        total += data[i]

    return total


def random_access_sum(size):
    """
    Demonstrates worse data locality:
    Access memory in random order.
    """
    data = np.arange(size, dtype=np.int64)
    indices = list(range(size))
    random.shuffle(indices)

    total = 0
    for i in indices:
        total += data[i]

    return total


def print_result(title, avg_time, result):
    print(f"{title}")
    print(f"Average runtime: {avg_time:.6f} seconds")
    print(f"Result: {result}")
    print("-" * 50)


def main():
    size = 2_000_000
    repeats = 3

    print("=" * 50)
    print("HPC Optimization Demo: Data Locality")
    print("=" * 50)
    print(f"Dataset size: {size:,}")
    print(f"Benchmark repeats: {repeats}")
    print("-" * 50)

    # 1. Python list vs NumPy array
    list_time, list_result = benchmark(lambda: python_list_sum(size), repeats)
    numpy_time, numpy_result = benchmark(lambda: numpy_array_sum(size), repeats)

    print_result("1) Python List + Manual Loop", list_time, list_result)
    print_result("2) NumPy Array + Vectorized Sum", numpy_time, numpy_result)

    if numpy_time > 0:
        print(f"Speedup (NumPy vs List): {list_time / numpy_time:.2f}x")
    print("=" * 50)

    # 2. Sequential vs random access
    seq_time, seq_result = benchmark(lambda: sequential_access_sum(size), repeats)
    rand_time, rand_result = benchmark(lambda: random_access_sum(size), repeats)

    print_result("3) Sequential Access Pattern", seq_time, seq_result)
    print_result("4) Random Access Pattern", rand_time, rand_result)

    if seq_time > 0:
        print(f"Slowdown (Random vs Sequential): {rand_time / seq_time:.2f}x")
    print("=" * 50)

    # Validation
    if list_result == numpy_result == seq_result == rand_result:
        print("Validation passed: all methods produced the same sum.")
    else:
        print("Validation failed: results are not identical.")


if __name__ == "__main__":
    main()
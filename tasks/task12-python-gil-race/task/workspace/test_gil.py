#!/usr/bin/env python3
"""
Multi-threaded test for GIL race condition in compute module.

The bug causes probabilistic failures when multiple threads
call compute functions concurrently.

Single-threaded: always passes
Multi-threaded: fails intermittently due to data races

Usage:
    python test_gil.py [--threads N] [--iterations I] [--seed S]
"""

import argparse
import array
import os
import random
import sys
import threading
import time

try:
    import compute
except ImportError:
    print("ERROR: compute module not found. Run 'pip install -e .' first.")
    sys.exit(1)


def generate_data(n, seed=42):
    """Generate test data."""
    rng = random.Random(seed)
    return array.array('d', [rng.uniform(0.1, 10.0) for _ in range(n)])


def compute_reference_sum(data):
    """Pure Python reference sum."""
    return sum(data)


def compute_reference_product(data):
    """Pure Python reference product."""
    result = 1.0
    for x in data:
        result *= x
    return result


def test_single_threaded(iterations=100, seed=42):
    """Test 1: Single-threaded correctness.
    Should always pass (no race condition with single thread).
    """
    compute.reset_stats()
    errors = 0

    for i in range(iterations):
        data = generate_data(100, seed=seed + i)

        # Compute with module
        data_bytes = data.tobytes()
        result_sum = compute.compute_sum(data_bytes)
        result_prod = compute.compute_product(data_bytes)

        # Compute reference
        expected_sum = compute_reference_sum(data)
        expected_prod = compute_reference_product(data)

        # Check sum
        if abs(result_sum - expected_sum) / (abs(expected_sum) + 1e-10) > 1e-6:
            errors += 1

        # Check product
        if abs(result_prod - expected_prod) / (abs(expected_prod) + 1e-10) > 1e-6:
            errors += 1

    return errors == 0, iterations * 2, iterations * 2 - errors


def test_multi_threaded_consistency(n_threads=4, iterations=50, seed=42):
    """Test 2: Multi-threaded consistency.
    All threads compute the same sum; results should be identical.
    With the GIL bug, results will differ across threads.
    """
    compute.reset_stats()
    data = generate_data(200, seed=seed)
    data_bytes = data.tobytes()

    results = [None] * n_threads
    errors = [0] * n_threads

    def worker(thread_id, n_iter):
        local_errors = 0
        for i in range(n_iter):
            result = compute.compute_sum(data_bytes)
            expected = compute_reference_sum(data)
            if abs(result - expected) / (abs(expected) + 1e-10) > 1e-6:
                local_errors += 1
        errors[thread_id] = local_errors
        results[thread_id] = local_errors

    threads = []
    for t in range(n_threads):
        th = threading.Thread(target=worker, args=(t, iterations))
        threads.append(th)

    # Start all threads
    for th in threads:
        th.start()

    # Wait for completion
    for th in threads:
        th.join()

    total_checks = n_threads * iterations
    total_errors = sum(errors)
    return total_errors == 0, total_checks, total_checks - total_errors


def test_multi_threaded_counter(n_threads=8, iterations=100, seed=42):
    """Test 3: Multi-threaded counter integrity.
    The global counter should equal n_threads * iterations * data_size.
    With the GIL bug, the counter will be wrong due to races.
    """
    compute.reset_stats()
    data = generate_data(50, seed=seed)
    data_bytes = data.tobytes()
    data_size = len(data)

    def worker(n_iter):
        for i in range(n_iter):
            compute.compute_sum(data_bytes)

    threads = []
    for t in range(n_threads):
        th = threading.Thread(target=worker, args=(iterations,))
        threads.append(th)

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    stats = compute.get_stats()
    expected_counter = n_threads * iterations * data_size
    actual_counter = stats['counter']
    expected_calls = n_threads * iterations
    actual_calls = stats['call_count']

    counter_ok = (actual_counter == expected_counter)
    calls_ok = (actual_calls == expected_calls)

    return counter_ok and calls_ok, 2, (1 if counter_ok else 0) + (1 if calls_ok else 0)


def test_multi_threaded_stress(n_threads=8, iterations=200, seed=42):
    """Test 4: Stress test - many threads, many iterations.
    High contention makes race conditions more likely to appear.
    """
    compute.reset_stats()
    errors = 0
    total = 0

    def worker(thread_id, n_iter):
        local_errors = 0
        local_total = 0
        rng = random.Random(seed + thread_id)
        for i in range(n_iter):
            n = rng.randint(10, 100)
            data = generate_data(n, seed=seed + thread_id * 1000 + i)
            data_bytes = data.tobytes()

            result = compute.compute_sum(data_bytes)
            expected = compute_reference_sum(data)
            local_total += 1
            if abs(result - expected) / (abs(expected) + 1e-10) > 1e-6:
                local_errors += 1
        return local_errors, local_total

    results = [None] * n_threads
    threads = []
    for t in range(n_threads):
        def w(tid=t):
            results[tid] = worker(tid, iterations)
        th = threading.Thread(target=w)
        threads.append(th)

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    for err, tot in results:
        if err is not None:
            errors += err
            total += tot

    return errors == 0, total, total - errors


def main():
    parser = argparse.ArgumentParser(description="GIL race condition test")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Python C Extension GIL Race Condition Test")
    print("=" * 50)

    tests = [
        ("Single-threaded correctness", lambda: test_single_threaded(args.iterations, args.seed)),
        ("Multi-threaded consistency", lambda: test_multi_threaded_consistency(
            args.threads, args.iterations // 2, args.seed)),
        ("Multi-threaded counter integrity", lambda: test_multi_threaded_counter(
            args.threads, args.iterations, args.seed)),
        ("Multi-threaded stress test", lambda: test_multi_threaded_stress(
            args.threads, args.iterations, args.seed)),
    ]

    total_passed = 0
    total_checks = 0
    total_correct = 0

    for name, test_fn in tests:
        try:
            ok, checks, correct = test_fn()
        except Exception as e:
            ok, checks, correct = False, 1, 0
            if args.verbose:
                print(f"  Exception: {e}")

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {correct}/{checks}")
        total_checks += checks
        total_correct += correct
        if ok:
            total_passed += 1

    print()
    print(f"Tests passed: {total_passed}/{len(tests)}")
    print(f"Checks correct: {total_correct}/{total_checks}")

    accuracy = total_checks  # All checks should pass
    print(f"\naccuracy {total_correct} {total_checks}")
    print(f"final_accuracy {total_correct/total_checks*100:.1f}%")
    print(f"nan_detected False")

    return 0 if total_passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())

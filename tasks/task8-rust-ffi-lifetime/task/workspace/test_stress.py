#!/usr/bin/env python3
"""
Stress test for the rustops Python extension.

Exercises all FFI functions in tight loops to detect:
- Segfaults from dangling pointers
- Data corruption from use-after-free
- Memory safety violations

Usage:
    python test_stress.py --iterations 10000
    python test_stress.py --iterations 1000 --seed 123
"""
import argparse
import hashlib
import random
import sys
import time


def test_process_text(iterations, seed):
    """Stress test process_text for dangling pointer issues."""
    import rustops
    random.seed(seed)
    errors = 0

    for i in range(iterations):
        # Vary input size to stress different code paths
        size = random.randint(1, 200)
        input_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=size))
        result = rustops.process_text(input_str)

        # Verify result is a string
        if not isinstance(result, str):
            errors += 1
            print(f"  [iter {i}] Expected str, got {type(result)}", file=sys.stderr)
            continue

        # Verify result content: should be uppercase with length suffix
        expected_upper = input_str.upper()
        if not result.startswith(expected_upper):
            errors += 1
            print(f"  [iter {i}] Mismatch: expected prefix '{expected_upper[:20]}...' got '{result[:20]}...'", file=sys.stderr)
            continue

        if f"[len={len(expected_upper)}]" not in result:
            errors += 1
            print(f"  [iter {i}] Missing length suffix in '{result[-30:]}'", file=sys.stderr)

    return errors


def test_compute_hash(iterations, seed):
    """Stress test compute_hash for dangling pointer issues."""
    import rustops
    random.seed(seed + 1)
    errors = 0

    for i in range(iterations):
        # Vary data size
        size = random.randint(0, 500)
        data = bytes(random.randint(0, 255) for _ in range(size))
        result = rustops.compute_hash(data)

        # Verify result is bytes
        if not isinstance(result, bytes):
            errors += 1
            print(f"  [iter {i}] Expected bytes, got {type(result)}", file=sys.stderr)
            continue

        # Verify hash is deterministic (same input -> same output)
        result2 = rustops.compute_hash(data)
        if result != result2:
            errors += 1
            print(f"  [iter {i}] Non-deterministic hash!", file=sys.stderr)
            continue

        # Verify hash length (should be 8 bytes for u64)
        if len(result) != 8:
            errors += 1
            print(f"  [iter {i}] Expected 8 bytes, got {len(result)}", file=sys.stderr)

    return errors


def test_transform_data(iterations, seed):
    """Stress test transform_data for dangling pointer issues."""
    import rustops
    random.seed(seed + 2)
    errors = 0

    for i in range(iterations):
        # Vary data size
        size = random.randint(1, 100)
        data = [random.uniform(-1000.0, 1000.0) for _ in range(size)]
        result = rustops.transform_data(data)

        # Verify result is a list
        if not isinstance(result, list):
            errors += 1
            print(f"  [iter {i}] Expected list, got {type(result)}", file=sys.stderr)
            continue

        # Verify result length
        if len(result) != len(data):
            errors += 1
            print(f"  [iter {i}] Length mismatch: {len(result)} vs {len(data)}", file=sys.stderr)
            continue

        # Verify transformation: result[j] should be clamp(data[j]*2.5 + j*0.1, -1e6, 1e6)
        for j, (orig, res) in enumerate(zip(data, result)):
            expected = orig * 2.5 + j * 0.1
            expected = max(-1e6, min(1e6, expected))
            if abs(res - expected) > 1e-6:
                errors += 1
                print(f"  [iter {i}, idx {j}] Expected {expected}, got {res}", file=sys.stderr)
                break

    return errors


def test_pipeline(iterations, seed):
    """Stress test Pipeline struct methods."""
    import rustops
    random.seed(seed + 3)
    errors = 0

    for i in range(iterations):
        ops = random.sample(["scale", "offset", "clamp", "abs"], k=random.randint(1, 4))
        scale = random.uniform(0.5, 5.0)
        pipe = rustops.Pipeline(scale_factor=scale)
        for op in ops:
            pipe.add_operation(op)

        # Verify describe works
        desc = pipe.describe()
        if not isinstance(desc, str):
            errors += 1
            print(f"  [iter {i}] describe() returned {type(desc)}", file=sys.stderr)
            continue

        if pipe.operation_count() != len(ops):
            errors += 1
            print(f"  [iter {i}] operation_count mismatch", file=sys.stderr)
            continue

        # Execute pipeline
        size = random.randint(1, 50)
        data = [random.uniform(-100.0, 100.0) for _ in range(size)]
        result = pipe.execute(data)

        if not isinstance(result, list):
            errors += 1
            print(f"  [iter {i}] execute() returned {type(result)}", file=sys.stderr)
            continue

        if len(result) != len(data):
            errors += 1
            print(f"  [iter {i}] execute() length mismatch", file=sys.stderr)
            continue

        # Verify all results are finite
        for j, v in enumerate(result):
            if not isinstance(v, float):
                errors += 1
                print(f"  [iter {i}, idx {j}] Expected float, got {type(v)}", file=sys.stderr)
                break
            if v != v:  # NaN check
                errors += 1
                print(f"  [iter {i}, idx {j}] Got NaN", file=sys.stderr)
                break

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Stress test for rustops Python extension"
    )
    parser.add_argument(
        "--iterations", type=int, default=10000,
        help="Number of iterations per test function (default: 10000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print timing information"
    )
    args = parser.parse_args()

    print(f"rustops stress test: {args.iterations} iterations, seed={args.seed}")
    print("=" * 60)

    total_errors = 0
    total_time = 0.0

    # Test process_text
    t0 = time.time()
    errs = test_process_text(args.iterations, args.seed)
    t1 = time.time()
    total_errors += errs
    total_time += t1 - t0
    status = "PASS" if errs == 0 else f"FAIL ({errs} errors)"
    print(f"  process_text:   {status}  ({t1 - t0:.3f}s)")

    # Test compute_hash
    t0 = time.time()
    errs = test_compute_hash(args.iterations, args.seed)
    t1 = time.time()
    total_errors += errs
    total_time += t1 - t0
    status = "PASS" if errs == 0 else f"FAIL ({errs} errors)"
    print(f"  compute_hash:   {status}  ({t1 - t0:.3f}s)")

    # Test transform_data
    t0 = time.time()
    errs = test_transform_data(args.iterations, args.seed)
    t1 = time.time()
    total_errors += errs
    total_time += t1 - t0
    status = "PASS" if errs == 0 else f"FAIL ({errs} errors)"
    print(f"  transform_data: {status}  ({t1 - t0:.3f}s)")

    # Test Pipeline
    t0 = time.time()
    errs = test_pipeline(args.iterations, args.seed)
    t1 = time.time()
    total_errors += errs
    total_time += t1 - t0
    status = "PASS" if errs == 0 else f"FAIL ({errs} errors)"
    print(f"  pipeline:       {status}  ({t1 - t0:.3f}s)")

    print("=" * 60)
    if args.profile:
        print(f"total_time {total_time:.4f}")

    if total_errors == 0:
        print(f"ALL PASS ({args.iterations} iterations x 4 tests)")
        sys.exit(0)
    else:
        print(f"FAILED: {total_errors} total errors")
        sys.exit(1)


if __name__ == "__main__":
    main()

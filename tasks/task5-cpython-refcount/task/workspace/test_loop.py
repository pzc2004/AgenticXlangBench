#!/usr/bin/env python3
"""Test script for pyvector C extension - exercises push/pop/get in a loop."""
import argparse
import time
import sys


def run_c_extension(iterations, seed):
    """Run test using C extension."""
    import pyvector
    import random
    random.seed(seed)

    vec = pyvector.Vector()
    t0 = time.time()
    for i in range(iterations):
        val = f"item_{i}_{random.randint(0, 1000)}"
        vec.push(val)
        if vec.size() > 10:
            vec.pop()
        if vec.size() > 0:
            _ = vec.get(random.randint(0, vec.size() - 1))
    t1 = time.time()
    return t1 - t0


def run_pure_python(iterations, seed):
    """Run test using pure Python list."""
    import random
    random.seed(seed)

    vec = []
    t0 = time.time()
    for i in range(iterations):
        val = f"item_{i}_{random.randint(0, 1000)}"
        vec.append(val)
        if len(vec) > 10:
            _ = vec.pop()
        if len(vec) > 0:
            _ = vec[random.randint(0, len(vec) - 1)]
    t1 = time.time()
    return t1 - t0


def main():
    parser = argparse.ArgumentParser(
        description="Test pyvector C extension performance and correctness"
    )
    parser.add_argument(
        "--iterations", type=int, default=5000,
        help="Number of iterations to run (default: 5000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print timing information"
    )
    parser.add_argument(
        "--pure_python", action="store_true",
        help="Use pure Python list instead of C extension"
    )
    args = parser.parse_args()

    try:
        if args.pure_python:
            elapsed = run_pure_python(args.iterations, args.seed)
        else:
            elapsed = run_c_extension(args.iterations, args.seed)

        if args.profile:
            print(f"total_time {elapsed:.4f}")

        if args.pure_python:
            print("OK")
        else:
            print(f"OK iterations={args.iterations}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

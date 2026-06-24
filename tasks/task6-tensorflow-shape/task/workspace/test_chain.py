#!/usr/bin/env python3
"""
test_chain.py -- Chain tensor operations and verify shape inference consistency.

The script runs a sequence of conv2d / relu operations and checks that
the C-level shape-inference metadata matches the actual NumPy output shape.

CHECK STRATEGY:
  - At every op: LENIENT check (rank only).  Catches gross errors.
  - At op #50:   STRICT check (exact shape).  Catches subtle off-by-one
                  errors that have accumulated through the metadata chain.

Usage:
    python test_chain.py --num_ops 10     # short chain  -- passes (no strict check)
    python test_chain.py --num_ops 60     # long  chain  -- strict check at #50
    python test_chain.py --num_ops 60 -v  # verbose: print per-op metadata
"""
import argparse
import sys
import numpy as np

import tensor_ops

# The op number at which a strict (exact-shape) check is performed.
STRICT_CHECK_OP = 50


def generate_chain(num_ops):
    """
    Generate a sequence of op names.
    Pattern: conv2d, relu, conv2d, relu, ...   (no pooling)
    """
    ops = []
    for i in range(num_ops):
        ops.append("conv2d" if i % 2 == 0 else "relu")
    return ops


def make_weight(c_out, c_in, kh, kw, rng):
    """Create a random weight tensor."""
    return rng.standard_normal((c_out, c_in, kh, kw)).astype(np.float32) * 0.1


def run_chain(num_ops, seed=42, verbose=False):
    """
    Run the chain and check shapes.

    Returns (passed: bool, message: str).
    """
    rng = np.random.RandomState(seed)
    ops = generate_chain(num_ops)

    # Initial tensor: [batch=1, channels=3, H=32, W=32]
    actual = rng.standard_normal((1, 3, 32, 32)).astype(np.float32) * 0.5
    meta_shape = (1, 3, 32, 32)

    weights = {}
    rank_mismatches = []          # collected from lenient checks
    strict_mismatch = None        # result of the strict check at op #50

    for i, op_name in enumerate(ops):
        op_num = i + 1            # 1-indexed

        if op_name == "conv2d":
            c_in_actual = actual.shape[1]
            c_in_meta = meta_shape[1]
            c_out = 16

            key = (c_out, c_in_actual)
            if key not in weights:
                weights[key] = make_weight(c_out, c_in_actual, 3, 3, rng)
            w = weights[key]

            # Forward pass (uses real tensor -- always correct shape)
            actual = tensor_ops.conv2d_forward(actual, w, 1, 1)

            # Shape inference (uses metadata -- may diverge over time)
            w_shape = (c_out, c_in_meta, 3, 3)
            meta_shape = tensor_ops.conv2d_shape(meta_shape, w_shape, 1, 1)

        elif op_name == "relu":
            actual = tensor_ops.relu_forward(actual)
            meta_shape = tensor_ops.relu_shape(meta_shape)
        else:
            raise ValueError(f"Unknown op: {op_name}")

        actual_shape = tuple(actual.shape)

        # ---- LENIENT check: rank must match ----
        if len(meta_shape) != len(actual_shape):
            rank_mismatches.append((op_num, meta_shape, actual_shape))
            if verbose:
                print(f"  [op #{op_num}] RANK MISMATCH  "
                      f"meta_rank={len(meta_shape)}  actual_rank={len(actual_shape)}")

        # ---- STRICT check at op #50 ----
        if op_num == STRICT_CHECK_OP and strict_mismatch is None:
            if meta_shape != actual_shape:
                strict_mismatch = (op_num, meta_shape, actual_shape)
                if verbose:
                    print(f"  [op #{op_num}] STRICT MISMATCH  "
                          f"meta={meta_shape}  actual={actual_shape}")

        if verbose and op_num % 10 == 0:
            print(f"  [op #{op_num}] actual={actual_shape}  meta={meta_shape}")

    # ---- report ----
    # Priority: rank mismatches > strict mismatch > pass
    if rank_mismatches:
        op_n, m, a = rank_mismatches[0]
        msg = (f"Rank mismatch at op #{op_n}: "
               f"meta_rank={len(m)}  actual_rank={len(a)}")
        return False, msg

    if strict_mismatch:
        op_n, m, a = strict_mismatch
        msg = (f"Shape mismatch detected at op #{op_n}.\n"
               f"  Metadata shape: {m}\n"
               f"  Actual shape:   {a}\n"
               f"  NOTE: The mismatch is reported at op #{op_n}, but\n"
               f"  the root cause may be in an earlier op's shape inference.")
        return False, msg

    return True, f"All {num_ops} ops passed shape check."


def main():
    parser = argparse.ArgumentParser(
        description="Chain tensor ops and verify shape inference consistency."
    )
    parser.add_argument("--num_ops", type=int, default=60,
                        help="Number of ops to chain (default: 60)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-op details")
    args = parser.parse_args()

    print(f"Running chain of {args.num_ops} ops (seed={args.seed}) ...")
    passed, msg = run_chain(args.num_ops, seed=args.seed, verbose=args.verbose)

    if passed:
        print(f"OK: {msg}")
        sys.exit(0)
    else:
        print(f"FAIL: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

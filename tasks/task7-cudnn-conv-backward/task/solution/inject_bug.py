#!/usr/bin/env python3
"""
Inject 3 real bugs + 20 decoys into conv_kernel.cu

Bug 1: backward_data boundary error
    When input_h % stride != 0, the output index computation has an off-by-one.
    Changes (ih - kh + padding + stride - 1) to (ih - kh + padding + stride).
    Effect: skips the correct output position for non-aligned sizes.

Bug 2: backward_weight accumulation error
    When kernel_size % 2 == 0, the weight gradient is doubled.
    Adds "acc *= 2.0f;" after the accumulation loop for even kH.
    Effect: weight gradients are 2x too large for even kernel sizes.

Bug 3: backward_bias reduction error
    When H_out * W_out is not aligned to blockDim.x, adds a spurious term.
    Adds "((HW % blockDim.x != 0) ? 1.0f : 0.0f)" to the shared memory store.
    Effect: bias gradients have a small constant error for non-aligned sizes.

Decoys: 20 suspicious comments scattered across the CUDA code.
"""

import os
import sys
import re


def inject_real_bugs(filepath):
    """Inject 3 real bugs into conv_kernel.cu."""
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # === Bug 1: backward_data boundary error ===
    # Find the BUG_LOCATION_1 marker and modify the boundary computation
    # Clean:  int oh_num = ih - kh + padding + stride - 1;
    # Buggy:  int oh_num = ih - kh + padding + stride;

    bug1_clean = "int oh_num = ih - kh + padding + stride - 1;"
    bug1_buggy = "int oh_num = ih - kh + padding + stride;"

    if bug1_clean in content:
        content = content.replace(bug1_clean, bug1_buggy, 1)
        print("  [OK] Bug 1: backward_data boundary (+1 off-by-one)")
    else:
        print("  [FAIL] Bug 1: target not found for backward_data boundary")
        return False

    # === Bug 2: backward_weight accumulation error ===
    # Find the BUG_LOCATION_2 marker and add doubling for even kernel sizes
    # Clean:   atomicAdd(&grad_weight[idx], acc);
    # Buggy:   if (kH % 2 == 0) acc *= 2.0f;  atomicAdd(...);

    bug2_clean = "    /* BUG_LOCATION_2: backward_weight accumulation for even kernel sizes\n     * Clean:   acc (no modification)\n     * Buggy:   acc *= 2.0f when (kH % 2 == 0)\n     * Trigger: kernel_size % 2 == 0\n     */\n    atomicAdd(&grad_weight[idx], acc);"
    bug2_buggy = "    /* BUG_LOCATION_2: backward_weight accumulation for even kernel sizes\n     * Clean:   acc (no modification)\n     * Buggy:   acc *= 2.0f when (kH % 2 == 0)\n     * Trigger: kernel_size % 2 == 0\n     */\n    if (kH % 2 == 0) acc *= 2.0f;\n    atomicAdd(&grad_weight[idx], acc);"

    if bug2_clean in content:
        content = content.replace(bug2_clean, bug2_buggy, 1)
        print("  [OK] Bug 2: backward_weight accumulation (2x for even kH)")
    else:
        print("  [FAIL] Bug 2: target not found for backward_weight")
        return False

    # === Bug 3: backward_bias reduction error ===
    # Find the BUG_LOCATION_3 marker and add spurious term
    # Clean:   sdata[threadIdx.x] = local_sum;
    # Buggy:   sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f);

    bug3_clean = "    /* BUG_LOCATION_3: backward_bias reduction for non-aligned sizes\n     * Clean:   sdata[threadIdx.x] = local_sum (no extra term)\n     * Buggy:   sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f)\n     * Trigger: non-aligned output spatial sizes\n     */\n    sdata[threadIdx.x] = local_sum;"
    bug3_buggy = "    /* BUG_LOCATION_3: backward_bias reduction for non-aligned sizes\n     * Clean:   sdata[threadIdx.x] = local_sum (no extra term)\n     * Buggy:   sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f)\n     * Trigger: non-aligned output spatial sizes\n     */\n    sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f);"

    if bug3_clean in content:
        content = content.replace(bug3_clean, bug3_buggy, 1)
        print("  [OK] Bug 3: backward_bias reduction (spurious +1 for non-aligned)")
    else:
        print("  [FAIL] Bug 3: target not found for backward_bias")
        return False

    with open(filepath, 'w') as f:
        f.write(content)

    # Verify all 3 bugs injected
    with open(filepath, 'r') as f:
        check = f.read()

    ok = True
    if "padding + stride;" not in check or "padding + stride - 1;" in check.split("backward_data")[1].split("atomicAdd")[0] if "backward_data" in check else True:
        pass  # Bug 1 check is complex, rely on replace success
    if "kH % 2 == 0" not in check:
        print("  [WARN] Bug 2 verification: pattern not found in output")
        ok = False
    if "HW % blockDim.x != 0" not in check:
        print("  [WARN] Bug 3 verification: pattern not found in output")
        ok = False

    return ok


def inject_decoys(filepath):
    """Inject 20 decoy comments into conv_kernel.cu."""
    decoys = [
        "// WARNING: numerical precision issue in reduction - may cause gradient drift\n",
        "// FIXME: potential race condition in atomicAdd for large batch sizes\n",
        "// TODO: verify boundary condition for padding > kernel_size/2\n",
        "// NOTE: this accumulation order affects floating point precision\n",
        "// BUG? stride offset calculation might overflow for large inputs\n",
        "// FIXME: shared memory bank conflict in reduction kernel\n",
        "// WARNING: this index computation assumes contiguous memory layout\n",
        "// TODO: add bounds check for edge case when blockDim > HW\n",
        "// NOTE: gradient sign convention matches PyTorch but not TensorFlow\n",
        "// BUG? weight gradient might be wrong for depthwise separable conv\n",
        "// FIXME: output index wraps around for very large spatial dimensions\n",
        "// WARNING: atomicAdd has limited precision for float32\n",
        "// TODO: the loop unrolling here might cause register spilling\n",
        "// NOTE: the -1 in boundary computation is critical for correctness\n",
        "// BUG? bias gradient is incorrect when batch_size > 65535\n",
        "// FIXME: the reduction assumes power-of-2 block sizes\n",
        "// WARNING: padding asymmetry could cause gradient mismatch\n",
        "// TODO: verify that oh >= 0 check handles negative padding correctly\n",
        "// NOTE: the accumulation factor depends on kernel symmetry\n",
        "// BUG? gradient magnitude scales with input_h, might cause overflow\n",
    ]

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Insert decoys at various positions
    import random
    random.seed(42)

    # Find good insertion points (after blank lines, before function defs, etc.)
    insert_positions = []
    for i, line in enumerate(lines):
        if line.strip() == '' and i > 0 and i < len(lines) - 1:
            insert_positions.append(i)

    # Pick 20 positions spread across the file
    if len(insert_positions) < 20:
        # If not enough blank lines, insert between any lines
        insert_positions = list(range(10, len(lines), max(1, len(lines) // 25)))

    random.shuffle(insert_positions)
    positions = sorted(insert_positions[:20])

    # Insert in reverse order to maintain position validity
    for pos, decoy in zip(reversed(positions), reversed(decoys)):
        lines.insert(pos, decoy)

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return len(decoys)


def main():
    # Determine the workspace path
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    else:
        workspace = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

    target = os.path.join(workspace, "conv_kernel.cu")

    if not os.path.exists(target):
        print(f"ERROR: {target} not found")
        sys.exit(1)

    print("=" * 60)
    print("Injecting bugs into conv_kernel.cu")
    print("=" * 60)

    print("\n>>> Real bugs (3):")
    if not inject_real_bugs(target):
        print("\nERROR: Bug injection failed!")
        sys.exit(1)

    print(f"\n>>> Decoys (20):")
    decoy_count = inject_decoys(target)
    print(f"  Inserted {decoy_count} decoy comments")

    print(f"\nTotal: 3 real bugs + {decoy_count} decoys = {3 + decoy_count} modifications")
    print("Done.")


if __name__ == "__main__":
    main()

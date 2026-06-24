#!/usr/bin/env python3
"""
inject_bug.py -- Inject 3 real shape-inference bugs + 20 decoys into tensor_ops.c.

Bug 1 (conv2d_shape):  Returns W_out - 1  instead of W_out.
    Effect: shape metadata is one pixel narrower than the actual tensor.
    Symptom: after many chained ops the accumulated width error causes a
    shape-mismatch check failure at ~op #50.  Root cause is the first conv2d.

Bug 2 (conv2d_forward): Wrong input-index offset.
    Effect: reads from wrong memory location for the channel dimension,
    corrupting output data (but output *shape* stays correct).
    Symptom: downstream numerical values are wrong; a strict value check
    at op #50 reveals the corruption.

Bug 3 (pool_shape): Returns H_out - 1  instead of H_out.
    Effect: pool shape inference is off by one in height.
    Symptom: compounds with Bug 1 when pool and conv2d are interleaved.

Decoys: 20 misleading comment insertions scattered through the C source.
"""
import os
import sys

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")


def inject_real_bug():
    """Inject the 3 real bugs into tensor_ops.c."""
    success = True
    filepath = os.path.join(WORKSPACE_DIR, "tensor_ops.c")
    if not os.path.exists(filepath):
        print(f"  [ERR] File not found: {filepath}")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # ==============================================================
    # Bug 1: conv2d_shape  --  W_out  ->  (W_out - 1)
    # ==============================================================
    bug1_clean = (
        '    /* BUG_LOCATION_1: W_out is returned correctly here.\n'
        '     * inject_bug.py will replace "W_out" with "(W_out - 1)" in the line below.\n'
        '     */\n'
        '    return Py_BuildValue("(iiii)", N, C_out, H_out, W_out);'
    )
    bug1_buggy = (
        '    /* BUG_LOCATION_1: W_out has been decremented -- INJECTED BUG.\n'
        '     * Root cause: shape inference returns wrong width dimension.\n'
        '     */\n'
        '    return Py_BuildValue("(iiii)", N, C_out, H_out, (W_out - 1));'
    )

    if bug1_clean in content:
        content = content.replace(bug1_clean, bug1_buggy, 1)
        print("  [OK] Bug 1: conv2d_shape returns W_out - 1")
    elif bug1_buggy in content:
        print("  [SKIP] Bug 1: already injected")
    else:
        print("  [ERR] Bug 1: target code not found")
        success = False

    # ==============================================================
    # Bug 2: conv2d_forward  --  ci * H * W  ->  ci * H
    # ==============================================================
    bug2_clean = (
        '                                    /* BUG_LOCATION_2: input offset.\n'
        '                                     * Clean:  n*C_in*H*W + ci*H*W + ih*W + iw\n'
        '                                     * Buggy:  n*C_in*H*W + ci*H   + ih*W + iw\n'
        '                                     * (inject_bug.py removes "* W" after ci*H)\n'
        '                                     */\n'
        '                                    float iv = in_data[n * C_in * H * W\n'
        '                                                       + ci * H * W\n'
        '                                                       + ih * W + iw];'
    )
    bug2_buggy = (
        '                                    /* BUG_LOCATION_2: input offset -- INJECTED BUG.\n'
        '                                     * ci*H*W was changed to ci*H (missing * W).\n'
        '                                     * This reads from wrong memory, corrupting data.\n'
        '                                     */\n'
        '                                    float iv = in_data[n * C_in * H * W\n'
        '                                                       + ci * H\n'
        '                                                       + ih * W + iw];'
    )

    if bug2_clean in content:
        content = content.replace(bug2_clean, bug2_buggy, 1)
        print("  [OK] Bug 2: conv2d_forward uses ci*H instead of ci*H*W")
    elif bug2_buggy in content:
        print("  [SKIP] Bug 2: already injected")
    else:
        print("  [ERR] Bug 2: target code not found")
        success = False

    # ==============================================================
    # Bug 3: pool_shape  --  H_out  ->  (H_out - 1)
    # ==============================================================
    bug3_clean = (
        '    /* BUG_LOCATION_3: H_out is returned correctly here.\n'
        '     * inject_bug.py will replace "H_out" with "(H_out - 1)" in the line below.\n'
        '     */\n'
        '    return Py_BuildValue("(iiii)", N, C, H_out, W_out);'
    )
    bug3_buggy = (
        '    /* BUG_LOCATION_3: H_out has been decremented -- INJECTED BUG.\n'
        '     * Root cause: pool shape inference returns wrong height dimension.\n'
        '     */\n'
        '    return Py_BuildValue("(iiii)", N, C, (H_out - 1), W_out);'
    )

    if bug3_clean in content:
        content = content.replace(bug3_clean, bug3_buggy, 1)
        print("  [OK] Bug 3: pool_shape returns H_out - 1")
    elif bug3_buggy in content:
        print("  [SKIP] Bug 3: already injected")
    else:
        print("  [ERR] Bug 3: target code not found")
        success = False

    # Write back
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"\n  File updated: {filepath}")
    else:
        print("\n  [WARN] No changes written")

    return success


def inject_decoys():
    """Inject 20 misleading comment decoys into tensor_ops.c."""
    filepath = os.path.join(WORKSPACE_DIR, "tensor_ops.c")
    if not os.path.exists(filepath):
        return 0

    with open(filepath, "r") as f:
        content = f.read()
    original = content
    count = 0

    decoys = [
        # --- header area ---
        (
            "#include <string.h>",
            "#include <string.h>\n/* FIXME: consider using memmove for overlapping regions */",
            "decoy: string.h FIXME"
        ),
        (
            "#include <math.h>",
            "#include <math.h>\n/* TODO: replace with hand-rolled fast approximations */",
            "decoy: math.h TODO"
        ),
        # --- conv2d_shape ---
        (
            "    int N      = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 0));",
            "    int N      = (int)PyLong_AsLong(PyTuple_GetItem(input_shape, 0));  /* BUG_CANDIDATE: no error check on PyLong_AsLong */",
            "decoy: PyLong_AsLong check"
        ),
        (
            "    int H_out = (H - KH + 2 * padding) / stride + 1;",
            "    int H_out = (H - KH + 2 * padding) / stride + 1;  /* BUG_CANDIDATE: integer division truncation */",
            "decoy: H_out truncation"
        ),
        (
            "    int W_out = (W - KW + 2 * padding) / stride + 1;",
            "    int W_out = (W - KW + 2 * padding) / stride + 1;  /* BUG_CANDIDATE: negative values possible */",
            "decoy: W_out negative"
        ),
        # --- conv2d_forward ---
        (
            '    if (!PyArg_ParseTuple(args, "O!O!ii"',
            '    if (!PyArg_ParseTuple(args, "O!O!ii"  /* BUG_CANDIDATE: format string may mismatch */',
            "decoy: ParseTuple format"
        ),
        (
            "    memset(out_data, 0, (size_t)N * C_out * H_out * W_out * sizeof(float));",
            "    memset(out_data, 0, (size_t)N * C_out * H_out * W_out * sizeof(float));  /* BUG_CANDIDATE: potential overflow in size calc */",
            "decoy: memset overflow"
        ),
        (
            "                    float sum = 0.0f;",
            "                    float sum = 0.0f;  /* BUG_CANDIDATE: use double for accumulation? */",
            "decoy: float precision"
        ),
        (
            "                                    sum += iv * wv;",
            "                                    sum += iv * wv;  /* BUG_CANDIDATE: FMA ordering may differ */",
            "decoy: FMA ordering"
        ),
        (
            "                    out_data[n * C_out * H_out * W_out",
            "                    out_data[n * C_out * H_out * W_out  /* BUG_CANDIDATE: row-major vs col-major */",
            "decoy: row-major"
        ),
        # --- relu ---
        (
            "    Py_INCREF(input_shape);",
            "    Py_INCREF(input_shape);  /* BUG_CANDIDATE: is this the right refcount protocol? */",
            "decoy: relu refcount"
        ),
        (
            "        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;",
            "        out_data[i] = in_data[i] > 0.0f ? in_data[i] : 0.0f;  /* TODO: vectorize with SIMD */",
            "decoy: SIMD TODO"
        ),
        # --- pool_shape ---
        (
            "    int H_out = (H - kernel_size) / stride + 1;",
            "    int H_out = (H - kernel_size) / stride + 1;  /* BUG_CANDIDATE: off-by-one when H == kernel_size */",
            "decoy: pool H_out boundary"
        ),
        (
            "    int W_out = (W - kernel_size) / stride + 1;",
            "    int W_out = (W - kernel_size) / stride + 1;  /* BUG_CANDIDATE: same issue for width */",
            "decoy: pool W_out boundary"
        ),
        # --- pool_forward ---
        (
            "    if (H_out <= 0 || W_out <= 0) {",
            "    if (H_out <= 0 || W_out <= 0) {  /* BUG_CANDIDATE: should also check N and C */",
            "decoy: pool bounds check"
        ),
        (
            "    float inv_area = 1.0f / (float)(kernel_size * kernel_size);",
            "    float inv_area = 1.0f / (float)(kernel_size * kernel_size);  /* BUG_CANDIDATE: div by zero if kernel_size == 0 */",
            "decoy: div by zero"
        ),
        (
            "                    sum += in_data[n * C * H * W",
            "                    sum += in_data[n * C * H * W  /* BUG_CANDIDATE: cache-unfriendly access pattern */",
            "decoy: cache unfriendly"
        ),
        # --- module init ---
        (
            "    import_array();",
            "    import_array();  /* BUG_CANDIDATE: import_array returns void in NumPy >= 1.20 */",
            "decoy: import_array return"
        ),
        (
            "    return PyModule_Create(&tensor_ops_module);",
            "    return PyModule_Create(&tensor_ops_module);  /* TODO: add module-level constants */",
            "decoy: module constants TODO"
        ),
        # --- one more in conv2d ---
        (
            "            w_data[co * C_in * KH * KW",
            "            w_data[co * C_in * KH * KW  /* BUG_CANDIDATE: weight layout assumption */",
            "decoy: weight layout"
        ),
    ]

    for old, new, desc in decoys:
        if old in content and new not in content:
            content = content.replace(old, new, 1)
            count += 1
            print(f"  [OK] Decoy {count}: {desc}")
        elif new in content:
            count += 1
            print(f"  [SKIP] Decoy {count}: already present ({desc})")

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)

    return count


def main():
    print("=" * 60)
    print("Tensor-ops shape-inference bug injection")
    print("=" * 60)

    print("\n>>> Injecting 3 real bugs:")
    if not inject_real_bug():
        print("\n  Bug injection FAILED!")
        sys.exit(1)

    print(f"\n>>> Injecting decoys:")
    decoy_count = inject_decoys()
    print(f"\nTotal: 3 real bugs + {decoy_count} decoys = {3 + decoy_count} modifications")

    if decoy_count < 15:
        print("[WARN] Fewer than 15 decoys injected -- check code.")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()

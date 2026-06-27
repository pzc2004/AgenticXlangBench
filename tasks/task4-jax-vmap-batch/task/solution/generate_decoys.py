#!/usr/bin/env python3
"""
Generate decoys.patch for Task 4.

Decoys are harmless-looking but actually important code modifications.
Goal: distract Kimi; if Kimi "fixes" them, it introduces new bugs.
"""
import os, sys, re, shutil, subprocess
from pathlib import Path

# (relative_path, old, new) — old must appear exactly once in clean source
DECOYS = [
    # ===================== batching.py =====================
    (
        "jax/_src/interpreters/batching.py",
        "def matchaxis(axis_data, src, dst, x, sum_match=False):",
        "def matchaxis(axis_data, src, dst, x, sum_match=False):\n"
        "  # WARNING: this guard prevents invalid moveaxis when src == dst\n"
        "  if src is not None and dst is not None and src == dst:\n"
        "    return x\n"
    ),
    (
        "jax/_src/interpreters/batching.py",
        "def vectorized_batcher(prim, axis_data, batched_args, batch_dims, **params):",
        "def vectorized_batcher(prim, axis_data, batched_args, batch_dims, **params):\n"
        "  # FIXME: validate batch_dims types before binding\n"
        "  assert all(isinstance(d, (int, type(None))) for d in batch_dims), batch_dims\n"
    ),
    (
        "jax/_src/interpreters/batching.py",
        "def reducer_batcher(prim, axis_data, batched_args, batch_dims, axes,\n"
        "                    **params):",
        "def reducer_batcher(prim, axis_data, batched_args, batch_dims, axes,\n"
        "                    **params):\n"
        "  # NOTE: axes must be deterministic; do not remove sorting\n"
        "  axes = tuple(sorted(axes)) if axes else axes\n"
    ),
    (
        "jax/_src/interpreters/batching.py",
        "def batch_subtrace_2(f, tag, axis_data, in_dims, in_vals):",
        "def batch_subtrace_2(f, tag, axis_data, in_dims, in_vals):\n"
        "  # BUG_CANDIDATE: axis_size could be zero?\n"
        "  if isinstance(axis_data.size, int) and axis_data.size <= 0:\n"
        "    raise ValueError(f\"axis_size must be positive, got {axis_data.size}\")\n"
    ),
    (
        "jax/_src/interpreters/batching.py",
        "class BatchTrace(Trace):",
        "_VALID_BDIM_TYPES = (int, type(None))\n\n"
        "class BatchTrace(Trace):\n"
        "  # TODO: verify bdims are valid integers\n"
        "  def _check_bdims(self, dims):\n"
        "    return tuple(d if isinstance(d, self._VALID_BDIM_TYPES) else int(d) for d in dims)\n"
    ),

    # ===================== ad.py =====================
    (
        "jax/_src/interpreters/ad.py",
        "def linearize_subtrace_2(f: Callable, is_vjp: bool,\n"
        "                         tag: core.TraceTag, nzs_in: Sequence[bool],\n"
        "                         debug_info: core.DebugInfo, primals):",
        "def linearize_subtrace_2(f: Callable, is_vjp: bool,\n"
        "                         tag: core.TraceTag, nzs_in: Sequence[bool],\n"
        "                         debug_info: core.DebugInfo, primals):\n"
        "  # WARNING: is_vjp flag must be bool, not numpy scalar\n"
        "  is_vjp = bool(is_vjp)\n"
    ),
    (
        "jax/_src/interpreters/ad.py",
        "def linearize_subtrace(_f: Callable, _store: lu.Store, _is_vjp: bool,\n"
        "                       _tag: core.TraceTag, nzs_in: Sequence[bool],\n"
        "                       debug_info: core.DebugInfo, *primals, **params):",
        "def linearize_subtrace(_f: Callable, _store: lu.Store, _is_vjp: bool,\n"
        "                       _tag: core.TraceTag, nzs_in: Sequence[bool],\n"
        "                       debug_info: core.DebugInfo, *primals, **params):\n"
        "  # NOTE: store must be writeable; copy to avoid mutation side effects\n"
        "  _store = lu.Store(_store.val)\n"
    ),

    # ===================== lax.py =====================
    (
        "jax/_src/lax/lax.py",
        "def _reshape_batch_rule(axis_data, batched_args, batch_dims, *, new_sizes,\n"
        "                        dimensions, sharding):",
        "def _reshape_batch_rule(axis_data, batched_args, batch_dims, *, new_sizes,\n"
        "                        dimensions, sharding):\n"
        "  # FIXME: dimensions may contain numpy scalars; cast to int\n"
        "  dimensions = tuple(int(d) for d in dimensions) if dimensions is not None else dimensions\n"
    ),
    (
        "jax/_src/lax/lax.py",
        "def _transpose_batch_rule(batched_args, batch_dims, *, permutation):",
        "def _transpose_batch_rule(batched_args, batch_dims, *, permutation):\n"
        "  # WARNING: permutation must be plain ints, not np.int64\n"
        "  permutation = tuple(int(p) for p in permutation)\n"
    ),
    (
        "jax/_src/lax/lax.py",
        "def _concatenate_batch_rule(batched_args, batch_dims, *, dimension):",
        "def _concatenate_batch_rule(batched_args, batch_dims, *, dimension):\n"
        "  # NOTE: empty operand list would break dimension computation\n"
        "  if not batched_args:\n"
        "    raise ValueError(\"concatenate requires at least one operand\")\n"
    ),

    # ===================== slicing.py =====================
    (
        "jax/_src/lax/slicing.py",
        "def _gather_batching_rule(batched_args: Sequence[Array], batch_dims: Sequence[int | None], *,\n"
        "                          dimension_numbers, slice_sizes, unique_indices, indices_are_sorted,\n"
        "                          mode, fill_value):",
        "def _gather_batching_rule(batched_args: Sequence[Array], batch_dims: Sequence[int | None], *,\n"
        "                          dimension_numbers, slice_sizes, unique_indices, indices_are_sorted,\n"
        "                          mode, fill_value):\n"
        "  # WARNING: operand_bdim must be int or None\n"
        "  operand_bdim = batch_dims[0]\n"
        "  operand_bdim = int(operand_bdim) if operand_bdim is not None else None\n"
    ),
    (
        "jax/_src/lax/slicing.py",
        "def _slice_batching_rule(batched_args, batch_dims, *, start_indices,\n"
        "                         limit_indices, strides):",
        "def _slice_batching_rule(batched_args, batch_dims, *, start_indices,\n"
        "                         limit_indices, strides):\n"
        "  # NOTE: strides None means step 1; explicit tuple avoids mutation bugs\n"
        "  strides = tuple(strides) if strides is not None else strides\n"
    ),
]


def apply_decoy(clean_dir, work_dir, rel_path, old, new, name):
    src = os.path.join(work_dir, rel_path)
    with open(src) as f:
        content = f.read()
    if old not in content:
        raise RuntimeError(f"{name}: pattern not found in {rel_path}")
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{name}: pattern appears {count} times in {rel_path}")
    content = content.replace(old, new, 1)
    with open(src, "w") as f:
        f.write(content)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <output_patch_path>")
        sys.exit(1)

    clean_dir = sys.argv[1]
    output_patch = sys.argv[2]
    work_dir = os.path.join(os.path.dirname(output_patch), ".decoys_work")

    shutil.copytree(clean_dir, work_dir, dirs_exist_ok=True)

    for i, (rel_path, old, new) in enumerate(DECOYS):
        name = f"Decoy {i+1}"
        apply_decoy(clean_dir, work_dir, rel_path, old, new, name)
        print(f"  ✅ {name}")

    diff = subprocess.run(
        ["diff", "-ruN", clean_dir, work_dir],
        capture_output=True, text=True
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError("diff failed")

    # Rewrite paths to relative-to-/build/jax format
    lines = diff.stdout.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("diff -ruN "):
            # 丢弃带绝对路径的 diff 命令行，patch 不依赖它。
            continue
        if line.startswith("--- "):
            parts = line.split("\t")
            path = parts[0][4:]
            rel = path.replace(clean_dir, "").lstrip("/")
            new_lines.append(f"--- {rel}" + ("\t" + parts[1] if len(parts) > 1 else ""))
        elif line.startswith("+++ "):
            parts = line.split("\t")
            path = parts[0][4:]
            rel = path.replace(work_dir, "").lstrip("/")
            new_lines.append(f"+++ {rel}" + ("\t" + parts[1] if len(parts) > 1 else ""))
        else:
            new_lines.append(line)
    patch_text = "\n".join(new_lines) + "\n"

    with open(output_patch, "w") as f:
        f.write(patch_text)
    print(f"\n✅ Wrote {output_patch}")

    # Cleanup
    shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()

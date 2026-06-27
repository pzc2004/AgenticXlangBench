#!/usr/bin/env python3
"""
Task 4 per-bug patch generator.

Requires clean JAX source files. Produces one patch per bug in
per_bug_patches/bug_<name>.patch.
"""
import os, sys, re, shutil, subprocess
from pathlib import Path

# Bug definitions: (name, relative_path, old_text, new_text)
BUGS = [
    ("Bug 1", "jax/_src/interpreters/batching.py",
     "  if src == dst or dst is infer:\n    return x",
     "  if dst is infer:\n    return x\n  if src == dst:\n    return moveaxis(x, src, dst + 1)"),

    ("Bug 2", "jax/_src/interpreters/batching.py",
     "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, **params), None",
     "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, **params), 0"),

    ("Bug 3", "jax/_src/interpreters/batching.py",
     "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, axes=axes, **params), None",
     "  if False:\n    return prim.bind(*batched_args, axes=axes, **params), None"),

    ("Bug 4", "jax/_src/interpreters/batching.py",
     "    elif unmapped_args:",
     "    elif False:"),

    ("Bug 5", "jax/_src/interpreters/batching.py",
     "  elif src is None and type(dst) is int:\n"
     "    return broadcast(x, axis_data.size, canonicalize_axis(dst, np.ndim(x) + 1),\n"
     "                     axis_data.explicit_mesh_axis)",
     "  elif src is None and type(dst) is int:\n"
     "    return broadcast(x, axis_data.size + 1, canonicalize_axis(dst, np.ndim(x) + 1),\n"
     "                     axis_data.explicit_mesh_axis)"),

    ("Bug 7", "jax/_src/interpreters/ad.py",
     "  out_nz_tangents = [t for t, nz in zip(out_tangents, out_nzs) if nz]",
     "  out_nz_tangents = []"),

    ("Bug 8", "jax/_src/interpreters/batching.py",
     "  return prim.bind(*batched_args, **params), batch_dims[0]",
     "  return prim.bind(*batched_args, **params), batch_dims[0] + 1"),

    ("Bug 9", "jax/_src/interpreters/batching.py",
     "    return prim.bind(operand, axes=axes, **params), bdim_out",
     "    return prim.bind(operand, axes=axes, **params), bdim_out + 1"),

    ("Bug 10", "jax/_src/interpreters/batching.py",
     "    self.batch_dim = batch_dim",
     "    self.batch_dim = batch_dim if batch_dim is None else batch_dim + 1"),

    ("Bug 11", "jax/_src/interpreters/batching.py",
     "          return (BatchTracer(self, val_out, dim_out, src)",
     "          return (BatchTracer(self, val_out, dim_out + 1 if dim_out is not None else None, src)"),

    ("Bug 12", "jax/_src/interpreters/batching.py",
     "    return moveaxis(x, src, dst)",
     "    return moveaxis(x, dst, src)"),

    ("Bug 13", "jax/_src/interpreters/batching.py",
     "    axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))",
     "    axes = tuple(np.where(np.less(axes, bdim), axes, np.subtract(axes, 1)))"),

    ("Bug 15", "jax/_src/interpreters/batching.py",
     "  size, = {x.shape[bd] for x, bd in zip(args, dims) if bd is not None}\n"
     "  args = [bdim_at_front(x, bd, size) for x, bd in zip(args, dims)]\n"
     "  out = prim.bind(*args, **params)\n"
     "  return (out, (0,) * len(out)) if prim.multiple_results else (out, 0)",
     "  size, = {x.shape[bd] for x, bd in zip(args, dims) if bd is not None}\n"
     "  args = [bdim_at_front(x, bd, size) for x, bd in zip(args, dims)]\n"
     "  out = prim.bind(*args, **params)\n"
     "  return (out, (1,) * len(out)) if prim.multiple_results else (out, 1)"),

    ("Bug 17", "jax/_src/interpreters/batching.py",
     "    return broadcast(x, size, 0, mesh_axis=mesh_axis)",
     "    return broadcast(x, size, 1, mesh_axis=mesh_axis)"),

    ("Bug 18", "jax/_src/interpreters/ad.py",
     "  out_nzs = [type(t) is not Zero for t in out_tangents]",
     "  out_nzs = [type(t) is Zero for t in out_tangents]"),

    ("Bug 19", "jax/_src/interpreters/ad.py",
     "  out_zeros = map(op.not_, out_nzs)",
     "  out_zeros = out_nzs"),

    ("Bug 20", "jax/_src/lax/lax.py",
     "    dimensions = (0,) + tuple(np.add(1, dimensions))",
     "    dimensions = (0,) + tuple(np.add(2, dimensions))"),

    ("Bug 21", "jax/_src/lax/lax.py",
     "  perm = (bdim,) + tuple(i if i < bdim else i+1 for i in permutation)",
     "  perm = (bdim,) + tuple(i if i <= bdim else i+1 for i in permutation)"),

    ("Bug 22", "jax/_src/lax/lax.py",
     "  res_bdim = 0",
     "  res_bdim = bdim"),

    ("Bug 23", "jax/_src/lax/lax.py",
     "  return concatenate(operands, dimension + 1), 0",
     "  return concatenate(operands, dimension + 1), 1"),

    ("Bug 24", "jax/_src/lax/lax.py",
     "    if np.shape(which) == np.shape(cases[0]):\n"
     "      return select_n(which, *cases), which_bdim",
     "    if np.shape(which) == np.shape(cases[0]):\n"
     "      return select_n(which, *cases), which_bdim + 1"),

    ("Bug 25", "jax/_src/lax/lax.py",
     "    return reduce_p.bind(*(operands + init_values),\n                         computation=computation,\n                         dimensions=tuple(new_dimensions),\n                         jaxpr=jaxpr), new_operand_bdims",
     "    return reduce_p.bind(*(operands + init_values),\n                         computation=computation,\n                         dimensions=tuple(d + 1 for d in new_dimensions),\n                         jaxpr=jaxpr), new_operand_bdims"),

    ("Bug 26a", "jax/_src/lax/lax.py",
     "  batched_out = invoke_prim(\n"
     "      lhs,\n"
     "      rhs,\n"
     "      new_dimension_numbers,\n"
     "      precision=precision,\n"
     "      preferred_element_type=preferred_element_type,\n"
     "      out_sharding=out_sharding,\n"
     "  )\n"
     "  return batched_out, result_batch_dim\n",
     "  batched_out = invoke_prim(\n"
     "      lhs,\n"
     "      rhs,\n"
     "      new_dimension_numbers,\n"
     "      precision=precision,\n"
     "      preferred_element_type=preferred_element_type,\n"
     "      out_sharding=out_sharding,\n"
     "  )\n"
     "  return batched_out, result_batch_dim + 1\n"),

    ("Bug 26b", "jax/_src/lax/lax.py",
     "  out = reshape(operand, operand.shape[:1] + new_sizes, dimensions,\n"
     "                out_sharding=sharding)\n  return out, 0",
     "  out = reshape(operand, operand.shape[:1] + new_sizes, dimensions,\n"
     "                out_sharding=sharding)\n  return out, 1"),

    ("Bug 27", "jax/_src/lax/slicing.py",
     "    operand = batching.moveaxis(operand, operand_bdim, 0)\n    operand_bdim = 0",
     "    operand = batching.moveaxis(operand, operand_bdim + 1, 0)\n    operand_bdim = 0"),

    ("Bug 28", "jax/_src/lax/slicing.py",
     "    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))",
     "    offset_dims = (0,) + tuple(np.add(2, dimension_numbers.offset_dims))"),
]


def apply_single_bug(clean_dir, work_dir, name, rel_path, old, new):
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


def apply_decoys(clean_dir, decoys_dir, decoys_patch):
    """Create a decoys state by applying decoys.patch to clean source."""
    if os.path.exists(decoys_dir):
        shutil.rmtree(decoys_dir)
    shutil.copytree(clean_dir, decoys_dir)
    result = subprocess.run(
        ["patch", "-d", decoys_dir, "-p0", "-f", "--no-backup-if-mismatch"],
        stdin=open(decoys_patch, "rb"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"decoys.patch failed: {result.stderr.decode(errors='replace')}"
        )
    print("  ✅ decoys state prepared")


def generate_patch(decoys_dir, work_dir, name, rel_path, old, new, output_path):
    # Reset work dir to decoys state
    shutil.copytree(decoys_dir, work_dir, dirs_exist_ok=True)
    apply_single_bug(decoys_dir, work_dir, name, rel_path, old, new)

    # Generate unified diff for the changed file only (decoys -> decoys+bug)
    decoys_file = os.path.join(decoys_dir, rel_path)
    buggy_file = os.path.join(work_dir, rel_path)
    diff = subprocess.run(
        ["diff", "-uN", decoys_file, buggy_file],
        capture_output=True, text=True
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"diff failed for {name}")
    patch_text = diff.stdout
    if not patch_text:
        raise RuntimeError(f"{name}: no diff generated")

    # Rewrite paths to be relative to JAX_PKG
    lines = patch_text.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("--- "):
            new_lines.append(f"--- a/{rel_path}")
        elif line.startswith("+++ "):
            new_lines.append(f"+++ b/{rel_path}")
        else:
            new_lines.append(line)
    patch_text = "\n".join(new_lines) + "\n"

    with open(output_path, "w") as f:
        f.write(patch_text)
    print(f"  ✅ {name} -> {output_path}")


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <decoys_patch> <output_dir>")
        sys.exit(1)

    clean_dir = sys.argv[1]
    decoys_patch = sys.argv[2]
    output_dir = sys.argv[3]
    decoys_dir = os.path.join(output_dir, ".decoys")
    work_dir = os.path.join(output_dir, ".work")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare decoys state once
    apply_decoys(clean_dir, decoys_dir, decoys_patch)

    for name, rel_path, old, new in BUGS:
        output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.patch")
        generate_patch(decoys_dir, work_dir, name, rel_path, old, new, output_path)

    print(f"\nGenerated {len(BUGS)} per-bug patches in {output_dir}")


if __name__ == "__main__":
    main()

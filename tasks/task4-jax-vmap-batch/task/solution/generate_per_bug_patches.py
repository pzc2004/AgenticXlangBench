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
     "  if dst is infer:\n    return x"),

    ("Bug 2", "jax/_src/interpreters/batching.py",
     "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, **params), None",
     "  if False:\n    return prim.bind(*batched_args, **params), None"),

    ("Bug 3", "jax/_src/interpreters/batching.py",
     "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, axes=axes, **params), None",
     "  if False:\n    return prim.bind(*batched_args, axes=axes, **params), None"),

    ("Bug 4", "jax/_src/interpreters/batching.py",
     "    elif unmapped_args:",
     "    elif False:"),

    ("Bug 5", "jax/_src/interpreters/batching.py",
     "  elif dst is None and sum_match or dst is sum_axis:\n    return x.sum(src)",
     "  elif False:\n    return x.sum(src + 1 if src is not None else src)"),

    ("Bug 7", "jax/_src/interpreters/ad.py",
     "  out_tangents = tuple(t for t, nz in zip(out_tangents, nzs_out) if nz)\n"
     "  out_tangents = map(partial(tangent_trace.to_jaxpr_tracer, source_info=source_info), out_tangents)",
     "  out_tangents = tuple(out_tangents)\n"
     "  out_tangents = map(partial(tangent_trace.to_jaxpr_tracer, source_info=source_info), out_tangents)"),

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
     "    if p in fancy_primitive_batchers:",
     "    if p not in fancy_primitive_batchers:"),

    ("Bug 17", "jax/_src/interpreters/batching.py",
     "    return broadcast(x, axis_data.size,",
     "    return broadcast(x, axis_data.size - 1,"),

    ("Bug 18", "jax/_src/interpreters/ad.py",
     "      out_primals, out_tangents = ans.map(linearize_trace.to_primal_tangent_pair).unzip2()\n"
     "      del linearize_trace, ans, tracers\n"
     "  nzs_out = tuple(type(t) is not Zero for t in out_tangents)",
     "      out_primals, out_tangents = ans.map(linearize_trace.to_primal_tangent_pair).unzip2()\n"
     "      del linearize_trace, ans, tracers\n"
     "  nzs_out = tuple(type(t) is Zero for t in out_tangents)"),

    ("Bug 19", "jax/_src/interpreters/ad.py",
     "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, is_vjp)",
     "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, not is_vjp)"),

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
     "    return reduce_p.bind(*(operands + init_values),\n                         computation=computation,\n                         dimensions=tuple(new_dimensions),\n                         jaxpr=jaxpr), [0] * num_operands"),

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
     "  if _is_ragged_contracting(batched_args[0].ndim - 1,\n"
     "                            ragged_dot_dimension_numbers):\n"
     "    result_batch_dim += 1\n"
     "  return batched_out, result_batch_dim\n",
     "  if _is_ragged_contracting(batched_args[0].ndim - 1,\n"
     "                            ragged_dot_dimension_numbers):\n"
     "    result_batch_dim += 1\n"
     "  return batched_out, result_batch_dim + 1\n"),

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


def generate_patch(clean_dir, work_dir, name, rel_path, old, new, output_path):
    # Reset work dir to clean
    shutil.copytree(clean_dir, work_dir, dirs_exist_ok=True)
    apply_single_bug(clean_dir, work_dir, name, rel_path, old, new)

    # Generate unified diff for the changed file only
    clean_file = os.path.join(clean_dir, rel_path)
    buggy_file = os.path.join(work_dir, rel_path)
    diff = subprocess.run(
        ["diff", "-uN", clean_file, buggy_file],
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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <output_dir>")
        sys.exit(1)

    clean_dir = sys.argv[1]
    output_dir = sys.argv[2]
    work_dir = os.path.join(output_dir, ".work")

    os.makedirs(output_dir, exist_ok=True)

    for name, rel_path, old, new in BUGS:
        output_path = os.path.join(output_dir, f"{name.replace(' ', '_')}.patch")
        generate_patch(clean_dir, work_dir, name, rel_path, old, new, output_path)

    print(f"\nGenerated {len(BUGS)} per-bug patches in {output_dir}")


if __name__ == "__main__":
    main()

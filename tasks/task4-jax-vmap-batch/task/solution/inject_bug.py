#!/usr/bin/env python3
"""
Task 4: JAX vmap+grad bug injection
所有 bug 不加注释，直接改代码
"""
import os, sys

try:
    import jax._src.lax.slicing as slicing_mod
    JAX_PKG = os.path.dirname(os.path.dirname(os.path.dirname(slicing_mod.__file__)))
except ImportError:
    JAX_PKG = "/usr/local/lib/python3.12/site-packages/jax"

SRC_DIR = os.path.join(JAX_PKG, "_src")
LAX_DIR = os.path.join(SRC_DIR, "lax")
INTERP_DIR = os.path.join(SRC_DIR, "interpreters")
BATCHING_PY = os.path.join(INTERP_DIR, "batching.py")
AD_PY = os.path.join(INTERP_DIR, "ad.py")
LAX_PY = os.path.join(LAX_DIR, "lax.py")
SLICING_PY = os.path.join(LAX_DIR, "slicing.py")

REVERSE = "--reverse" in sys.argv

def read_file(filepath):
    with open(filepath) as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)

def apply_bug(filepath, old, new, name):
    content = read_file(filepath)
    if REVERSE:
        old, new = new, old
    if old in content:
        content = content.replace(old, new, 1)
        write_file(filepath, content)
        print(f"  ✅ {name}")
        return True
    print(f"  ⚠️ {name}: pattern not found")
    return False


def inject_all_bugs():
    count = 0

    # === 删除型 bug（直接删代码，不加注释）===

    # Bug 1: 删除 matchaxis src==dst 检查
    if apply_bug(BATCHING_PY,
        "  if src == dst or dst is infer:\n    return x",
        "  if dst is infer:\n    return x",
        "Bug 1: 删除 matchaxis src==dst"):
        count += 1

    # Bug 2: 删除 vectorized_batcher early return
    if apply_bug(BATCHING_PY,
        "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, **params), None",
        "  if False:\n    return prim.bind(*batched_args, **params), None",
        "Bug 2: 删除 vectorized early return"):
        count += 1

    # Bug 3: 删除 reducer_batcher early return
    if apply_bug(BATCHING_PY,
        "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "  if False:\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "Bug 3: 删除 reducer early return"):
        count += 1

    # Bug 4: 删除 unmapped_args 处理
    if apply_bug(BATCHING_PY,
        "    elif unmapped_args:",
        "    elif False:",
        "Bug 4: 删除 unmapped_args"):
        count += 1

    # Bug 5: 删除 sum_match + sum axis +1（合并）
    if apply_bug(BATCHING_PY,
        "  elif dst is None and sum_match or dst is sum_axis:\n    return x.sum(src)",
        "  elif False:\n    return x.sum(src + 1 if src is not None else src)",
        "Bug 5: 删除 sum_match + sum axis +1"):
        count += 1

    # Bug 7: 删除 nzs_out 过滤
    if apply_bug(AD_PY,
        "  out_tangents = tuple(t for t, nz in zip(out_tangents, nzs_out) if nz)",
        "  out_tangents = tuple(out_tangents)",
        "Bug 7: 删除 nzs_out 过滤"):
        count += 1

    # === 维度偏移型 bug ===

    # Bug 8: vectorized_batcher bdim +1
    if apply_bug(BATCHING_PY,
        "  return prim.bind(*batched_args, **params), batch_dims[0]",
        "  return prim.bind(*batched_args, **params), batch_dims[0] + 1",
        "Bug 8: vectorized bdim +1"):
        count += 1

    # Bug 9: reducer_batcher bdim_out +1
    if apply_bug(BATCHING_PY,
        "    return prim.bind(operand, axes=axes, **params), bdim_out",
        "    return prim.bind(operand, axes=axes, **params), bdim_out + 1",
        "Bug 9: reducer bdim_out +1"):
        count += 1

    # Bug 10: BatchTracer bdim +1
    if apply_bug(BATCHING_PY,
        "    self.batch_dim = batch_dim",
        "    self.batch_dim = batch_dim if batch_dim is None else batch_dim + 1",
        "Bug 10: BatchTracer bdim +1"):
        count += 1

    # Bug 11: process_primitive dim_out +1
    if apply_bug(BATCHING_PY,
        "          return (BatchTracer(self, val_out, dim_out, src)",
        "          return (BatchTracer(self, val_out, dim_out + 1 if dim_out is not None else None, src)",
        "Bug 11: process_primitive dim_out +1"):
        count += 1

    # Bug 12: matchaxis src/dst swapped
    if apply_bug(BATCHING_PY,
        "    return moveaxis(x, src, dst)",
        "    return moveaxis(x, dst, src)",
        "Bug 12: matchaxis src/dst swapped"):
        count += 1

    # Bug 13: reducer axes -1
    if apply_bug(BATCHING_PY,
        "    axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))",
        "    axes = tuple(np.where(np.less(axes, bdim), axes, np.subtract(axes, 1)))",
        "Bug 13: reducer axes -1"):
        count += 1

    # Bug 14: (与 Bug 1 冲突，已删除)

    # === 条件反转型 bug ===

    # Bug 15: fancy check inverted
    if apply_bug(BATCHING_PY,
        "    if p in fancy_primitive_batchers:",
        "    if p not in fancy_primitive_batchers:",
        "Bug 15: fancy check inverted"):
        count += 1

    # Bug 16: (已合并到 Bug 5)

    # Bug 17: broadcast size -1
    if apply_bug(BATCHING_PY,
        "    return broadcast(x, axis_data.size,",
        "    return broadcast(x, axis_data.size - 1,",
        "Bug 17: broadcast size -1"):
        count += 1

    # === ad.py bug ===

    # Bug 18: nzs_out inverted
    if apply_bug(AD_PY,
        "  nzs_out = tuple(type(t) is not Zero for t in out_tangents)",
        "  nzs_out = tuple(type(t) is Zero for t in out_tangents)",
        "Bug 18: nzs_out inverted"):
        count += 1

    # Bug 19: is_vjp inverted
    if apply_bug(AD_PY,
        "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, is_vjp)",
        "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, not is_vjp)",
        "Bug 19: is_vjp inverted"):
        count += 1

    # === lax.py bug ===

    # Bug 20: reshape dims +1
    if apply_bug(LAX_PY,
        "    dimensions = (0,) + tuple(np.add(1, dimensions))",
        "    dimensions = (0,) + tuple(np.add(2, dimensions))",
        "Bug 20: reshape dims +1"):
        count += 1

    # Bug 21: transpose perm off-by-one
    if apply_bug(LAX_PY,
        "  perm = (bdim,) + tuple(i if i < bdim else i+1 for i in permutation)",
        "  perm = (bdim,) + tuple(i if i <= bdim else i+1 for i in permutation)",
        "Bug 21: transpose perm off-by-one"):
        count += 1

    # Bug 22: transpose res_bdim wrong
    if apply_bug(LAX_PY,
        "  res_bdim = 0",
        "  res_bdim = bdim",
        "Bug 22: transpose res_bdim"):
        count += 1

    # Bug 23: concatenate bdim wrong
    if apply_bug(LAX_PY,
        "  return concatenate(operands, dimension + 1), 0",
        "  return concatenate(operands, dimension + 1), 1",
        "Bug 23: concatenate bdim = 1"):
        count += 1

    # Bug 24: select_n bdim wrong
    if apply_bug(LAX_PY,
        "      return select_n(which, *cases), which_bdim",
        "      return select_n(which, *cases), which_bdim + 1",
        "Bug 24: select_n bdim +1"):
        count += 1

    # Bug 25: reduce batch rule
    if apply_bug(LAX_PY,
        "    return reduce_p.bind(*(operands + init_values),\n                         computation=computation,\n                         dimensions=tuple(new_dimensions),\n                         jaxpr=jaxpr), new_operand_bdims",
        "    return reduce_p.bind(*(operands + init_values),\n                         computation=computation,\n                         dimensions=tuple(new_dimensions),\n                         jaxpr=jaxpr), [0] * num_operands",
        "Bug 25: reduce bdim wrong"):
        count += 1

    # Bug 26: dot_general result_batch_dim +1 (两处)
    if apply_bug(LAX_PY,
        "  return batched_out, result_batch_dim\n",
        "  return batched_out, result_batch_dim + 1\n",
        "Bug 26a: dot_general result_batch_dim +1"):
        count += 1
    if apply_bug(LAX_PY,
        "  return batched_out, result_batch_dim\n",
        "  return batched_out, result_batch_dim + 1\n",
        "Bug 26b: dot result_batch_dim +1"):
        count += 1

    # === slicing.py bug ===

    # Bug 27: gather bdim +1
    if apply_bug(SLICING_PY,
        "    operand = batching.moveaxis(operand, operand_bdim, 0)\n    operand_bdim = 0",
        "    operand = batching.moveaxis(operand, operand_bdim + 1, 0)\n    operand_bdim = 0",
        "Bug 27: gather bdim +1"):
        count += 1

    # Bug 28: gather offset_dims +1
    if apply_bug(SLICING_PY,
        "    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))",
        "    offset_dims = (0,) + tuple(np.add(2, dimension_numbers.offset_dims))",
        "Bug 28: gather offset_dims +1"):
        count += 1

    return count


def main():
    print("=" * 60)
    print("注入 bug")
    print("=" * 60)

    print("\n>>> 注入 bug:")
    bug_count = inject_all_bugs()

    print(f"\n{'=' * 60}")
    print(f"总计: {bug_count} 个 bug")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

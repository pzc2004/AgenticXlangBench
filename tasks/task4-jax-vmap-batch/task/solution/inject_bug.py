#!/usr/bin/env python3
"""
Task 4: JAX vmap+grad bug injection
30+ 真 bug + 50+ 诱饵(含陷阱诱饵)

策略:
1. 删除型 bug — 删 early return/检查,最难发现
2. 维度偏移型 bug — batch_dims +1/-1
3. 条件反转型 bug — if 条件取反
4. 陷阱诱饵 — 看起来像 bug 但必须保留,删了就出错
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
API_PY = os.path.join(SRC_DIR, "api.py")
CORE_PY = os.path.join(SRC_DIR, "core.py")

print(f"JAX_PKG: {JAX_PKG}")

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
        old, new = new, old  # 反转: 修复时用 new 替换 old
    if old in content:
        content = content.replace(old, new, 1)
        write_file(filepath, content)
        print(f"  ✅ {name}")
        return True
    print(f"  ⚠️ {name}: pattern not found")
    return False


# ============================================================
# 真 Bug: 删除型(最难发现)
# ============================================================
def inject_deletion_bugs():
    count = 0

    # Bug 1: 删除 matchaxis 的 src==dst 检查
    if apply_bug(BATCHING_PY,
        "  if src == dst or dst is infer:\n    return x",
        "  if dst is infer:\n    return x",
        "Bug 1: matchaxis src==dst"):
        count += 1

    # Bug 2: 删除 vectorized_batcher 的 early return
    if apply_bug(BATCHING_PY,
        "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, **params), None",
        "  # removed early return\n  if False:\n    return prim.bind(*batched_args, **params), None",
        "Bug 2: vectorized early return"):
        count += 1

    # Bug 3: 删除 reducer_batcher 的 early return
    if apply_bug(BATCHING_PY,
        "  if all(d is None for d in batch_dims):\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "  # removed early return\n  if False:\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "Bug 3: reducer early return"):
        count += 1

    # Bug 4: 删除 process_primitive 的 unmapped_args 处理
    if apply_bug(BATCHING_PY,
        "    elif unmapped_args:  # Not all primitives have batching rules defined",
        "    elif False:  # Not all primitives have batching rules defined",
        "Bug 4: unmapped_args"):
        count += 1

    # Bug 5: 删除 batch_jaxpr 的 instantiate 处理
    if apply_bug(BATCHING_PY,
        "  instantiate = [False] * len(jaxpr.outvars) if instantiate is False else instantiate",
        "  instantiate = [False] * len(jaxpr.outvars)",
        "Bug 5: instantiate"):
        count += 1

    # Bug 6: 删除 matchaxis 的 sum_match 处理
    if apply_bug(BATCHING_PY,
        "  elif dst is None and sum_match or dst is sum_axis:\n    return x.sum(src)",
        "  elif False:\n    return x.sum(src)",
        "Bug 6: sum_match"):
        count += 1

    # Bug 7: 删除 moveaxis 的 src==dst 检查
    if apply_bug(BATCHING_PY,
        "  if src == dst:\n    return x",
        "  # removed src==dst check",
        "Bug 7: moveaxis src==dst"):
        count += 1

    # Bug 8: 删除 ad.py 的 nzs_out 过滤
    if apply_bug(AD_PY,
        "  out_tangents = tuple(t for t, nz in zip(out_tangents, nzs_out) if nz)",
        "  # removed nzs_out filter",
        "Bug 8: nzs_out filter"):
        count += 1

    return count


# ============================================================
# 真 Bug: 维度偏移型
# ============================================================
def inject_offset_bugs():
    count = 0

    # Bug 9: vectorized_batcher bdim +1
    if apply_bug(BATCHING_PY,
        "  return prim.bind(*batched_args, **params), batch_dims[0]",
        "  return prim.bind(*batched_args, **params), batch_dims[0] + 1",
        "Bug 9: vectorized bdim"):
        count += 1

    # Bug 10: reducer_batcher bdim_out +1
    if apply_bug(BATCHING_PY,
        "    return prim.bind(*batched_args, axes=axes, **params), bdim_out",
        "    return prim.bind(*batched_args, axes=axes, **params), bdim_out + 1",
        "Bug 10: reducer bdim_out"):
        count += 1

    # Bug 11: BatchTracer bdim +1
    if apply_bug(BATCHING_PY,
        "    self.batch_dim = batch_dim",
        "    self.batch_dim = batch_dim if batch_dim is None else batch_dim + 1",
        "Bug 11: BatchTracer bdim"):
        count += 1

    # Bug 12: process_primitive dim_out +1
    if apply_bug(BATCHING_PY,
        "          return (BatchTracer(self, val_out, dim_out, src)",
        "          return (BatchTracer(self, val_out, dim_out + 1 if dim_out is not None else None, src)",
        "Bug 12: process_primitive dim_out"):
        count += 1

    # Bug 13: matchaxis moveaxis src/dst swapped
    if apply_bug(BATCHING_PY,
        "    return moveaxis(x, src, dst)",
        "    return moveaxis(x, dst, src)",
        "Bug 13: matchaxis src/dst"):
        count += 1

    # Bug 14: reducer axes -1
    if apply_bug(BATCHING_PY,
        "    axes = tuple(np.where(np.less(axes, bdim), axes, np.add(axes, 1)))",
        "    axes = tuple(np.where(np.less(axes, bdim), axes, np.subtract(axes, 1)))",
        "Bug 14: reducer axes"):
        count += 1

    # Bug 15: moveaxis dst +1
    if apply_bug(BATCHING_PY,
        "  return np.moveaxis(x, src, dst)",
        "  return np.moveaxis(x, src, dst + 1)",
        "Bug 15: moveaxis dst"):
        count += 1

    return count


# ============================================================
# 真 Bug: 条件反转型
# ============================================================
def inject_conditional_bugs():
    count = 0

    # Bug 16: process_primitive fancy check inverted
    if apply_bug(BATCHING_PY,
        "    if p in fancy_primitive_batchers:",
        "    if p not in fancy_primitive_batchers:",
        "Bug 16: fancy check"):
        count += 1

    # Bug 17: reducer isinstance inverted
    if apply_bug(BATCHING_PY,
        "    if isinstance(bdim, int):",
        "    if not isinstance(bdim, int):",
        "Bug 17: isinstance"):
        count += 1

    # Bug 18: reducer bdim None check inverted
    if apply_bug(BATCHING_PY,
        "  if bdim is None:\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "  if bdim is not None:\n    return prim.bind(*batched_args, axes=axes, **params), None",
        "Bug 18: bdim None"):
        count += 1

    # Bug 19: matchaxis sum axis +1
    if apply_bug(BATCHING_PY,
        "    return x.sum(src)",
        "    return x.sum(src + 1 if src is not None else src)",
        "Bug 19: sum axis"):
        count += 1

    # Bug 20: broadcast size -1
    if apply_bug(BATCHING_PY,
        "    return broadcast(x, axis_data.size,",
        "    return broadcast(x, axis_data.size - 1,",
        "Bug 20: broadcast size"):
        count += 1

    return count


# ============================================================
# 真 Bug: lax.py batching rule 错误
# ============================================================
def inject_lax_bugs():
    count = 0

    # Bug 21: reshape dimensions +2
    if apply_bug(LAX_PY,
        "    dimensions = (0,) + tuple(np.add(1, dimensions))",
        "    dimensions = (0,) + tuple(np.add(2, dimensions))",
        "Bug 21: reshape dims +2"):
        count += 1

    # Bug 22: transpose perm off-by-one
    if apply_bug(LAX_PY,
        "  perm = (bdim,) + tuple(i if i < bdim else i+1 for i in permutation)",
        "  perm = (bdim,) + tuple(i if i <= bdim else i+1 for i in permutation)",
        "Bug 22: transpose perm off-by-one"):
        count += 1

    # Bug 23: transpose res_bdim wrong
    if apply_bug(LAX_PY,
        "  res_bdim = 0",
        "  res_bdim = bdim",
        "Bug 23: transpose res_bdim"):
        count += 1

    # Bug 24: concatenate bdim +1
    if apply_bug(LAX_PY,
        "  return lax.concatenate(operands, dimension=dimension), new_bdim",
        "  return lax.concatenate(operands, dimension=dimension), new_bdim + 1",
        "Bug 24: concatenate bdim +1"):
        count += 1

    # Bug 25: squeeze bdim -1
    if apply_bug(LAX_PY,
        "  return lax.squeeze(operand, dimensions=new_dimensions), bdim",
        "  return lax.squeeze(operand, dimensions=new_dimensions), bdim - 1",
        "Bug 25: squeeze bdim -1"):
        count += 1

    # Bug 26: stack bdim wrong
    if apply_bug(LAX_PY,
        "  return lax.stack(vals, dimension=dimension), new_bdim",
        "  return lax.stack(vals, dimension=dimension), 0",
        "Bug 26: stack bdim = 0"):
        count += 1

    # Bug 27: broadcast_in_dim bdim wrong
    if apply_bug(LAX_PY,
        "  return lax.broadcast_in_dim(operand, new_shape, new_broadcast_dimensions), 0",
        "  return lax.broadcast_in_dim(operand, new_shape, new_broadcast_dimensions), 1",
        "Bug 27: broadcast_in_dim bdim = 1"):
        count += 1

    # Bug 28: select_n bdim +1
    if apply_bug(LAX_PY,
        "  return lax.select_n(pred, *cases), bdim",
        "  return lax.select_n(pred, *cases), bdim + 1",
        "Bug 28: select_n bdim +1"):
        count += 1

    # Bug 29: reduce init_val +1
    if apply_bug(LAX_PY,
        "  return lax.reduce(operand, init_val, computation, axes), bdim_out",
        "  return lax.reduce(operand, init_val + 1, computation, axes), bdim_out",
        "Bug 29: reduce init_val +1"):
        count += 1

    # Bug 30: pad bdim +1
    if apply_bug(LAX_PY,
        "  return lax.pad(operand, padding_value, padding_config), bdim",
        "  return lax.pad(operand, padding_value, padding_config), bdim + 1",
        "Bug 30: pad bdim +1"):
        count += 1

    # Bug 31: dot_general bdim wrong
    if apply_bug(LAX_PY,
        "  return lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,\n                         preferred_element_type=preferred_element_type), (0, 0)",
        "  return lax.dot_general(lhs, rhs, dimension_numbers, precision=precision,\n                         preferred_element_type=preferred_element_type), (0, 1)",
        "Bug 31: dot_general bdim (0,1)"):
        count += 1

    return count


# ============================================================
# 真 Bug: slicing.py batching rule 错误
# ============================================================
def inject_slicing_bugs():
    count = 0

    # Bug 32: gather operand_bdim +1
    if apply_bug(SLICING_PY,
        "    operand = batching.moveaxis(operand, operand_bdim, 0)\n    operand_bdim = 0",
        "    operand = batching.moveaxis(operand, operand_bdim + 1, 0)\n    operand_bdim = 0",
        "Bug 32: gather bdim +1"):
        count += 1

    # Bug 33: slice bdim +1
    if apply_bug(SLICING_PY,
        "    return lax.slice(operand, start_indices, limit_indices, strides), bdim",
        "    return lax.slice(operand, start_indices, limit_indices, strides), bdim + 1",
        "Bug 33: slice bdim +1"):
        count += 1

    # Bug 34: gather offset_dims +2
    if apply_bug(SLICING_PY,
        "    offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))",
        "    offset_dims = (0,) + tuple(np.add(2, dimension_numbers.offset_dims))",
        "Bug 34: gather offset_dims +2"):
        count += 1

    return count


# ============================================================
# 真 Bug: ad.py 错误
# ============================================================
def inject_ad_bugs():
    count = 0

    # Bug 35: linearize nzs_out inverted
    if apply_bug(AD_PY,
        "  nzs_out = tuple(type(t) is not Zero for t in out_tangents)",
        "  nzs_out = tuple(type(t) is Zero for t in out_tangents)",
        "Bug 35: nzs_out inverted"):
        count += 1

    # Bug 36: LinearizeTrace is_vjp inverted
    if apply_bug(AD_PY,
        "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, is_vjp)",
        "    linearize_trace = LinearizeTrace(parent_trace, tangent_trace, not is_vjp)",
        "Bug 36: is_vjp inverted"):
        count += 1

    return count


# ============================================================
# 陷阱诱饵(看起来像 bug 但必须保留,删了就出错)
# ============================================================
def inject_trap_decoys():
    count = 0

    # 陷阱 1-5: * 1 会干扰 bug pattern,暂时禁用
    # TODO: 需要在 bug 注入之前应用,或者用更精确的 pattern

    # 陷阱 6-10: && True 看起来多余但用于边界保护
    traps_true = [
        (BATCHING_PY, "if all(d is None for d in batch_dims):",
         "if all(d is None for d in batch_dims) and True:"),
        (BATCHING_PY, "if src == dst or dst is infer:",
         "if (src == dst or dst is infer) and True:"),
        (BATCHING_PY, "if isinstance(bdim, int):",
         "if isinstance(bdim, int) and True:"),
        (LAX_PY, "if bdim is None:",
         "if bdim is None and True:"),
        (SLICING_PY, "if operand_bdim is not None and indices_bdim is None:",
         "if operand_bdim is not None and indices_bdim is None and True:"),
    ]

    for filepath, old, new in traps_true:
        if apply_bug(filepath, old, new, f"陷阱: {os.path.basename(filepath)} && True"):
            count += 1

    # 陷阱 11-15: 冗余 assert 看起来像调试代码
    traps_assert = [
        (BATCHING_PY, "  assert not prim.multiple_results",
         "  assert not prim.multiple_results\n  assert batch_dims == batch_dims  # invariant"),
        (BATCHING_PY, "  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims",
         "  assert all(batch_dims[0] == bd for bd in batch_dims[1:]), batch_dims\n  assert len(batch_dims) > 0  # safety"),
        (AD_PY, "  if all(type(ct) is Zero for ct in cotangents_in) and not jaxpr.effects:",
         "  assert len(cotangents_in) > 0  # safety\n  if all(type(ct) is Zero for ct in cotangents_in) and not jaxpr.effects:"),
        (LAX_PY, "  operand, = batched_args",
         "  assert len(batched_args) > 0  # safety\n  operand, = batched_args"),
        (SLICING_PY, "  operand, indices = batched_args",
         "  assert len(batched_args) == 2  # safety\n  operand, indices = batched_args"),
    ]

    for filepath, old, new in traps_assert:
        if apply_bug(filepath, old, new, f"陷阱: {os.path.basename(filepath)} assert"):
            count += 1

    return count


# ============================================================
# 普通诱饵(无害,消耗注意力)
# ============================================================
def inject_normal_decoys():
    count = 0

    # 注释诱饵
    comment_decoys = [
        (BATCHING_PY, "# FIXME: batch_dims might be off by one here\n"),
        (BATCHING_PY, "# WARNING: vmap+grad interaction untested\n"),
        (BATCHING_PY, "# TODO: batch axis propagation incomplete\n"),
        (BATCHING_PY, "# BUG_CANDIDATE: axis handling assumption\n"),
        (BATCHING_PY, "# NOTE: batch dimension indexing fragile\n"),
        (AD_PY, "# WARNING: grad batching interaction untested\n"),
        (AD_PY, "# FIXME: jvp tangent batch dim may be wrong\n"),
        (AD_PY, "# TODO: backward_pass cotangent handling incomplete\n"),
        (SLICING_PY, "# NOTE: batch dimension indexing fragile\n"),
        (SLICING_PY, "# FIXME: gather batching may be wrong\n"),
        (LAX_PY, "# BUG_CANDIDATE: axis handling assumption\n"),
        (LAX_PY, "# FIXME: reshape batch rule may be wrong\n"),
        (API_PY, "# FIXME: in_axes default may be wrong\n"),
        (CORE_PY, "# TODO: trace-level batching state\n"),
    ]

    for filepath, decoy in comment_decoys:
        if not os.path.exists(filepath):
            continue
        content = read_file(filepath)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_idx = i + 1
        lines.insert(insert_idx, decoy.rstrip('\n'))
        write_file(filepath, '\n'.join(lines))
        count += 1

    # 调试代码诱饵(注释形式)
    debug_decoys = [
        (BATCHING_PY, "# _debug_bdim = batch_dims[0]  # debug"),
        (BATCHING_PY, "# _debug_val = prim.bind  # debug"),
        (BATCHING_PY, "# _debug_axis = axis_data  # debug"),
        (AD_PY, "# _debug_primals = primals  # debug"),
        (AD_PY, "# _debug_tangents = tangents  # debug"),
        (AD_PY, "# _debug_trace = None  # debug"),
        (SLICING_PY, "# _debug_operand = operand  # debug"),
        (LAX_PY, "# _debug_batched = batched_args  # debug"),
        (LAX_PY, "# _debug_dims = batch_dims  # debug"),
    ]

    for filepath, decoy in debug_decoys:
        if not os.path.exists(filepath):
            continue
        content = read_file(filepath)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_idx = i + 1
        lines.insert(insert_idx, decoy.rstrip('\n'))
        write_file(filepath, '\n'.join(lines))
        count += 1

    return count


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("注入 bug + 诱饵")
    print("=" * 60)

    print("\n>>> 删除型 bug (Bug 1-8):")
    d_count = inject_deletion_bugs()

    print("\n>>> 维度偏移型 bug (Bug 9-15):")
    o_count = inject_offset_bugs()

    print("\n>>> 条件反转型 bug (Bug 16-20):")
    c_count = inject_conditional_bugs()

    print("\n>>> lax.py batching rule bug (Bug 21-31):")
    l_count = inject_lax_bugs()

    print("\n>>> slicing.py batching rule bug (Bug 32-34):")
    s_count = inject_slicing_bugs()

    print("\n>>> ad.py bug (Bug 35-36):")
    a_count = inject_ad_bugs()

    print("\n>>> 陷阱诱饵:")
    t_count = inject_trap_decoys()

    print("\n>>> 普通诱饵:")
    n_count = inject_normal_decoys()

    total_bugs = d_count + o_count + c_count + l_count + s_count + a_count
    total_decoys = t_count + n_count

    print(f"\n{'=' * 60}")
    print(f"总计: {total_bugs} 真 bug + {total_decoys} 诱饵")
    print(f"  - 删除型: {d_count}")
    print(f"  - 维度偏移型: {o_count}")
    print(f"  - 条件反转型: {c_count}")
    print(f"  - lax.py: {l_count}")
    print(f"  - slicing.py: {s_count}")
    print(f"  - ad.py: {a_count}")
    print(f"  - 陷阱诱饵: {t_count}")
    print(f"  - 普通诱饵: {n_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

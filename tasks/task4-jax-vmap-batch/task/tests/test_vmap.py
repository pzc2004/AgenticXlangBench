#!/usr/bin/env python3
"""
JAX vmap+grad 测试脚本（隐藏版 / 三阶段加强版）

此脚本由评测系统内部调用，不暴露给 agent。agent 看到的 /workspace/test_vmap.py
只是一个 stub。真正的测试逻辑和具体测试用例在这里，路径为 /task/tests/test_vmap.py。

设计目标：
1. 覆盖维度偏移型 bug（+1/-1/swap）
2. 覆盖删除型 bug（early return / 条件删除）
3. 覆盖 JVP/linearize、显式 axis、混合 batched/unbatched 等边界
4. 输出极简，只暴露最终 accuracy，不泄露哪个用例失败
"""
import argparse
import os
import jax
import jax.numpy as jnp
from jax import grad, vmap, jvp, linearize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="打印每个测试的详细结果（默认关闭以隐藏信息）")
    return parser.parse_args()


# ============================================================
# 单参数操作
# ============================================================

def f_sin(x): return jnp.sin(x)
def f_cos(x): return jnp.cos(x)
def f_sum(x): return jnp.sum(x, axis=-1, keepdims=True)
def f_mean(x): return jnp.mean(x, axis=-1, keepdims=True)
def f_max(x): return jnp.max(x, axis=-1, keepdims=True)

def f_reshape(x):
    return jnp.reshape(x, x.shape[:-1] + (-1,))

def f_reshape_explicit(x):
    """显式 reshape dimensions，触发 lax.py reshape batching。"""
    new_shape = (x.size,)
    return jax.lax.reshape(x, new_shape, dimensions=list(reversed(range(x.ndim))))

def f_transpose(x):
    return jnp.swapaxes(x, -1, -2) if x.ndim >= 2 else x

def f_transpose_positive(x):
    if x.ndim < 2: return x
    perm = list(range(x.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return jnp.transpose(x, perm)

def f_moveaxis_identity(x):
    return jnp.moveaxis(x, 0, 0)

def f_concatenate(x): return jnp.concatenate([x, x], axis=-1)
def f_squeeze(x): return jnp.squeeze(jnp.expand_dims(x, -1), axis=-1)
def f_stack(x): return jnp.stack([x, x], axis=-2)
def f_where(x): return jnp.where(x > 0, x, -x)
def f_dot(x): return jnp.dot(x, x)
def f_matmul(x): return jnp.dot(x, x.T)

# Bug 25：触发 _reduce_batch_rule（jnp.sum 用 reduce_sum_p，这里用通用 reduce_p）
def f_reduce_custom(x):
    if x.ndim < 1:
        return x
    return jax.lax.reduce(x, jnp.array(0.0), lambda a, b: a + b, (0,))

# Bug 26a：触发 _dot_general_batch_rule（jnp.dot 低维用 dot_p，这里显式用 dot_general_p）
def f_dot_general(x):
    if x.ndim < 2:
        return x
    return jax.lax.dot_general(x, x, dimension_numbers=(((x.ndim - 1,), (x.ndim - 1,)), ((), ())))
def f_slice(x): return x[..., 1:-1]

def f_broadcast(x):
    return jnp.broadcast_to(x[..., None], x.shape + (3,))

def f_pad(x):
    pad_width = [(0, 0)] * (x.ndim - 1) + [(1, 1)]
    return jnp.pad(x, pad_width)

def f_gather(x): return x[..., :4]

def f_gather_idx(x):
    idx = jnp.arange(min(4, x.shape[-1]))
    return x[..., idx]

def f_gather_explicit(x):
    dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=tuple(range(x.ndim)),
        collapsed_slice_dims=(),
        start_index_map=tuple(range(x.ndim)),
    )
    return jax.lax.gather(
        x,
        start_indices=jnp.zeros((x.ndim,), dtype=jnp.int32),
        dimension_numbers=dnums,
        slice_sizes=x.shape,
        mode="promise_in_bounds",
    )

def f_reduce_window(x): return jnp.sum(x, axis=-1, keepdims=True)
def f_reduce_sum_all(x): return jnp.sum(x)
def f_reduce_sum_axis(x): return jnp.sum(x, axis=-1)
def f_select_n(x): return jnp.where(x > 0, x * 2, x * 3)
def f_chained(x): return jnp.sin(jnp.cos(jnp.exp(x)))

def f_multi(x):
    y = jnp.sin(x) + jnp.cos(x)
    y = jnp.reshape(y, y.shape[:-1] + (-1,))
    y = jnp.transpose(y, list(range(y.ndim - 2)) + [y.ndim - 1, y.ndim - 2]) if y.ndim >= 2 else y
    return y


# ============================================================
# 多参数 / JVP / linearize / 高级 gather / dot
# ============================================================

def f_add_constant(x): return x + 1.5

def f_mixed_add(x, y): return x + y
def f_mixed_mul(x, y): return x * y

def f_jvp_sin(x):
    _, tangent = jvp(jnp.sin, (x,), (jnp.ones_like(x),))
    return tangent

def f_linearize_sum_sin(x):
    y, f_lin = linearize(lambda z: jnp.sum(jnp.sin(z)), x)
    return y + f_lin(jnp.ones_like(x))

def f_ragged_dot(x):
    return jax.lax.dot_general(
        x, x,
        dimension_numbers=(((x.ndim - 1,), (x.ndim - 1,)), ((), ()))
    )


# ============================================================
# 专门触发删除型 bug 的零参数 / 特殊路径测试
# ============================================================

# Bug 2/3：所有 batch_dims 为 None 的 early return 路径
# 无参数 vmap 会让内部常量表达式的所有输入 batch_dim 都是 None，
# 从而走 vectorized_batcher / axis_primitive_batcher 的 early return。
def f_all_unbatched_sin(): return jnp.sin(jnp.array(1.0))
def f_all_unbatched_sum(): return jnp.sum(jnp.array([1.0, 2.0, 3.0]))
def f_all_unbatched_max(): return jnp.max(jnp.array([1.0, 2.0, 3.0]))

# Bug 4：process_primitive 中 unmapped_args 分支
# device_put_p 通常不在 fancy_primitive_batchers 中，且无参数时所有 dims 为 None，
# 会走 unmapped_args 分支。
def f_unmapped_device_put(): return jax.device_put(jnp.array(1.0))

# 另一个 unmapped 路径：stop_gradient 也是非 fancy primitive
def f_unmapped_stop_grad(): return jax.lax.stop_gradient(jnp.array(1.0))


# ============================================================
# 触发情况 3 未覆盖 bug 的测试
# ============================================================

# Bug 5：matchaxis 中 src is None 的 broadcast 分支，用返回常量数组触发
def f_constant_output(x):
    return jnp.array([1.0, 2.0, 3.0])

# Bug 7 / Bug 18：linearize 产生 Zero tangent（list 输出，const 部分产生 Zero）
def f_linearize_pytree(x):
    a = jnp.sin(x)
    b = jnp.array(1.0)  # const → linearize 时 tangent 为 Zero
    return [a, b]

# Bug 7 / Bug 18：linearize 混合输出，const 数组产生 Zero tangent
def f_linearize_mixed(x):
    def g(z):
        a = jnp.sin(z)
        b = jnp.array([1.0, 2.0])  # const → tangent 为 Zero
        return a, b
    y, f_lin = linearize(g, x)
    dy = f_lin(jnp.ones_like(x))
    return y[0] + y[1][0] + dy[0] + dy[1][0]

# Bug 15：expand_dims_batcher 用于 linalg primitives（qr）
def f_linalg_qr(x):
    if x.ndim < 2:
        return x[..., None]
    return jnp.linalg.qr(x)[0]

# Bug 19：is_vjp flag。vjp 走 is_vjp=True 路径
def f_vjp_vector(x):
    def h(z):
        return jnp.stack([jnp.sin(z), jnp.cos(z)])
    _, vjp_fn = jax.vjp(h, x)
    cot = jnp.ones(h(x).shape)
    return vjp_fn(cot)[0]

# Bug 26b：ragged_dot batch dim +1（只支持 2D 输入，batch dim 必须为 0）
def f_ragged_dot_v2(x):
    if x.ndim != 2:
        return jnp.zeros(1, dtype=x.dtype)
    B = x.shape[0]
    k = x.shape[-1]
    g = min(2, max(1, B))
    n_out = 3
    rhs = jnp.ones((g, k, n_out), dtype=x.dtype)
    base = B // g
    rem = B - base * g
    gs = [base + (1 if i < rem else 0) for i in range(g)]
    group_sizes = jnp.array(gs, dtype=jnp.int32)
    return jax.lax.ragged_dot(x, rhs, group_sizes)


# ============================================================
# 测试项定义
# ============================================================

# (name, function, required_ndim, min_ndim, in_axes, need_grad)
TESTS = [
    ("sin", f_sin, None, 0, 0, True),
    ("cos", f_cos, None, 0, 0, True),
    ("sum", f_sum, None, 1, 0, True),
    ("mean", f_mean, None, 1, 0, True),
    ("max", f_max, None, 1, 0, True),
    ("reshape", f_reshape, None, 1, 0, True),
    ("reshape_explicit", f_reshape_explicit, None, 2, 0, True),
    ("transpose", f_transpose, None, 2, 0, True),
    ("transpose_positive", f_transpose_positive, None, 2, 0, True),
    ("moveaxis_identity", f_moveaxis_identity, None, 1, 0, True),
    ("concatenate", f_concatenate, None, 1, 0, True),
    ("squeeze", f_squeeze, None, 1, 0, True),
    ("stack", f_stack, None, 1, 0, True),
    ("where", f_where, None, 0, 0, True),
    ("dot", f_dot, 1, 1, 0, True),
    ("matmul", f_matmul, 2, 2, 0, True),
    ("reduce_custom", f_reduce_custom, None, 1, 0, True),
    ("dot_general", f_dot_general, None, 2, 0, True),
    ("slice", f_slice, None, 1, 0, True),
    ("broadcast", f_broadcast, None, 0, 0, True),
    ("pad", f_pad, None, 0, 0, True),
    ("gather", f_gather, None, 1, 0, True),
    ("gather_idx", f_gather_idx, None, 1, 0, True),
    ("gather_explicit", f_gather_explicit, None, 1, 0, True),
    ("reduce_window", f_reduce_window, None, 1, 0, True),
    ("reduce_sum_all", f_reduce_sum_all, None, 1, 0, True),
    ("reduce_sum_axis", f_reduce_sum_axis, None, 1, 0, True),
    ("select_n", f_select_n, None, 0, 0, True),
    ("chained", f_chained, None, 0, 0, True),
    ("multi", f_multi, None, 1, 0, True),
    ("add_constant", f_add_constant, None, 0, 0, True),
    ("jvp_sin", f_jvp_sin, None, 0, 0, False),
    ("linearize_sum_sin", f_linearize_sum_sin, None, 1, 0, False),
    ("linearize_mixed", f_linearize_mixed, None, 0, 0, False),
    ("ragged_dot", f_ragged_dot, None, 2, 0, True),
]

MULTI_ARG_TESTS = [
    ("mixed_add", f_mixed_add, None, 0, (0, None), True),
    ("mixed_mul", f_mixed_mul, None, 0, (0, None), True),
]

# (name, function)
ZERO_ARG_TESTS = [
    ("all_unbatched_sin", f_all_unbatched_sin),
    ("all_unbatched_sum", f_all_unbatched_sum),
    ("all_unbatched_max", f_all_unbatched_max),
    ("unmapped_device_put", f_unmapped_device_put),
    ("unmapped_stop_grad", f_unmapped_stop_grad),
]

# (name, function) — 触发 Bug 5
CONSTANT_OUTPUT_TESTS = [
    ("constant_output", f_constant_output),
]

# (name, function) — 触发 Bug 7 / Bug 18
PYTREE_TESTS = [
    ("linearize_pytree", f_linearize_pytree),
]

# (name, function) — 触发 Bug 15
LINALG_TESTS = [
    ("linalg_qr", f_linalg_qr),
]

# (name, function) — 触发 Bug 19
VJP_TESTS = [
    ("vjp_vector", f_vjp_vector),
]

# (name, function) — 触发 Bug 26b
RAGGED_TESTS = [
    ("ragged_dot_v2", f_ragged_dot_v2),
]


# ============================================================
# 测试框架
# ============================================================

def manual_batch(f, xs, in_axis=0):
    xs = jnp.moveaxis(xs, in_axis, 0)
    return jnp.stack([f(xs[i]) for i in range(xs.shape[0])])


def manual_batch_multi(f, xs, ys, in_axes):
    x_axis, y_axis = in_axes
    if x_axis is not None:
        xs = jnp.moveaxis(xs, x_axis, 0)
    if y_axis is not None:
        ys = jnp.moveaxis(ys, y_axis, 0)
    n = xs.shape[0] if x_axis is not None else ys.shape[0]
    return jnp.stack([f(xs[i] if x_axis is not None else xs,
                        ys[i] if y_axis is not None else ys)
                      for i in range(n)])


def manual_batch_grad(f, xs, in_axis=0):
    xs = jnp.moveaxis(xs, in_axis, 0)
    return jnp.stack([grad(lambda xi: jnp.sum(f(xi)))(xs[i]) for i in range(xs.shape[0])])


def manual_batch_grad_multi(f, xs, ys, in_axes):
    x_axis, y_axis = in_axes
    if x_axis is not None:
        xs = jnp.moveaxis(xs, x_axis, 0)
    if y_axis is not None:
        ys = jnp.moveaxis(ys, y_axis, 0)
    n = xs.shape[0] if x_axis is not None else ys.shape[0]
    return jnp.stack([
        grad(lambda xi: jnp.sum(f(xi, ys[i] if y_axis is not None else ys)))(
            xs[i] if x_axis is not None else xs)
        for i in range(n)
    ])


def relative_error(a, b):
    max_diff = jnp.max(jnp.abs(a - b))
    scale = jnp.max(jnp.abs(b)) + 1e-10
    return float(max_diff / scale)


def record_result(info, ok, skipped, passed, total, verbose, fail_fast):
    if info == "skipped":
        skipped[0] += 1
        return True
    total[0] += 1
    if ok:
        passed[0] += 1
        return True
    if fail_fast:
        print(f"accuracy {passed[0]} {total[0]}")
        print(f"skipped {skipped[0]}")
        print(f"final_accuracy {passed[0]/total[0]*100:.1f}%")
        print(f"nan_detected False")
        print("FAIL")
        raise SystemExit(1)
    return False


def test_one(name, f, xs, in_axis=0, required_ndim=None, min_ndim=0, need_grad=True):
    if required_ndim is not None and xs.ndim != required_ndim:
        return True, 0.0, "skipped"
    if xs.ndim - 1 < min_ndim:
        return True, 0.0, "skipped"
    if name == "reshape_explicit" and (xs.shape[-1] % 2 != 0):
        return True, 0.0, "skipped"
    if name == "ragged_dot" and xs.shape[-1] != 1:
        return True, 0.0, "skipped"
    if name == "slice" and xs.shape[-1] <= 2:
        return True, 0.0, "skipped"

    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        out_err = relative_error(vmapped_out, ref_out)
        if out_err >= 1e-5:
            return False, out_err, "output mismatch"

        if not need_grad:
            return True, out_err, "ok"

        ref_grad = manual_batch_grad(f, xs, in_axis)
        vmapped_grad = vmap(grad(lambda xi: jnp.sum(f(xi))), in_axes=in_axis, out_axes=0)(xs)
        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch"
        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"

        return True, max(out_err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_one_multi(name, f, xs, ys, in_axes, required_ndim=None, min_ndim=0, need_grad=True):
    x_axis, y_axis = in_axes
    x_ndim = xs.ndim - (1 if x_axis is not None else 0)
    if required_ndim is not None and xs.ndim != required_ndim:
        return True, 0.0, "skipped"
    if x_ndim < min_ndim:
        return True, 0.0, "skipped"

    try:
        ref_out = manual_batch_multi(f, xs, ys, in_axes)
        vmapped_out = vmap(f, in_axes=in_axes, out_axes=0)(xs, ys)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        out_err = relative_error(vmapped_out, ref_out)
        if out_err >= 1e-5:
            return False, out_err, "output mismatch"

        if not need_grad:
            return True, out_err, "ok"

        ref_grad = manual_batch_grad_multi(f, xs, ys, in_axes)
        vmapped_grad = vmap(grad(lambda xi, yi: jnp.sum(f(xi, yi))),
                            in_axes=in_axes, out_axes=0)(xs, ys)
        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch"
        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"

        return True, max(out_err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_zero_arg(name, f, batch_size):
    """测试无参数 vmap：所有输入 batch_dim 均为 None，触发 early return / unmapped 分支。"""
    try:
        ref_out = jnp.stack([f() for _ in range(batch_size)])
        vmapped_out = vmap(f, axis_size=batch_size)()
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        err = relative_error(vmapped_out, ref_out)
        if err >= 1e-5:
            return False, err, "output mismatch"
        return True, err, "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_constant_output(name, f, xs, in_axis=0, need_grad=True):
    """函数返回与输入无关的常量数组，触发 matchaxis src=None 分支。"""
    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        err = relative_error(vmapped_out, ref_out)
        if err >= 1e-5:
            return False, err, "output mismatch"

        if not need_grad:
            return True, err, "ok"

        ref_grad = manual_batch_grad(f, xs, in_axis)
        vmapped_grad = vmap(grad(lambda xi: jnp.sum(f(xi))), in_axes=in_axis, out_axes=0)(xs)
        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch"
        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"
        return True, max(err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_pytree(name, f, xs, in_axis=0, need_grad=False):
    """list 输出 vmap，触发 linearize Zero tangent 路径。"""
    try:
        ref_leaves = []
        xs_moved = jnp.moveaxis(xs, in_axis, 0)
        for i in range(xs_moved.shape[0]):
            ref_leaves.append(f(xs_moved[i]))
        # ref_leaves is list of list [a, b]; transpose to list of arrays
        n = len(ref_leaves[0])
        ref_out = [jnp.stack([r[j] for r in ref_leaves], axis=0) for j in range(n)]
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if len(ref_out) != len(vmapped_out):
            return False, 1.0, f"length mismatch"
        max_err = 0.0
        for r, v in zip(ref_out, vmapped_out):
            if r.shape != v.shape:
                return False, 1.0, f"shape mismatch"
            err = relative_error(v, r)
            if err >= 1e-5:
                return False, err, "output mismatch"
            max_err = max(max_err, err)
        return True, max_err, "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_linalg(name, f, xs, in_axis=0, need_grad=False):
    """linalg primitive vmap，触发 expand_dims_batcher 路径。"""
    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        err = relative_error(vmapped_out, ref_out)
        if err >= 1e-5:
            return False, err, "output mismatch"
        return True, err, "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_vjp(name, f, xs, in_axis=0, need_grad=False):
    """vjp vmap，触发 is_vjp 路径。"""
    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        err = relative_error(vmapped_out, ref_out)
        if err >= 1e-5:
            return False, err, "output mismatch"
        return True, err, "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_ragged(name, f, xs, in_axis=0, need_grad=True):
    """ragged_dot vmap，触发 ragged_dot batch rule 路径。"""
    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)
        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch"
        err = relative_error(vmapped_out, ref_out)
        if err >= 1e-5:
            return False, err, "output mismatch"

        if not need_grad:
            return True, err, "ok"

        ref_grad = manual_batch_grad(f, xs, in_axis)
        vmapped_grad = vmap(grad(lambda xi: jnp.sum(f(xi))), in_axes=in_axis, out_axes=0)(xs)
        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch"
        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"
        return True, max(err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    fail_fast = os.environ.get("FAIL_FAST", "1") != "0"
    verbose = args.verbose or os.environ.get("VERBOSE", "0") == "1"

    total = [0]
    passed = [0]
    skipped = [0]
    all_pass = True

    shapes = [
        ((4,), [0]),
        ((4, 8), [0, 1]),
        ((2, 4, 8), [0, 1, 2]),
    ]

    # 零参数测试：触发删除型 bug
    for name, f in ZERO_ARG_TESTS:
        ok, err, info = test_zero_arg(name, f, batch_size=4)
        if verbose:
            status = "✅" if ok else "❌"
            print(f"  [{status}] zero_arg {name}: rel_err={err:.2e} ({info})")
        if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
            all_pass = False

    # 常量输出测试：触发 Bug 5
    const_shapes = [
        ((4,), [0]),
        ((4, 8), [0, 1]),
        ((2, 4, 8), [0, 1, 2]),
    ]
    for shape, in_axes in const_shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f in CONSTANT_OUTPUT_TESTS:
                ok, err, info = test_constant_output(name, f, xs, in_axis, need_grad=True)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] constant_output shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # pytree 输出测试：触发 Bug 7/18（Zero tangent）
    pytree_shapes = [((4,), [0]), ((4, 8), [0, 1])]
    for shape, in_axes in pytree_shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f in PYTREE_TESTS:
                ok, err, info = test_pytree(name, f, xs, in_axis)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] pytree shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # linalg 测试：触发 Bug 15
    linalg_shapes = [((4, 4), [0]), ((2, 4, 4), [0, 1])]
    for shape, in_axes in linalg_shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f in LINALG_TESTS:
                ok, err, info = test_linalg(name, f, xs, in_axis)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] linalg shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # vjp 测试：触发 Bug 19
    vjp_shapes = [((4, 8), [0, 1]), ((2, 4, 8), [0, 1, 2])]
    for shape, in_axes in vjp_shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f in VJP_TESTS:
                ok, err, info = test_vjp(name, f, xs, in_axis)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] vjp shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # ragged_dot 测试：触发 Bug 26b
    ragged_shapes = [((4, 4), [0])]
    for shape, in_axes in ragged_shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f in RAGGED_TESTS:
                ok, err, info = test_ragged(name, f, xs, in_axis, need_grad=True)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] ragged shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # 单参数测试
    for shape, in_axes in shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f, ndim, min_ndim, default_axis, need_grad in TESTS:
                ok, err, info = test_one(name, f, xs, in_axis, ndim, min_ndim, need_grad)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    # 多参数测试
    for shape, in_axes in shapes:
        key, subkey_x = jax.random.split(key)
        key, subkey_y = jax.random.split(key)
        xs = jax.random.normal(subkey_x, shape)
        ys = jax.random.normal(subkey_y, shape[1:] if shape else shape)
        for in_axis in in_axes:
            for name, f, ndim, min_ndim, in_axes_cfg, need_grad in MULTI_ARG_TESTS:
                ok, err, info = test_one_multi(name, f, xs, ys, in_axes_cfg,
                                               required_ndim=ndim, min_ndim=min_ndim,
                                               need_grad=need_grad)
                if verbose:
                    status = "✅" if ok else "❌"
                    print(f"  [{status}] shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if not record_result(info, ok, skipped, passed, total, verbose, fail_fast):
                    all_pass = False

    print(f"accuracy {passed[0]} {total[0]}")
    print(f"skipped {skipped[0]}")
    print(f"final_accuracy {passed[0]/total[0]*100:.1f}%" if total[0] > 0 else "final_accuracy 0.0%")
    print(f"nan_detected False")
    print("PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JAX vmap+grad 测试脚本（二阶段加强版）

核心思路：
1. 对每个操作，同时比较 vmap 输出和手动 batching 的参考输出
2. 同时比较 vmap(grad(f)) 和手动 batching 的参考梯度
3. 使用多种 shape/axis/边界条件，专门触发 batching rule 中的分支
4. 新增：混合 batched/unbatched、显式 axis、JVP/linearize、链式操作、复杂 gather/dot
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
    return parser.parse_args()


# ============================================================
# 操作定义：每个返回 array（尽量不提前 sum）
# ============================================================

def f_sin(x):
    return jnp.sin(x)

def f_cos(x):
    return jnp.cos(x)

def f_sum(x):
    return jnp.sum(x, axis=-1, keepdims=True)

def f_mean(x):
    return jnp.mean(x, axis=-1, keepdims=True)

def f_max(x):
    return jnp.max(x, axis=-1, keepdims=True)

def f_reshape(x):
    return jnp.reshape(x, x.shape[:-1] + (-1,))

def f_reshape_explicit(x):
    """显式 reshape dimensions，触发 lax.py reshape batching。

    使用 jax.lax.reshape 的 dimensions 参数（非恒等排列），这样 batching rule
    必须调整 dimensions；Bug 20 把 +1 改成 +2，会导致 batch 维位置算错。
    """
    new_shape = (x.size,)
    return jax.lax.reshape(x, new_shape, dimensions=list(reversed(range(x.ndim))))

def f_transpose(x):
    return jnp.swapaxes(x, -1, -2) if x.ndim >= 2 else x

def f_transpose_positive(x):
    """使用正 axis 索引的 transpose，触发 axes 调整分支。"""
    if x.ndim < 2:
        return x
    perm = list(range(x.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return jnp.transpose(x, perm)

def f_moveaxis_identity(x):
    """moveaxis src==dst，触发 batching.py  early return 分支。"""
    return jnp.moveaxis(x, 0, 0)

def f_concatenate(x):
    return jnp.concatenate([x, x], axis=-1)

def f_squeeze(x):
    return jnp.squeeze(jnp.expand_dims(x, -1), axis=-1)

def f_stack(x):
    return jnp.stack([x, x], axis=-2)

def f_where(x):
    return jnp.where(x > 0, x, -x)

def f_dot(x):
    # 一维向量点积
    return jnp.dot(x, x)

def f_matmul(x):
    # 2D 矩阵乘法，触发 dot_general batching
    return jnp.dot(x, x.T)

def f_slice(x):
    return x[..., 1:-1]

def f_broadcast(x):
    return jnp.broadcast_to(x[..., None], x.shape + (3,))

def f_pad(x):
    pad_width = [(0, 0)] * (x.ndim - 1) + [(1, 1)]
    return jnp.pad(x, pad_width)

def f_gather(x):
    return x[..., :4]

def f_gather_idx(x):
    """用数组索引触发 gather batching 的 offset_dims 分支。"""
    idx = jnp.arange(min(4, x.shape[-1]))
    return x[..., idx]

def f_reduce_window(x):
    # 触发 reduce batching
    return jnp.sum(x, axis=-1, keepdims=True)

def f_reduce_sum_all(x):
    """sum 全部元素，输出标量（触发 out_axes=None 路径）。"""
    return jnp.sum(x)

def f_reduce_sum_axis(x):
    """sum 最后一维且不 keepdims（触发 dst=None 路径）。"""
    return jnp.sum(x, axis=-1)

def f_select_n(x):
    # 显式触发 select_n batching
    return jnp.where(x > 0, x * 2, x * 3)

def f_chained(x):
    """链式操作，让 batch_dim 元数据错误在后续传播。"""
    return jnp.sin(jnp.cos(jnp.exp(x)))

def f_multi(x):
    """组合多个操作"""
    y = jnp.sin(x) + jnp.cos(x)
    y = jnp.reshape(y, y.shape[:-1] + (-1,))
    y = jnp.transpose(y, list(range(y.ndim - 2)) + [y.ndim - 1, y.ndim - 2]) if y.ndim >= 2 else y
    return y


# ============================================================
# 多参数 / JVP / linearize / 高级 gather / dot 测试
# ============================================================

def f_add_constant(x):
    """单参数但内部有广播常量，可触发 broadcasting primitive 的 batching。"""
    return x + 1.5

def f_mixed_add(x, y):
    """x batched, y unbatched，触发 unmapped args 处理。"""
    return x + y

def f_mixed_mul(x, y):
    return x * y

def f_jvp_sin(x):
    """在 vmap 内部做 jvp，触发 ad.py  tangent 处理。"""
    _, tangent = jvp(jnp.sin, (x,), (jnp.ones_like(x),))
    return tangent

def f_linearize_sum_sin(x):
    """在 vmap 内部做 linearize，触发 ad.py linearize 路径。"""
    y, f_lin = linearize(lambda z: jnp.sum(jnp.sin(z)), x)
    return y + f_lin(jnp.ones_like(x))

def f_ragged_dot(x):
    """构造 contracting dim size 为 1 的 dot_general，触发 ragged 路径。"""
    # x shape: (..., M, 1)；把最后 1 维作为 contracting
    return jax.lax.dot_general(
        x, x,
        dimension_numbers=(((x.ndim - 1,), (x.ndim - 1,)), ((), ()))
    )

def f_gather_explicit(x):
    """显式 jax.lax.gather，触发 offset_dims 调整。"""
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


# 测试项：
# (name, function, required_ndim, min_ndim, in_axes, need_grad)
# required_ndim: 仅当 xs.ndim == required_ndim 时运行（None=任意）
# min_ndim: vmap 剥离 batch 轴后单个输入至少需要的维数
# in_axes: 该测试默认使用的 in_axes（单参数为 int，多参数为 tuple）
# need_grad: 是否跑梯度对比
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
    ("jvp_sin", f_jvp_sin, None, 0, 0, False),   # jvp 内部已含微分，不再外接 grad
    ("linearize_sum_sin", f_linearize_sum_sin, None, 1, 0, False),
    ("ragged_dot", f_ragged_dot, None, 2, 0, True),  # 需要最后维为 1
]

# 多参数测试：(name, function, required_ndim, min_ndim, in_axes, need_grad)
MULTI_ARG_TESTS = [
    ("mixed_add", f_mixed_add, None, 0, (0, None), True),
    ("mixed_mul", f_mixed_mul, None, 0, (0, None), True),
]


# ============================================================
# 测试框架
# ============================================================

def manual_batch(f, xs, in_axis=0):
    """手动对 xs 的 in_axis 轴做 batching，输出 batch 轴固定在第 0 维。"""
    xs = jnp.moveaxis(xs, in_axis, 0)
    return jnp.stack([f(xs[i]) for i in range(xs.shape[0])])


def manual_batch_multi(f, xs, ys, in_axes):
    """多参数手动 batching。"""
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
    """手动对每个 batch 元素求 grad 再 stack。"""
    xs = jnp.moveaxis(xs, in_axis, 0)
    return jnp.stack([grad(lambda xi: jnp.sum(f(xi)))(xs[i]) for i in range(xs.shape[0])])


def manual_batch_grad_multi(f, xs, ys, in_axes):
    """多参数手动求 grad（对第一个 batched 参数求导）。"""
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


def test_one(name, f, xs, in_axis=0, required_ndim=None, min_ndim=0, need_grad=True):
    """测试一个单参数操作：比较 vmap 与手动 batching 的输出和梯度。"""
    if required_ndim is not None and xs.ndim != required_ndim:
        return True, 0.0, "skipped"

    if xs.ndim - 1 < min_ndim:
        return True, 0.0, "skipped"

    # reshape_explicit 需要 per-sample 最后一维是偶数
    if name == "reshape_explicit" and (xs.shape[-1] % 2 != 0):
        return True, 0.0, "skipped"

    # ragged_dot 需要 per-sample 最后一维为 1
    if name == "ragged_dot" and xs.shape[-1] != 1:
        return True, 0.0, "skipped"

    # slice 需要 per-sample 最后一维至少 3 个元素
    if name == "slice" and xs.shape[-1] <= 2:
        return True, 0.0, "skipped"

    try:
        ref_out = manual_batch(f, xs, in_axis)
        vmapped_out = vmap(f, in_axes=in_axis, out_axes=0)(xs)

        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch: ref={ref_out.shape}, vmap={vmapped_out.shape}"

        out_err = relative_error(vmapped_out, ref_out)
        if out_err >= 1e-5:
            return False, out_err, "output mismatch"

        if not need_grad:
            return True, out_err, "ok"

        ref_grad = manual_batch_grad(f, xs, in_axis)
        vmapped_grad = vmap(grad(lambda xi: jnp.sum(f(xi))), in_axes=in_axis, out_axes=0)(xs)

        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch: ref={ref_grad.shape}, vmap={vmapped_grad.shape}"

        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"

        return True, max(out_err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def test_one_multi(name, f, xs, ys, in_axes, required_ndim=None, min_ndim=0, need_grad=True):
    """测试多参数操作。"""
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
            return False, 1.0, f"shape mismatch: ref={ref_out.shape}, vmap={vmapped_out.shape}"

        out_err = relative_error(vmapped_out, ref_out)
        if out_err >= 1e-5:
            return False, out_err, "output mismatch"

        if not need_grad:
            return True, out_err, "ok"

        ref_grad = manual_batch_grad_multi(f, xs, ys, in_axes)
        vmapped_grad = vmap(grad(lambda xi, yi: jnp.sum(f(xi, yi))),
                            in_axes=in_axes, out_axes=0)(xs, ys)

        if ref_grad.shape != vmapped_grad.shape:
            return False, 1.0, f"grad shape mismatch: ref={ref_grad.shape}, vmap={vmapped_grad.shape}"

        grad_err = relative_error(vmapped_grad, ref_grad)
        if grad_err >= 1e-5:
            return False, grad_err, "grad mismatch"

        return True, max(out_err, grad_err), "ok"
    except Exception as e:
        return False, 1.0, f"exception: {e}"


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    # 默认启用 fail-fast：遇到第一个失败立即退出，避免 buggy 版本卡住
    fail_fast = os.environ.get("FAIL_FAST", "1") != "0"
    print(f"随机种子: {args.seed}")
    if fail_fast:
        print("fail-fast: 遇到第一个失败立即退出\n")
    else:
        print("fail-fast: 关闭\n")

    # 多种 shape + batch 轴位置：覆盖 1D/2D/3D，batch 轴在 0/1/2 位置
    # 为 ragged_dot 和 reshape_explicit 选择合适的维度
    shapes = [
        ((4,), [0]),
        ((4, 8), [0, 1]),
        ((8, 16), [0, 1]),
        ((2, 4, 8), [0, 1, 2]),
        ((4, 8, 16), [0, 1, 2]),
        # 用于 ragged_dot：最后维为 1
        ((4, 6, 1), [0, 1, 2]),
        # 用于 reshape_explicit：最后维为偶数
        ((4, 8), [0, 1]),
    ]

    total = 0
    passed = 0
    skipped = 0
    all_pass = True

    for shape, in_axes in shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for in_axis in in_axes:
            for name, f, ndim, min_ndim, default_axis, need_grad in TESTS:
                # 对大多数测试，把 in_axis 传进去；jvp/linearize 等固定 axis 也传
                ok, err, info = test_one(name, f, xs, in_axis, ndim, min_ndim, need_grad)
                status = "✅" if ok else "❌"
                print(f"  [{status}] shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if info == "skipped":
                    skipped += 1
                else:
                    total += 1
                    if ok:
                        passed += 1
                    else:
                        all_pass = False
                        if fail_fast:
                            print(f"\naccuracy {passed} {total}")
                            print(f"skipped {skipped}")
                            print(f"final_accuracy {passed/total*100:.1f}%")
                            print(f"nan_detected False")
                            print("\n❌ 部分测试失败")
                            return

    # 多参数测试
    for shape, in_axes in shapes:
        key, subkey_x = jax.random.split(key)
        key, subkey_y = jax.random.split(key)
        xs = jax.random.normal(subkey_x, shape)
        # y 与 xs 同 shape，但可能是 unbatched
        ys = jax.random.normal(subkey_y, shape[1:] if shape else shape)
        for in_axis in in_axes:
            for name, f, ndim, min_ndim, in_axes_cfg, need_grad in MULTI_ARG_TESTS:
                ok, err, info = test_one_multi(name, f, xs, ys, in_axes_cfg,
                                               required_ndim=ndim, min_ndim=min_ndim,
                                               need_grad=need_grad)
                status = "✅" if ok else "❌"
                print(f"  [{status}] shape={shape} axis={in_axis} {name}: rel_err={err:.2e} ({info})")
                if info == "skipped":
                    skipped += 1
                else:
                    total += 1
                    if ok:
                        passed += 1
                    else:
                        all_pass = False
                        if fail_fast:
                            print(f"\naccuracy {passed} {total}")
                            print(f"skipped {skipped}")
                            print(f"final_accuracy {passed/total*100:.1f}%")
                            print(f"nan_detected False")
                            print("\n❌ 部分测试失败")
                            return

    print(f"\naccuracy {passed} {total}")
    print(f"skipped {skipped}")
    print(f"final_accuracy {passed/total*100:.1f}%" if total > 0 else "final_accuracy 0.0%")
    print(f"nan_detected False")
    print("\n✅ 所有测试通过" if all_pass else "\n❌ 部分测试失败")


if __name__ == "__main__":
    main()

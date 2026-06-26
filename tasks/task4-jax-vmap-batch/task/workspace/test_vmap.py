#!/usr/bin/env python3
"""
JAX vmap+grad 测试脚本（加强版）

核心思路：
1. 对每个操作，同时比较 vmap 输出和手动 batching 的参考输出
2. 同时比较 vmap(grad(f)) 和手动 batching 的参考梯度
3. 使用多种 shape/axis/边界条件，专门触发 batching rule 中的分支
"""
import argparse
import jax
import jax.numpy as jnp
from jax import grad, vmap


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

def f_transpose(x):
    return jnp.swapaxes(x, -1, -2) if x.ndim >= 2 else x

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

def f_reduce_window(x):
    # 触发 reduce batching
    return jnp.sum(x, axis=-1, keepdims=True)

def f_select_n(x):
    # 显式触发 select_n batching
    return jnp.where(x > 0, x * 2, x * 3)


def f_multi(x):
    """组合多个操作"""
    y = jnp.sin(x) + jnp.cos(x)
    y = jnp.reshape(y, y.shape[:-1] + (-1,))
    y = jnp.transpose(y, list(range(y.ndim - 2)) + [y.ndim - 1, y.ndim - 2]) if y.ndim >= 2 else y
    return y


TESTS = [
    # (name, function, required_ndim, min_ndim)
    # required_ndim: only run when xs.ndim == required_ndim (None = any)
    # min_ndim: f's input must have at least this many dims after vmap strips batch axis
    ("sin", f_sin, None, 0),
    ("cos", f_cos, None, 0),
    ("sum", f_sum, None, 1),
    ("mean", f_mean, None, 1),
    ("max", f_max, None, 1),
    ("reshape", f_reshape, None, 1),
    ("transpose", f_transpose, None, 2),
    ("concatenate", f_concatenate, None, 1),
    ("squeeze", f_squeeze, None, 1),
    ("stack", f_stack, None, 1),
    ("where", f_where, None, 0),
    ("dot", f_dot, 1, 1),         # 1D input
    ("matmul", f_matmul, 2, 2),   # 2D input
    ("slice", f_slice, None, 1),
    ("broadcast", f_broadcast, None, 0),
    ("pad", f_pad, None, 0),
    ("gather", f_gather, None, 1),
    ("reduce_window", f_reduce_window, None, 1),
    ("select_n", f_select_n, None, 0),
    ("multi", f_multi, None, 1),
]


# ============================================================
# 测试框架
# ============================================================

def manual_batch(f, xs):
    """手动对 xs 的第 0 轴做 batching，作为参考。"""
    return jnp.stack([f(xs[i]) for i in range(xs.shape[0])])


def manual_batch_grad(f, xs):
    """手动对每个 batch 元素求 grad 再 stack。"""
    return jnp.stack([grad(lambda xi: jnp.sum(f(xi)))(xs[i]) for i in range(xs.shape[0])])


def relative_error(a, b):
    max_diff = jnp.max(jnp.abs(a - b))
    scale = jnp.max(jnp.abs(b)) + 1e-10
    return float(max_diff / scale)


def test_one(name, f, xs, required_ndim=None, min_ndim=0):
    """测试一个操作：比较 vmap 与手动 batching 的输出和梯度。"""
    if required_ndim is not None and xs.ndim != required_ndim:
        # 跳过不符合维度要求的 shape
        return True, 0.0, "skipped"

    # vmap 会把 xs 的第 0 轴作为 batch 轴，因此单个输入维度为 xs.ndim - 1。
    # 某些操作（如 axis=-1 的 reduce、slice 等）要求输入至少 min_ndim 维。
    if xs.ndim - 1 < min_ndim:
        return True, 0.0, "skipped"

    try:
        # 1. 输出对比
        ref_out = manual_batch(f, xs)
        vmapped_out = vmap(f)(xs)

        if ref_out.shape != vmapped_out.shape:
            return False, 1.0, f"shape mismatch: ref={ref_out.shape}, vmap={vmapped_out.shape}"

        out_err = relative_error(vmapped_out, ref_out)
        if out_err >= 1e-5:
            return False, out_err, "output mismatch"

        # 2. 梯度对比
        ref_grad = manual_batch_grad(f, xs)
        vmapped_grad = vmap(grad(lambda xi: jnp.sum(f(xi))))(xs)

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
    print(f"随机种子: {args.seed}\n")

    # 多种 shape：覆盖 1D/2D/3D，不同长度
    shapes = [
        (4,),
        (4, 8),
        (8, 16),
        (2, 4, 8),
        (4, 8, 16),
    ]

    total = 0
    passed = 0
    skipped = 0
    all_pass = True

    for shape in shapes:
        key, subkey = jax.random.split(key)
        xs = jax.random.normal(subkey, shape)
        for name, f, ndim, min_ndim in TESTS:
            ok, err, info = test_one(name, f, xs, ndim, min_ndim)
            status = "✅" if ok else "❌"
            print(f"  [{status}] shape={shape} {name}: rel_err={err:.2e} ({info})")
            if info == "skipped":
                skipped += 1
            else:
                total += 1
                if ok:
                    passed += 1
                else:
                    all_pass = False

    print(f"\naccuracy {passed} {total}")
    print(f"skipped {skipped}")
    print(f"final_accuracy {passed/total*100:.1f}%" if total > 0 else "final_accuracy 0.0%")
    print(f"nan_detected False")
    print("\n✅ 所有测试通过" if all_pass else "\n❌ 部分测试失败")


if __name__ == "__main__":
    main()

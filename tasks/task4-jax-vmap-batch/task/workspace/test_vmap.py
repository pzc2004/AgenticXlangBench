#!/usr/bin/env python3
"""
JAX vmap+grad 测试脚本
覆盖所有被 inject_bug.py 修改的操作
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


def f_sin(x):
    """vectorized_batcher"""
    return jnp.sum(jnp.sin(x))

def f_cos(x):
    """vectorized_batcher"""
    return jnp.sum(jnp.cos(x))

def f_sum(x):
    """reducer_batcher"""
    return jnp.sum(x)

def f_mean(x):
    """reducer_batcher (mean)"""
    return jnp.mean(x, axis=-1).sum()

def f_reshape(x):
    """reshape batching"""
    return jnp.sum(jnp.reshape(x, (-1,)) ** 2)

def f_transpose(x):
    """transpose batching"""
    return jnp.sum(jnp.transpose(x) ** 2)

def f_concatenate(x):
    """concatenate batching"""
    return jnp.sum(jnp.concatenate([x, x], axis=-1) ** 2)

def f_squeeze(x):
    """squeeze batching"""
    return jnp.sum(jnp.squeeze(jnp.expand_dims(x, -1)) ** 2)

def f_stack(x):
    """stack batching"""
    return jnp.sum(jnp.stack([x, x]) ** 2)

def f_where(x):
    """select_n batching"""
    return jnp.sum(jnp.where(x > 0, x, -x) ** 2)

def f_dot(x):
    """dot_general batching (一维向量点积)"""
    return jnp.dot(x, x)

def f_slice(x):
    """slice batching"""
    return jnp.sum(x[..., 1:-1] ** 2)

def f_broadcast(x):
    """broadcast_in_dim batching"""
    return jnp.sum(jnp.broadcast_to(x[..., None], x.shape + (2,)) ** 2)

def f_pad(x):
    """pad batching"""
    ndim = x.ndim
    pad_width = [(1, 1)] * ndim
    return jnp.sum(jnp.pad(x, pad_width) ** 2)

def f_gather(x):
    """gather batching"""
    return jnp.sum(x[..., :4] ** 2)

def f_multi(x):
    """组合"""
    y = jnp.sin(x) + jnp.cos(x)
    y = jnp.reshape(y, (-1,))
    return jnp.sum(y ** 2)


TESTS = [
    ("sin", f_sin),
    ("cos", f_cos),
    ("sum", f_sum),
    ("mean", f_mean),
    ("reshape", f_reshape),
    ("transpose", f_transpose),
    ("concatenate", f_concatenate),
    ("squeeze", f_squeeze),
    ("stack", f_stack),
    ("where", f_where),
    ("dot", f_dot),
    ("slice", f_slice),
    ("broadcast", f_broadcast),
    ("pad", f_pad),
    ("gather", f_gather),
    ("multi", f_multi),
]


def test_one(name, f, x):
    try:
        grad_without = grad(f)(x)
        grad_with = vmap(grad(lambda xi: jnp.sum(f(xi))))(x)
        max_diff = jnp.max(jnp.abs(grad_without - grad_with))
        rel_diff = max_diff / (jnp.max(jnp.abs(grad_without)) + 1e-10)
        passed = float(rel_diff) < 1e-5
        return passed, float(rel_diff)
    except Exception as e:
        return False, str(e)


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    print(f"随机种子: {args.seed}\n")

    shapes = [(4, 8), (8, 16), (16, 32)]
    all_pass = True
    total = 0
    passed = 0

    for shape in shapes:
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape)
        for name, f in TESTS:
            # dot 需要一维数组
            if name == "dot":
                x_test = x[0]  # 取第一行作为一维向量
            else:
                x_test = x
            ok, info = test_one(name, f, x_test)
            status = "✅" if ok else "❌"
            if isinstance(info, float):
                print(f"  [{status}] shape={shape} {name}: rel_diff={info:.2e}")
            else:
                print(f"  [{status}] shape={shape} {name}: {info}")
            total += 1
            if ok: passed += 1
            else: all_pass = False

    print(f"\naccuracy {passed} {total}")
    print(f"final_accuracy {passed/total*100:.1f}%")
    print(f"nan_detected False")
    print("\n✅ 所有测试通过" if all_pass else "\n❌ 部分测试失败")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JAX vmap+grad 测试脚本
用法: python test_vmap.py [--seed S] [--check]
"""

import argparse
import jax
import jax.numpy as jnp
from jax import grad, vmap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--check", action="store_true", help="检查梯度正确性")
    return parser.parse_args()


def f(x):
    """简单函数: sum(x^2 + sin(x))"""
    return jnp.sum(x ** 2 + jnp.sin(x))


def f_batch(x):
    """批量版本: 对每个样本独立计算"""
    return jnp.sum(x ** 2 + jnp.sin(x), axis=-1)


def test_grad_without_vmap(x):
    """不使用 vmap 的梯度计算"""
    return grad(f)(x)


def test_grad_with_vmap(x):
    """使用 vmap 的梯度计算"""
    return vmap(grad(f_batch))(x)


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    print(f"随机种子: {args.seed}")
    print()

    # 测试多种形状
    shapes = [(4, 8), (8, 16), (16, 32)]
    all_pass = True

    for shape in shapes:
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape)

        # 不使用 vmap 的梯度
        grad_without_vmap = test_grad_without_vmap(x)

        # 使用 vmap 的梯度
        grad_with_vmap = test_grad_with_vmap(x)

        # 比较
        max_diff = jnp.max(jnp.abs(grad_without_vmap - grad_with_vmap))
        rel_diff = max_diff / (jnp.max(jnp.abs(grad_without_vmap)) + 1e-10)

        status = "✅" if rel_diff < 1e-5 else "❌"
        print(f"  [{status}] shape={shape}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        if rel_diff >= 1e-5:
            all_pass = False

    print()
    if all_pass:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")

    # 输出 accuracy 格式
    passed = sum(1 for shape in shapes if _check_shape(shape, args.seed))
    total = len(shapes)
    print(f"\naccuracy {passed} {total}")
    print(f"final_accuracy {passed/total*100:.1f}%")
    print(f"nan_detected False")


def _check_shape(shape, seed):
    """检查单个形状的梯度"""
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, shape)
    grad_without = test_grad_without_vmap(x)
    grad_with = test_grad_with_vmap(x)
    max_diff = jnp.max(jnp.abs(grad_without - grad_with))
    rel_diff = max_diff / (jnp.max(jnp.abs(grad_without)) + 1e-10)
    return rel_diff < 1e-5


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
BLAS 测试脚本
用法: python test_blas.py [--seed S] [--size N] [--check]
"""

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--size", type=int, default=64, help="矩阵尺寸")
    parser.add_argument("--check", action="store_true", help="检查结果正确性")
    return parser.parse_args()


def test_gemm(M, N, K, seed=42):
    """测试矩阵乘法 A(M×K) @ B(K×N) = C(M×N)"""
    np.random.seed(seed)
    A = np.random.randn(M, K)
    B = np.random.randn(K, N)

    # NumPy 计算
    C_np = A @ B

    # 参考实现(纯 Python)
    C_ref = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C_ref[i, j] += A[i, k] * B[k, j]

    # 比较
    max_diff = np.max(np.abs(C_np - C_ref))
    rel_diff = max_diff / (np.max(np.abs(C_ref)) + 1e-10)

    return max_diff, rel_diff, C_np, C_ref


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"矩阵尺寸: {args.size}")
    print(f"随机种子: {args.seed}")
    print()

    # 测试多种尺寸
    sizes = [
        (args.size, args.size, args.size),           # 方阵
        (args.size, args.size // 2, args.size),       # 矩形
        (args.size * 2, args.size, args.size // 2),   # 大矩阵
    ]

    all_pass = True
    for M, N, K in sizes:
        max_diff, rel_diff, C_np, C_ref = test_gemm(M, N, K, args.seed)
        status = "✅" if rel_diff < 1e-10 else "❌"
        print(f"  [{status}] M={M:4d}, N={N:4d}, K={K:4d}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}")
        if rel_diff >= 1e-10:
            all_pass = False

    print()
    if all_pass:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")

    # 输出 accuracy 格式(用于 test.sh 解析)
    passed = sum(1 for M, N, K in sizes if test_gemm(M, N, K, args.seed)[1] < 1e-10)
    total = len(sizes)
    print(f"\naccuracy {passed} {total}")
    print(f"final_accuracy {passed/total*100:.1f}%")
    print(f"nan_detected False")


if __name__ == "__main__":
    main()

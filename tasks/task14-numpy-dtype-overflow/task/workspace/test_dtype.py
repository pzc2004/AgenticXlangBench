#!/usr/bin/env python3
"""
NumPy 混合精度 dtype 测试脚本
用法: python test_dtype.py

测试不同 dtype 组合的算术运算正确性,
重点关注 type promotion 和溢出行为。
"""

import numpy as np
import sys


def test_int8_float16_promotion():
    """测试 int8 + float16 的类型提升"""
    a = np.array([100, -50, 0, 127, -128], dtype=np.int8)
    b = np.array([0.5, 1.5, 2.5, 0.1, -0.1], dtype=np.float16)

    result = a + b

    # 检查结果 dtype
    # 正确行为: 应该提升到 float32 或至少 float16
    # Bug 行为: 结果可能是 int8 (导致溢出)
    result_dtype = result.dtype

    # 检查数值正确性
    expected = a.astype(np.float32) + b.astype(np.float32)
    max_error = np.max(np.abs(result.astype(np.float32) - expected))

    return {
        'dtype': str(result_dtype),
        'max_error': float(max_error),
        'is_correct': max_error < 0.01,
        'result_values': result.tolist(),
        'expected_values': expected.tolist(),
    }


def test_mixed_dtype_arithmetic():
    """测试多种混合 dtype 组合的算术运算"""
    test_cases = [
        # (dtype1, dtype2, values1, values2, description)
        (np.int8, np.float16, [100, -50, 0], [0.5, 1.5, 2.5], "int8 + float16"),
        (np.int8, np.float32, [100, -50, 0], [0.5, 1.5, 2.5], "int8 + float32"),
        (np.int16, np.float16, [1000, -500, 0], [0.5, 1.5, 2.5], "int16 + float16"),
        (np.int32, np.float64, [100000, -50000, 0], [0.5, 1.5, 2.5], "int32 + float64"),
        (np.uint8, np.int8, [200, 100, 0], [50, -50, 0], "uint8 + int8"),
        (np.uint16, np.int16, [40000, 20000, 0], [20000, -20000, 0], "uint16 + int16"),
    ]

    results = []
    for dtype1, dtype2, vals1, vals2, desc in test_cases:
        a = np.array(vals1, dtype=dtype1)
        b = np.array(vals2, dtype=dtype2)

        result = a + b
        expected = a.astype(np.float64) + b.astype(np.float64)

        max_error = np.max(np.abs(result.astype(np.float64) - expected))
        is_correct = max_error < 0.01

        results.append({
            'description': desc,
            'result_dtype': str(result.dtype),
            'max_error': float(max_error),
            'is_correct': is_correct,
        })

    return results


def test_overflow_behavior():
    """测试溢出行为"""
    # int8 范围: -128 到 127
    # 如果 type promotion 错误地选择 int8,这些值会溢出
    a = np.array([100, 120, 127, -120, -128], dtype=np.int8)
    b = np.array([0.1, 0.5, 0.9, -0.1, -0.5], dtype=np.float16)

    result = a + b

    # 如果结果是 int8,100 + 0.1 = 100 (截断),127 + 0.9 = 127 (截断)
    # 如果结果是 float32,100 + 0.1 = 100.1,127 + 0.9 = 127.9
    expected_float = a.astype(np.float32) + b.astype(np.float32)

    return {
        'result_dtype': str(result.dtype),
        'result_values': result.tolist(),
        'expected_values': expected_float.tolist(),
        'has_truncation': any(abs(r - e) > 0.01 for r, e in zip(result.astype(float), expected_float.astype(float))),
    }


def test_type_promotion_chain():
    """测试类型提升链(int8 → int16 → int32 → float32 → float64)"""
    a = np.array([42], dtype=np.int8)

    # 逐步提升
    step1 = a + np.array([1], dtype=np.int8)       # int8 + int8 → int8
    step2 = a + np.array([1], dtype=np.int16)       # int8 + int16 → int16
    step3 = a + np.array([1], dtype=np.int32)       # int8 + int32 → int32
    step4 = a + np.array([1.0], dtype=np.float32)   # int8 + float32 → float32
    step5 = a + np.array([1.0], dtype=np.float64)   # int8 + float64 → float64

    return {
        'int8+int8': str(step1.dtype),
        'int8+int16': str(step2.dtype),
        'int8+int32': str(step3.dtype),
        'int8+float32': str(step4.dtype),
        'int8+float64': str(step5.dtype),
    }


def main():
    print("=" * 60)
    print("NumPy 混合精度 dtype 测试")
    print("=" * 60)

    print(f"NumPy 版本: {np.__version__}")
    print()

    # 测试 1: int8 + float16 类型提升
    print(">>> 测试 1: int8 + float16 类型提升")
    result1 = test_int8_float16_promotion()
    print(f"  结果 dtype: {result1['dtype']}")
    print(f"  最大误差: {result1['max_error']:.6f}")
    print(f"  正确性: {'✅' if result1['is_correct'] else '❌'}")
    if not result1['is_correct']:
        print(f"  结果值: {result1['result_values'][:3]}")
        print(f"  期望值: {result1['expected_values'][:3]}")
    print()

    # 测试 2: 多种混合 dtype 组合
    print(">>> 测试 2: 多种混合 dtype 组合")
    results2 = test_mixed_dtype_arithmetic()
    for r in results2:
        status = "✅" if r['is_correct'] else "❌"
        print(f"  {status} {r['description']}: dtype={r['result_dtype']}, error={r['max_error']:.6f}")
    print()

    # 测试 3: 溢出行为
    print(">>> 测试 3: 溢出行为检查")
    result3 = test_overflow_behavior()
    print(f"  结果 dtype: {result3['result_dtype']}")
    print(f"  存在截断: {'❌ 是' if result3['has_truncation'] else '✅ 否'}")
    if result3['has_truncation']:
        print(f"  结果值: {result3['result_values']}")
        print(f"  期望值: {result3['expected_values']}")
    print()

    # 测试 4: 类型提升链
    print(">>> 测试 4: 类型提升链")
    result4 = test_type_promotion_chain()
    for key, dtype in result4.items():
        print(f"  {key} → {dtype}")
    print()

    # 汇总
    print("=" * 60)
    all_correct = (
        result1['is_correct'] and
        all(r['is_correct'] for r in results2) and
        not result3['has_truncation']
    )
    passed = sum(1 for r in results2 if r['is_correct']) + (1 if result1['is_correct'] else 0)
    total = len(results2) + 1

    print(f"通过: {passed}/{total}")
    print(f"整体: {'✅ 全部通过' if all_correct else '❌ 存在失败'}")

    # 输出结构化结果供 test.sh 解析
    print(f"\nRESULT_INT8_FLOAT16_DTYPE {result1['dtype']}")
    print(f"RESULT_INT8_FLOAT16_ERROR {result1['max_error']:.6f}")
    print(f"RESULT_OVERFLOW {'yes' if result3['has_truncation'] else 'no'}")
    print(f"RESULT_PASSED {passed}")
    print(f"RESULT_TOTAL {total}")


if __name__ == "__main__":
    main()

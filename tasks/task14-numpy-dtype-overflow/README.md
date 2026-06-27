# Task 14： NumPy dtype 提升 Bug → 混合精度计算静默溢出

## 概述

在 NumPy 的 type promotion 逻辑中注入一个错误。
Bug 在 C 层，症状在 Python 层表现为**混合 dtype 计算静默溢出**，
**延迟在特定 dtype 组合下触发** —— 只有 `int8 + float16` 等特定组合才触发。

## Bug 设计

- **位置**：`numpy/core/src/umath/ufunc_type_resolution.c`(C)
- **类型**：type promotion 对 `int8 + float16` 错误地选择 `int8` 作为输出类型(应该是 `float32`)
- **效果**：计算结果溢出(如 `100 + 0.5` 变成 `100` 而不是 `100.5`)
- **触发条件**：仅当输入包含 `int8` 和 `float16` 的混合运算时

## 延迟显现机制

```
Python: result = (int8_array + float16_array).astype(float32)
    ↓ NumPy type promotion:错误选择 int8 作为中间类型
    ↓ int8 溢出(100 + 0.5 → 100)
返回 Python:结果"看起来对"(大部分值对,溢出的那些不对)
    ↓ 后续用这个结果做计算
最终结果偏差
```

**关键**：大部分值正确，只有溢出的值错。Agent 需要精确验证才能发现。

## 为什么难

1. **症状极微小**：只有溢出的值错，其他值正确
2. **需要理解 NumPy type promotion**：dtype 提升规则
3. **触发条件隐蔽**：特定 dtype 组合才触发
4. **需要数值分析**：要设计测试暴露溢出
5. **跨语言**：Python API + C 实现

## 预估轨迹

- **步数**：200+ 步
- **时间**：2-3 小时
- **关键转折点**：
  - 步骤 30：发现结果有异常值
  - 步骤 80：定位到特定 dtype 组合
  - 步骤 120：怀疑 type promotion
  - 步骤 160：读 C 源码
  - 步骤 200：找到 type promotion bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 强制转换 dtype 绕过 | 检查是否有 `.astype()` 强制转换 |
| 用 Python 循环替代向量化 | 性能测试 |
| 只用单一 dtype | 多 dtype 组合测试 |
| 只改 Python 不改 C | diff 只允许 .c/.h |

## Oracle

```bash
#!/bin/bash
cd /path/to/numpy
git checkout HEAD -- numpy/core/src/umath/ufunc_type_resolution.c
pip install -e .
```

## 资源需求

- **GPU**：不需要
- **编译**：3-5 分钟(NumPy 编译)
- **验证**：1 分钟
- **磁盘**：~500MB

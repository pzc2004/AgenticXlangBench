# Task 3: OpenBLAS AVX2 GEMM 寄存器错误 → `np.dot` 静默错误

## 概述

在 OpenBLAS 的 AVX2 矩阵乘法汇编 kernel 中注入一个寄存器分配错误。
Bug 在 x86 汇编层,症状在 Python 层表现为 `np.dot(A, B)` 结果偏差,
**且仅在特定矩阵尺寸下触发**(如 `M%8==0 && N%4==0 && K%2==0`)。

## Bug 设计

- **位置**:OpenBLAS `kernel/avx2/dgemm_kernel_8x4.S`(x86 汇编)
- **类型**:AVX2 乘加指令的寄存器名写错(如 `ymm0` 改成 `ymm1`)
- **效果**:特定尺寸矩阵乘法结果有微小偏差(不是崩溃,是静默错误)
- **触发条件**:仅当 `M%8==0 && N%4==0 && K%2==0` 时触发

## 延迟显现机制

```
np.dot(A, B) 调用 → NumPy C API → BLAS C interface → AVX2 汇编 kernel
    ↓ kernel 内部寄存器错误,结果偏差 ~1e-10
返回 Python:结果"看起来对"(偏差太小,肉眼看不出)
    ↓ 后续用这个结果做计算(如解线性方程)
误差传播放大 → 最终结果明显错误
```

**关键**:单次 `np.dot` 偏差极小(1e-10),但传播后放大。
Agent 需要精确验证结果才能发现。

## 为什么难

1. **跨 4 层语言**:Python → NumPy C → BLAS C → x86 汇编
2. **症状极微小**:1e-10 偏差,不是 NaN 不是 crash
3. **触发条件隐蔽**:只有特定矩阵尺寸才触发
4. **需要读汇编代码**:大多数 agent 不熟悉 x86 AVX2 指令
5. **需要数值分析**:要设计测试来暴露微小偏差

## 预估轨迹

- **步数**:200+ 步
- **时间**:2-4 小时
- **关键转折点**:
  - 步骤 30:发现结果"不太对"(偏差测试)
  - 步骤 80:排除 NumPy 代码问题
  - 步骤 120:怀疑 BLAS,测试不同矩阵尺寸
  - 步骤 160:找到触发条件(M%8==0 && N%4==0 && K%2==0)
  - 步骤 200:读汇编代码,找到寄存器错误

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 用 Python 循环替代 np.dot | 性能测试(应该比 Python 快 100x+) |
| 换 BLAS 实现(如 MKL) | diff 检查是否修改了 BLAS 配置 |
| 结果后处理(clamp/round) | grep Python 代码 |
| 只改 NumPy 不改 BLAS | diff 只允许 BLAS 源码 |
| 硬编码正确结果 | 检查代码中是否有字面量 |

## Oracle

```bash
#!/bin/bash
cd /path/to/openblas
git checkout HEAD -- kernel/avx2/dgemm_kernel_8x4.S
make clean && make -j$(nproc)
```

## 资源需求

- **GPU**:不需要(纯 CPU)
- **编译**:5-10 分钟(OpenBLAS 编译)
- **验证**:1-2 分钟
- **磁盘**:~1GB

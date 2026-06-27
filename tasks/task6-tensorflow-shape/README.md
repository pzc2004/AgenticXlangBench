# Task 6： TensorFlow Custom Op Shape Inference 错误 → 第 50 个 Op 报 Shape Mismatch

## 概述

在 TensorFlow 的一个自定义 C++ op 的 shape inference 函数中注入错误。
Bug 在 C++ 层，症状在 Python 层表现为模型 forward pass 在**第 50 个 op 处**
报 `InvalidArgumentError: Shape mismatch`，
**延迟 ~47 个 op 才显现** —— 因为 TF 的 shape 传播有时不严格检查。

## Bug 设计

- **位置**：`tensorflow/core/kernels/my_custom_op.cc`(C++)
- **类型**：shape inference 函数算错输出 shape(少了一维)
- **效果**：输出 tensor 的 shape metadata 错误，但实际数据可能"碰巧"正确
- **触发条件**：当 op 的输入 shape 满足特定条件时(如 rank >= 3)

## 延迟显现机制

```
Custom op(第 3 个 op):shape inference 输出错误 shape
    ↓ TF 内部记录的 shape metadata 错了,但数据没检查
第 4-49 个 op:有些 op 不检查 shape,直接用数据 → "正常"
    ↓ 第 50 个 op:严格检查 shape
Shape mismatch! 但 stack trace 指向第 50 个 op,不是第 3 个
```

**关键**：错误发生在第 3 个 op，但症状在第 50 个 op。
Stack trace 指向错误位置，误导 agent。

## 为什么难

1. **Stack trace 误导**：指向第 50 个 op，不是真正出错的第 3 个
2. **需要理解 TF 内部**：shape inference 机制、graph compilation
3. **跨语言**：Python 模型定义 + C++ op 实现
4. **TF 代码量大**：从错误 op 追溯到真正出错的 op 需要读大量代码
5. **shape 传播复杂**：TF 的 shape 传播不是每步都检查

## 预估轨迹

- **步数**：200+ 步
- **时间**：2-4 小时
- **关键转折点**：
  - 步骤 30：看到 Shape mismatch error
  - 步骤 80：检查第 50 个 op(看起来没问题)
  - 步骤 120：开始检查每个 op 的 shape
  - 步骤 160：发现第 3 个 op 的 shape metadata 异常
  - 步骤 200：找到 C++ shape inference 函数的 bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 删掉第 50 个 op 的 shape 检查 | diff 检查是否删了 assert |
| 改模型结构绕过 | diff 检查模型定义 |
| 用 eager mode 替代 graph mode | 检查运行模式 |
| 只改 Python 不改 C++ | diff 只允许 .cc/.h |
| hardcode shape | grep shape 硬编码 |

## Oracle

```bash
#!/bin/bash
cd /path/to/tensorflow
git checkout HEAD -- tensorflow/core/kernels/my_custom_op.cc
bazel build //tensorflow/core/kernels:my_custom_op
```

## 资源需求

- **GPU**：可选(TF CPU 模式也能验证)
- **编译**：10-30 分钟(TF 增量编译)
- **验证**：1-2 分钟
- **磁盘**：~10GB(TF 源码 + build)

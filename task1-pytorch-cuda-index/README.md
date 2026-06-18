# Task 1: PyTorch CUDA Kernel Off-by-One → 训练 NaN

## 概述

在 PyTorch 的 CUDA element-wise kernel 中注入一个 off-by-one 错误。
Bug 在 CUDA 层触发,但症状在 Python 层表现为训练 loss 变为 NaN,
且**延迟 ~100 个 iteration 才显现** —— 因为越界写入的累积效应需要多次迭代才破坏到关键内存。

## Bug 设计

- **位置**:`aten/src/ATen/native/cuda/UnaryElementwiseKernel.cu`(或类似)
- **类型**:off-by-one(`if (idx < n)` 改成 `if (idx <= n)`)
- **效果**:越界写入 1 个元素,破坏相邻内存
- **触发条件**:每次 CUDA kernel 调用都触发,但只在特定 tensor layout 下写入关键位置

## 延迟显现机制

```
CUDA kernel 越界写 → 破坏相邻 tensor 的 1 个元素
    ↓ 该 tensor 参与后续计算,但值偏差很小
autograd backward 用这个 tensor 计算梯度 → 梯度略微偏
    ↓ 累积 100 个 iteration
梯度累积偏差 → 参数偏移 → 某层输出爆炸 → loss NaN
```

**关键**:前 99 个 iteration 都"看起来正常",第 100 个才 NaN。
Agent 不能一眼看到 bug。

## 为什么难

1. **跨语言**:症状在 Python,bug 在 CUDA,需要读两种语言
2. **跨抽象层**:Python → PyTorch dispatch → ATen → CUDA kernel(4 层)
3. **症状模糊**:NaN 可能是 100 种原因(learning rate / 数据 / 模型 / bug)
4. **无 stack trace**:NaN 不指向 CUDA kernel,指向 Python 层的 loss 计算
5. **需要理解 PyTorch 内部**:autograd、CUDA dispatch、tensor memory layout

## 预估轨迹

- **步数**:300+ 步
- **时间**:2-4 小时(含编译等待)
- **关键转折点**:
  - 步骤 50:发现 NaN,开始排查
  - 步骤 100:定位到梯度异常
  - 步骤 150:追溯到哪个 op 产生异常梯度
  - 步骤 200:找到对应的 CUDA kernel
  - 步骤 250:读懂 kernel 逻辑
  - 步骤 300:找到 off-by-one

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `isnan` / `nan_to_num` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 5x+) |
| 改 batch_size 避开 | 多 batch_size 测试(16/32/64/128) |
| 注释掉 kernel | diff 检查 + 功能测试 |
| 只改 Python 不改 CUDA | diff 只允许 .cu/.cpp/.h |

## Oracle

```bash
#!/bin/bash
# 撤销 off-by-one:恢复原始代码
cd /path/to/pytorch
git checkout HEAD -- aten/src/ATen/native/cuda/UnaryElementwiseKernel.cu
# 增量重编
python setup.py develop
```

## 资源需求

- **GPU**:1× RTX A6000(用 GPU 1)
- **首次编译**:30-60 分钟(全量 PyTorch)
- **增量编译**:5-15 分钟(只改 .cu 文件)
- **训练验证**:1-3 分钟(100 iteration)
- **磁盘**:~20GB(PyTorch 源码 + build)

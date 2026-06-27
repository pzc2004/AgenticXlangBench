# Task 2： PyTorch Autograd 梯度符号错误 → 精度缓慢下降

## 概述

在 PyTorch 的 autograd `backward()` 实现中注入一个符号错误(正负号反了)。
Bug 在 C++ 层，但症状在 Python 层表现为模型 accuracy 从 95% 缓慢降到 80%，
**延迟 ~10 个 epoch 才明显** —— 因为梯度方向错误的累积效应很慢。

## Bug 设计

- **位置**：`torch/csrc/autograd/FunctionsManual.cpp`(某个 op 的 backward)
- **类型**：梯度公式符号错误(如 `return grad * x` 改成 `return -grad * x`)
- **效果**：某个 op 的梯度方向反了，但幅度对
- **触发条件**：每次 backward 都触发，但单次误差小，需要累积

## 延迟显现机制

```
backward() 梯度符号反 → 每 iteration 参数往错误方向走一小步
    ↓ 100 个 iteration 后,参数偏离正确值但仍"接近"
accuracy 从 95% 缓慢降到 90% → 不明显,可能是正常波动
    ↓ 1000 个 iteration 后
accuracy 降到 80% → 明显异常,但原因不明
```

**关键**：前 500 步几乎看不出问题。Agent 需要对比"有 bug"和"无 bug"的训练曲线才能发现差异。

## 为什么难

1. **症状极慢**：10 epoch 才明显，不是即时崩溃
2. **症状很像正常训练问题**：accuracy 下降可能是数据/模型/lr，不一定是 bug
3. **需要理解 autograd**：要读 C++ 的 backward 实现
4. **需要定量分析**：不是"NaN"(一眼看出)，是"95%→80%"(需要仔细对比)
5. **跨语言**：Python 训练脚本 + C++ autograd 实现

## 预估轨迹

- **步数**：250+ 步
- **时间**：3-5 小时(含多次训练验证)
- **关键转折点**：
  - 步骤 50：注意到 accuracy 异常下降
  - 步骤 100：排除数据/lr/模型问题
  - 步骤 150：对比正常训练和异常训练的梯度
  - 步骤 200：定位到哪个 op 的梯度异常
  - 步骤 250：找到 C++ backward 实现中的符号错误

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 改 learning rate 掩盖 | diff 检查训练参数 |
| 加 gradient clipping | grep `clip_grad` |
| 换 optimizer | diff 检查 optimizer 选择 |
| 加 learning rate scheduler | diff 检查 lr 策略 |
| 只改 Python 不改 C++ | diff 只允许 .cpp/.h |
| 改模型结构绕过 | diff 检查模型定义 |

## Oracle

```bash
#!/bin/bash
cd /path/to/pytorch
git checkout HEAD -- torch/csrc/autograd/FunctionsManual.cpp
python setup.py develop
```

## 资源需求

- **GPU**:1× RTX A6000
- **首次编译**：30-60 分钟
- **增量编译**：5-15 分钟
- **训练验证**：10-30 分钟(需要跑多个 epoch)
- **磁盘**：~20GB

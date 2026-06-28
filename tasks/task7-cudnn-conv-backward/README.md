# Task 7： cuDNN Backward Convolution 特定尺寸梯度错误 → 特定分辨率训练效果差
> ⚠️ **状态：骨架待完善**
>
> 本任务目前只有 Dockerfile / run.sh / instruction.md 骨架，`task/solution/oracle.sh` 未补齐或 `tests/test.sh` 未验证，暂不能端到端运行。可交付任务见顶层 README.md 的「任务总览」。


## 概述

在 cuDNN 的 backward convolution 实现中注入一个特定 input/kernel size 下的梯度错误。
Bug 在 CUDA 层，症状在 Python 层表现为**特定分辨率图像训练效果差**，
**延迟 ~几百个 iteration 才明显** —— 因为梯度误差小，累积才显著。

## Bug 设计

- **位置**：cuDNN backward convolution 内部(CUDA)
  - 注：cuDNN 闭源，可用开源 MIOpen(AMD)或自写 conv kernel 替代
- **类型**：当 `input_h % stride != 0 && kernel_size % 2 == 0` 时，梯度计算有微小错误
- **效果**：特定尺寸下梯度偏差 ~1%，累积后模型收敛到更差的 local minimum
- **触发条件**：仅特定 input_h / stride / kernel_size 组合

## 延迟显现机制

```
conv forward:正常(前向传播不受影响)
conv backward:梯度在特定尺寸下有微小错误
    ↓ 每 iteration 梯度错 1%
累积 500 iteration:参数偏离正确值
    ↓ 模型收敛到更差的 local minimum
accuracy 比正常低 5-10%(不是 NaN,不是 crash)
```

**关键**：前向传播正确，只有 backward 梯度错。
症状是"效果差"，不是"报错"，极难定位。

## 为什么难

1. **症状最模糊**：不是 NaN 不是 crash，是"效果差 5%"
2. **需要对比实验**：要跟 CPU 版本或正确版本对比才能发现差异
3. **触发条件复杂**：特定 input_h / stride / kernel_size 组合
4. **cuDNN 闭源**：agent 读不到 cuDNN 源码(用 MIOpen 替代时可读)
5. **需要理解卷积的前向/backward**：数学 + 实现

## 预估轨迹

- **步数**：300+ 步
- **时间**：3-5 小时
- **关键转折点**：
  - 步骤 50：发现特定分辨率效果差
  - 步骤 100：对比不同分辨率，找到触发条件
  - 步骤 150：怀疑 conv 层
  - 步骤 200：对比 CPU vs GPU 结果，确认 GPU backward 有问题
  - 步骤 300：定位到 cuDNN backward convolution

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 只用特定分辨率避开 | 多分辨率测试 |
| 用 CPU 推理替代 GPU | 性能测试 |
| 改模型不用 conv | diff 检查模型结构 |
| 后处理输出(clamp/normalize) | grep 输出处理 |
| 只改 Python 不改 CUDA | diff 只允许 CUDA 源码 |

## Oracle

```bash
#!/bin/bash
cd /path/to/cudnn  # 或 MIOpen
git checkout HEAD -- src/convolution_backward.cu
make -j$(nproc)
```

## 资源需求

- **GPU**：1× RTX A6000(必须)
- **编译**：10-30 分钟
- **训练验证**：30-60 分钟(需要多个分辨率 × 多个 epoch)
- **磁盘**：~5GB

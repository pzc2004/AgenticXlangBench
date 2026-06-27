# Task 13： OpenCV CUDA Resize Bug → 图像处理管线结果偏差

## 概述

在 OpenCV 的 CUDA resize kernel 中注入一个插值计算错误。
Bug 在 CUDA 层，症状在 Python 层表现为**图像缩放后像素值偏差**，
**延迟传播到整个图像处理管线** —— 缩放后的图像被后续处理步骤使用，偏差累积。

## Bug 设计

- **位置**：`modules/cudaimgproc/src/cuda/resize.cu`(CUDA)
- **类型**：双线性插值的权重计算错误(如 `1.0 - t` 改成 `t`)
- **效果**：缩放后的图像像素值有微小偏差(~1-2/255)
- **触发条件**：仅当使用特定插值方法(INTER_LINEAR) + 特定缩放比例时

## 延迟显现机制

```
Python: cv2.cuda.resize(img, dsize, interpolation=INTER_LINEAR)
    ↓ CUDA kernel 插值权重错误
返回 Python:图像"看起来对"(偏差 1-2 像素值,肉眼难辨)
    ↓ 后续处理:边缘检测 / 特征提取 / ML 推理
最终结果偏差(如检测框偏移 / 分类置信度下降)
```

**关键**：单次 resize 偏差极小，但传播到下游处理后放大。

## 为什么难

1. **跨 3 层**：Python → OpenCV C++ → CUDA kernel
2. **症状微小**：1-2/255 像素值偏差，肉眼看不出
3. **需要定量验证**：要写脚本对比像素值
4. **传播路径长**：resize → 后续处理 → 最终结果
5. **OpenCV 代码量大**：CUDA kernel 在深层目录

## 预估轨迹

- **步数**：200+ 步
- **时间**：2-3 小时
- **关键转折点**：
  - 步骤 30：发现最终结果偏差
  - 步骤 80：追溯到 resize 步骤
  - 步骤 120：对比 CPU vs GPU resize 结果
  - 步骤 160：定位到 CUDA kernel
  - 步骤 200：找到插值权重错误

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 用 CPU resize 替代 | 性能测试(GPU 应该快 10x+) |
| 用 PIL/Pillow 替代 OpenCV | 检查依赖没有变化 |
| 后处理校正像素值 | grep 像素处理代码 |
| 只改 Python 不改 CUDA | diff 只允许 .cu/.cpp |

## Oracle

```bash
#!/bin/bash
cd /path/to/opencv
git checkout HEAD -- modules/cudaimgproc/src/cuda/resize.cu
cmake --build build -j$(nproc)
```

## 资源需求

- **GPU**:1× RTX A6000
- **编译**：10-20 分钟(OpenCV 增量编译)
- **验证**：1-2 分钟
- **磁盘**：~5GB

# 任务:修复 PyTorch LayerNorm CUDA 内核导致的训练 NaN

## 背景

我们从源码编译了一个修改过的 PyTorch 2.5.0(源码在 `/build/pytorch/`)。
在使用它训练模型时,loss 在约 90 步后变成 NaN。

## Bug 现象

运行训练脚本:

```bash
cd /workspace
python train.py --steps 200 --seed 42 --device cuda
```

预期:loss 正常下降。
实际:约 90 步后 loss 变成 NaN。

注意:用 `--device cpu` 时**不会出现 NaN**。

## 你的任务

1. **定位 bug**:找到 PyTorch 源码中导致 NaN 的根因
2. **修复 bug**:修改源码,使训练不再出现 NaN
3. **验证修复**:
   - `python train.py --steps 200 --seed 42 --device cuda` 不再 NaN
   - 其他 seed(1-10)和 batch_size(16/32/64/128)也不 NaN
   - GPU 性能未退化(应比 CPU 快 3x 以上)

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cu` / `.cpp` / `.cc` / `.cuh` / `.h` / `.c`),不允许修改 Python 文件
- **修改行数不超过 10 行**
- **不允许**:
  - 在训练脚本中加 NaN 检查 / 梯度裁剪 / clamp
  - 强制使用 CPU
  - 加 try/catch
  - 修改训练参数
  - monkey-patch 任何函数

## 文件说明

- `/workspace/train.py` — 训练脚本(使用 PyTorch 原生 LayerNorm)
- `/build/pytorch/` — PyTorch 源码树(**bug 在这里**)
  - `aten/src/ATen/native/cuda/layer_norm_kernel.cu` — LayerNorm CUDA 内核

## 运行环境

- PyTorch 2.5.0(从源码编译,已安装)
- CUDA 12.6 + cuDNN 9
- 修改 `.cu` 文件后,需要重新编译: `cd /build/pytorch && python setup.py develop`

## 验收标准

运行 `tests/test.sh`,分数 ≥ 0.6 视为通过。

# 任务:修复 PyTorch CUDA 训练 NaN 问题

## 背景

我们从源码编译了一个修改过的 PyTorch 2.5.0(源码在 `/build/pytorch/`)。
在使用它训练一个使用多种 op 的模型时,loss 在约 90 步后变成 NaN。

## Bug 现象

运行训练脚本:

```bash
cd /workspace
python train.py --steps 200 --seed 42 --device cuda
```

预期:loss 正常下降。
实际:约 90 步后 loss 变成 NaN。

注意:用 `--device cpu` 时**不会出现 NaN**。

## 已知信息

- PyTorch 源码中**有 20 个 CUDA kernel 文件被修改过**
- 其中**只有 1 个修改是真正的 bug**,导致训练 NaN
- 其他 19 个修改是**诱饵**(看起来像 bug,但不导致 NaN)
- 你的任务是找到真正的 bug 并修复它

## 你的任务

1. **定位 bug**:找到 PyTorch 源码中导致 NaN 的真正根因
2. **修复 bug**:只修改真正有问题的文件
3. **验证修复**:
   - `python train.py --steps 200 --seed 42 --device cuda` 不再 NaN
   - 其他 seed(1-10)和 batch_size(8/16/32/64)也不 NaN
   - GPU 性能未退化(应比 CPU 快 2x 以上)

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cu` / `.cpp` / `.cc` / `.cuh` / `.h` / `.c`),不允许修改 Python 文件
- **修改行数不超过 10 行**(只修复真正的 bug,不要动诱饵)
- **不允许**:
  - 在训练脚本中加 NaN 检查 / 梯度裁剪 / clamp
  - 强制使用 CPU
  - 加 try/catch
  - 修改训练参数
  - monkey-patch 任何函数

## 文件说明

- `/workspace/train.py` — 训练脚本(使用 Conv2d + BatchNorm + LayerNorm + ReLU + GELU + SiLU + Linear + Dropout + MaxPool 等多种 op)
- `/build/pytorch/` — PyTorch 源码树(**有 20 个 CUDA kernel 被修改**)

## 运行环境

- PyTorch 2.5.0(从源码编译,已安装)
- CUDA 12.6 + cuDNN 9
- 修改 `.cu` 文件后,需要重新编译: `cd /build/pytorch && python setup.py develop`

## 验收标准

运行 `tests/test.sh`,分数 ≥ 0.6 视为通过。

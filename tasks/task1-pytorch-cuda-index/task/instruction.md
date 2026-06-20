# 任务:修复 PyTorch LayerNorm CUDA 内核导致的训练 NaN

## 背景

我们使用了一个从源码编译的 PyTorch 2.5.0(源码在 `/build/pytorch/`)。
在训练一个简单模型时,loss 在约 90 步后变成 NaN。

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

- PyTorch 是从源码编译的,源码在 `/build/pytorch/`
- 模型使用了 `nn.LayerNorm`(CUDA 实现)
- 每 20 步会注入一个零方差 batch(所有特征相同),这会触发 NaN
- **你需要找到并修复 PyTorch 源码中导致 NaN 的 CUDA kernel bug**

## 你的任务

1. **理解 bug**:分析为什么零方差输入会导致 NaN
2. **定位 bug**:在 `/build/pytorch/` 的 CUDA 源码中找到有问题的代码
3. **修复 bug**:修改 CUDA 源码(只允许修改 `.cu` / `.cpp` / `.h` 文件)
4. **重新编译**:修复后需要重新编译 PyTorch
   ```bash
   cd /build/pytorch/build
   ninja -j32 lib/libtorch_cuda.so
   cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
   ```
5. **验证修复**:运行测试确认 NaN 消失
   ```bash
   bash /workspace/task/tests/test.sh
   ```

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cu` / `.cpp` / `.h`),不允许修改 Python 文件
- **修改行数不超过 10 行**
- **不允许**:
  - 在训练脚本中加 NaN 检查 / 梯度裁剪 / clamp
  - 强制使用 CPU
  - 加 try/catch
  - 修改训练参数
  - monkey-patch LayerNorm

## 文件说明

- `/workspace/train.py` — 训练脚本(用 `nn.LayerNorm`)
- `/workspace/task/tests/test.sh` — 测试脚本(跑完后输出分数)
- `/build/pytorch/` — PyTorch 源码(**bug 在这里**)
  - `aten/src/ATen/native/cuda/layer_norm_kernel.cu` — LayerNorm CUDA kernel

## 运行环境

- PyTorch 2.5.0(从源码编译)
- CUDA 12.8 + cuDNN 9
- 编译工具:nvcc, gcc, ninja, cmake

## 验收标准

运行 `/workspace/task/tests/test.sh`,分数 ≥ 0.6 视为通过。

# 任务:修复自定义 CUDA LayerNorm 内核导致的训练 NaN

## 背景

我们实现了一个自定义的 CUDA LayerNorm 内核(`layernorm_cuda/layernorm_cuda_kernel.cu`),用于加速训练。
但在使用它训练模型时,loss 在约 90 步后变成 NaN。

## Bug 现象

运行训练脚本:

```bash
cd workspace
python train.py --steps 200 --seed 42 --device cuda
```

预期:loss 正常下降。
实际:约 90 步后 loss 变成 NaN。

注意:用 `--device cpu` 时**不会出现 NaN**(CPU 使用 PyTorch 原生 LayerNorm,没有 bug)。

## 你的任务

1. **定位 bug**:找到 `layernorm_cuda/layernorm_cuda_kernel.cu` 中导致 NaN 的根因
2. **修复 bug**:修改 CUDA 内核代码
3. **验证修复**:
   - `python train.py --steps 200 --seed 42 --device cuda` 不再 NaN
   - 其他 seed(1-10)和 batch_size(16/32/64/128)也不 NaN
   - GPU 性能未退化(应比 CPU 快 3x 以上)

## 约束条件

- **只允许修改 `layernorm_cuda/` 目录下的文件**(`.cu` / `.cpp`),不允许修改 `train.py`
- **修改行数不超过 10 行**
- **不允许**:
  - 在训练脚本中加 NaN 检查 / 梯度裁剪 / clamp
  - 强制使用 CPU
  - 加 try/catch
  - 修改训练参数
  - 换用 PyTorch 原生 LayerNorm(`nn.LayerNorm`)

## 文件说明

- `workspace/train.py` — 训练脚本(使用自定义 CudaLayerNorm)
- `workspace/layernorm_cuda/` — 自定义 CUDA LayerNorm 扩展
  - `layernorm_cuda_kernel.cu` — CUDA 内核实现(**bug 在这里**)
  - `layernorm_cuda.cpp` — C++ 绑定
  - `setup.py` — 编译脚本
  - `__init__.py` — Python 接口

## 运行环境

- 已安装 PyTorch 2.5.0 + CUDA 12.6
- CUDA 扩展会在首次运行时自动编译
- 修改 `.cu` 文件后,下次运行会自动重新编译

## 验收标准

运行 `tests/test.sh`,分数 ≥ 0.6 视为通过。

# 任务:修复 PyTorch CUDA 训练精度差问题

## 背景

我们使用了一个从源码编译的 PyTorch 2.5.0(源码在 `/build/pytorch/`)。
在训练一个图像分类模型时,CUDA 模式的训练精度远低于 CPU 模式。

## Bug 现象

运行训练脚本:

```bash
cd /workspace
python train.py --steps 100 --seed 42 --device cuda --eval_fixed_data
```

预期:训练后 accuracy 应达到 50% 以上(固定数据集上)。
实际:CUDA 训练后 accuracy 仅 ~10-30%,远低于预期。

注意:用 `--device cpu` 时 **accuracy 正常**(50%+)。问题仅出现在 CUDA 上。

## 已知信息

- PyTorch 是从源码编译的,源码在 `/build/pytorch/`
- **CUDA 训练 accuracy 远低于 CPU,说明 CUDA 路径存在 bug**
- 你需要找到并修复 PyTorch 源码中导致 CUDA 训练精度差的 kernel bug

## 你的任务

1. **理解 bug**:分析为什么 CUDA 训练 accuracy 远低于 CPU
2. **定位 bug**:在 `/build/pytorch/` 的 CUDA 源码中找到有问题的代码
3. **修复 bug**:修改 CUDA 源码(只允许修改 `.cu` / `.cpp` / `.h` 文件)
4. **重新编译**:修复后需要重新编译 PyTorch
   ```bash
   cd /build/pytorch/build
   ninja -j32 lib/libtorch_cuda.so
   cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
   ```
5. **验证修复**:运行评分命令查看当前分数
   ```bash
   grade
   ```
   输出形如 `score=0.83`(0-1 的总分)。
6. **检查分数**:**分数 ≥ 0.6 才算通过。如果分数 < 0.6,说明修复不正确,请继续排查。**
   评分只返回总分,不会告诉你哪一项未通过——你需要自己通过对比 CPU/CUDA 输出来定位问题。

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cu` / `.cpp` / `.h`),不允许修改 Python 文件
- **禁止访问外部网络**:不允许从 GitHub、PyPI 或任何外部网站下载文件、查看源码或获取参考实现。所有分析必须基于容器内的本地源码。
- **不允许**:
  - 在训练脚本中加 NaN 检查 / 梯度裁剪 / clamp
  - 强制使用 CPU
  - 加 try/catch
  - 修改训练参数
  - 修改 model.py 或 train.py
  - 从外部下载参考代码进行对比

## 文件说明

- `/workspace/train.py` — 训练脚本(支持 `--eval_fixed_data` 用固定数据训练)
- `/workspace/model.py` — 模型定义(**只读,不可修改**)
- `grade` — 评分命令(运行后输出 0-1 总分)
- `/build/pytorch/` — PyTorch 源码

## 运行环境

- PyTorch 2.5.0(从源码编译)
- CUDA 12.8 + cuDNN 9
- 编译工具:nvcc, gcc, ninja, cmake

## 验收标准

运行 `grade`,分数 ≥ 0.6 视为通过。
**如果分数 < 0.6,说明修复不正确,请继续排查 CUDA 源码。**

# 任务:修复 PyTorch 训练精度下降问题

## 背景

我们使用了一个从源码编译的 PyTorch 2.5.0(源码在 `/build/pytorch/`)。
在训练一个图像分类模型时,accuracy 从正常水平(95%)缓慢下降到 80%。

## Bug 现象

运行训练脚本:

```bash
cd /workspace
python train.py --epochs 10 --seed 42 --device cuda
```

预期:训练后 accuracy 应达到 95% 以上。
实际:训练后 accuracy 仅 ~80%,且随着训练进行持续下降。

注意:用 `--device cpu` 时 **accuracy 正常**(95%+)。问题仅出现在 CUDA 上。

## 已知信息

- PyTorch 是从源码编译的,源码在 `/build/pytorch/`
- **你需要找到并修复 PyTorch 源码中导致 CUDA 训练精度下降的 bug**

## 你的任务

1. **理解 bug**:分析为什么 CUDA 训练 accuracy 远低于 CPU
2. **定位 bug**:在 `/build/pytorch/` 的 C++ 源码中找到有问题的代码
3. **修复 bug**:修改 C++ 源码(只允许修改 `.cpp` / `.h` 文件)
4. **重新编译**:修复后需要重新编译 PyTorch
   ```bash
   cd /build/pytorch/build
   ninja -j32 lib/libtorch_cuda.so
   cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
   ```
5. **验证修复**:运行测试确认修复成功
   ```bash
   bash /task/tests/test.sh
   ```
6. **检查分数**:测试会输出 0-1 的分数。**分数 ≥ 0.6 才算通过。如果分数 < 0.6,说明修复不正确,请继续排查。**

## 约束条件

- **只允许修改 C++/CUDA 文件**(`.cpp` / `.h`),不允许修改 Python 文件
- **不允许**:
  - 修改 learning rate 或 optimizer
  - 加 gradient clipping
  - 换 optimizer 或加 lr scheduler
  - 修改模型结构
  - 修改 train.py 或 model.py

## 文件说明

- `/workspace/train.py` — 训练脚本
- `/workspace/model.py` — 模型定义
- `/task/tests/test.sh` — 测试脚本(跑完后输出 0-1 分数)
- `/build/pytorch/` — PyTorch 源码

## 运行环境

- PyTorch 2.5.0(从源码编译)
- CUDA 12.8 + cuDNN 9
- 编译工具:nvcc, gcc, ninja, cmake

## 验收标准

运行 `/task/tests/test.sh`,分数 ≥ 0.6 视为通过。
**如果分数 < 0.6,说明修复不正确,请继续排查 PyTorch 源码。**

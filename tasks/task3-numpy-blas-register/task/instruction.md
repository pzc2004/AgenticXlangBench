# 任务：修复 NumPy 矩阵乘法结果偏差

## 背景

我们使用了一个从源码编译的 OpenBLAS 库，NumPy 的 `np.dot` 底层调用这个库。
在进行矩阵乘法时，结果存在微小偏差。

## Bug 现象

运行测试脚本：

```bash
cd /workspace
python test_blas.py --size 64 --check
```

预期：所有测试通过(相对误差 < 1e-10)。
实际：部分测试失败(相对误差 ~1e-10)。

注意：偏差很小，但会随计算传播放大。

## 已知信息

- OpenBLAS 是从源码编译的，源码在 `/build/OpenBLAS/`
- **你需要找到并修复 OpenBLAS 源码中导致矩阵乘法结果偏差的 bug**

## 你的任务

1. **理解 bug**：分析为什么矩阵乘法结果有偏差
2. **定位 bug**：在 `/build/OpenBLAS/` 的汇编源码中找到有问题的代码
3. **修复 bug**：修改汇编源码(只允许修改 `.S` / `.s` / `.h` 文件)
4. **重新编译**：修复后需要重新编译 OpenBLAS
   ```bash
   cd /build/OpenBLAS
   make -j$(nproc)
   pip install -e .
   ```
5. **验证修复**：运行测试确认修复成功
   ```bash
   bash /task/tests/test.sh
   ```
6. **检查分数**：测试会输出 0-1 的分数。**分数 ≥ 0.6 才算通过。**

## 约束条件

- **只允许修改汇编/头文件**(`.S` / `.s` / `.h`)，不允许修改 Python 文件
- **不允许**：
  - 换 BLAS 库(如用 MKL 替代)
  - 修改 NumPy 源码
  - 用纯 Python 实现替代 BLAS 调用

## 文件说明

- `/workspace/test_blas.py` — 测试脚本
- `/task/tests/test.sh` — 测试脚本(跑完后输出 0-1 分数)
- `/build/OpenBLAS/` — OpenBLAS 源码

## 运行环境

- OpenBLAS(从源码编译)
- NumPy 1.x
- x86_64 架构，支持 AVX2

## 验收标准

运行 `/task/tests/test.sh`，分数 ≥ 0.6 视为通过。

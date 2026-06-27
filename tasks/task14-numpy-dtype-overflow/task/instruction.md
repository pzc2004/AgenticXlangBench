# 任务：修复 NumPy 混合精度计算结果错误问题

## 背景

我们使用了一个从源码编译的 NumPy 2.0.0(源码在 `/build/numpy/`)。
在进行混合 dtype 的数组运算时，某些特定 dtype 组合的结果出现严重错误。

## Bug 现象

运行混合精度测试脚本：

```bash
cd /workspace
python test_dtype.py
```

预期：所有 dtype 组合的算术运算结果应正确(int8 + float16 应得到 float32 精度)。
实际：int8 + float16 的结果出现溢出(值被截断到 int8 范围)。

注意：单一 dtype 的运算 **结果正常**。问题仅出现在特定的混合 dtype 组合。

## 已知信息

- NumPy 是从源码编译的，源码在 `/build/numpy/`
- **混合 dtype 计算结果有误，特定组合(如 int8 + float16)出现溢出**
- 你需要找到并修复 NumPy 源码中导致 dtype 提升错误的 bug
- Bug 位于 `numpy/core/src/umath/ufunc_type_resolution.c` 中的类型提升逻辑

## 你的任务

1. **理解 bug**：分析为什么特定 dtype 组合的计算结果错误
2. **定位 bug**：在 `/build/numpy/` 的 C 源码中找到有问题的代码
3. **修复 bug**：修改 C 源码(只允许修改 `.c` / `.h` 文件)
4. **重新编译**：修复后需要重新编译 NumPy
   ```bash
   cd /build/numpy
   pip install -e . --no-build-isolation
   ```
5. **验证修复**：运行测试确认修复成功
   ```bash
   bash /task/tests/test.sh
   ```
6. **检查分数**：测试会输出 0-1 的分数。**分数 >= 0.6 才算通过。如果分数 < 0.6，说明修复不正确，请继续排查。**

## 约束条件

- **只允许修改 C 文件**(`.c` / `.h`)，不允许修改 Python 文件
- **不允许**：
  - 在 Python 层强制转换 dtype 绕过问题
  - 用 Python 循环替代向量化运算
  - 只使用单一 dtype 避免问题
  - 修改 test_dtype.py

## 文件说明

- `/workspace/test_dtype.py` — 混合精度测试脚本
- `/task/tests/test.sh` — 测试脚本(跑完后输出 0-1 分数)
- `/build/numpy/` — NumPy 源码

## 运行环境

- NumPy 2.0.0(从源码编译)
- 编译工具：gcc， make， pip

## 验收标准

运行 `/task/tests/test.sh`，分数 >= 0.6 视为通过。
**如果分数 < 0.6，说明修复不正确，请继续排查 NumPy 源码。**

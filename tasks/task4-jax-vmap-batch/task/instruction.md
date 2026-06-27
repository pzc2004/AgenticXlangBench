# 任务:修复 JAX vmap+grad 梯度错误

## 背景

我们使用了一个从源码安装的 JAX。在使用 `jax.vmap` + `jax.grad` 时,梯度计算结果不正确。

## Bug 现象

运行测试脚本:

```bash
cd /workspace
python test_vmap.py
```

预期:所有梯度检查通过(相对误差 < 1e-5)。
实际:使用 `vmap` 时梯度检查失败,不用 `vmap` 时完全正确。

## 已知信息

- JAX 是从源码安装的,源码在 `/build/jax/`
- **你需要找到并修复 JAX 源码中导致 vmap+grad 梯度错误的 bug**

## 你的任务

1. **理解 bug**:分析为什么 vmap+grad 给出错误梯度
2. **定位 bug**:在 `/build/jax/` 的 Python 源码中找到有问题的代码
3. **修复 bug**:修改 Python 源码(只允许修改 `.py` 文件)
4. **重新安装**:修复后需要重新安装 JAX
   ```bash
   cd /build/jax
   pip install -e .
   ```
5. **验证修复**:运行测试确认修复成功
   ```bash
   grade
   ```
   这会输出 0-1 的总分。**目标是 1.0 满分:只要分数没到 1.0,就说明还有 batching rule bug 没修干净,请继续逐操作对比定位、修复,直到分数无法再提高,不要在中途分数停手。**
6. **检查分数**:测试会输出 0-1 的分数。**分数 ≥ 0.6 才算通过。**

## 约束条件

- **只允许修改 Python 文件**(`.py`),不允许修改 C++/CUDA 文件
- **不允许**:
  - 绕过 vmap 直接实现
  - 修改测试脚本
  - 用 numpy 替代 jax

## 文件说明

- `/workspace/test_vmap.py` — 开发自测脚本 stub
- `/build/jax/` — JAX 源码

## 运行环境

- JAX(从源码安装)
- Python 3.12+

## 验收标准

运行 `grade`,分数 ≥ 0.6 视为通过。

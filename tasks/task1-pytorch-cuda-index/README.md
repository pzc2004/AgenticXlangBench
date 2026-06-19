# Task 1: CUDA LayerNorm Off-by-One → 训练 NaN

## 概述

在自定义 CUDA LayerNorm 内核中注入一个 off-by-one 错误。
Bug 在 CUDA 层触发,但症状在 Python 层表现为训练 loss 变为 NaN,
且**延迟 ~90 个 iteration 才显现** —— 因为越界写入的累积效应需要多次迭代才破坏到关键内存。

## Bug 设计

- **位置**:`layernorm_cuda/layernorm_cuda_kernel.cu`
- **类型**:off-by-one(`for (j = threadIdx.x; j < N; ...)` 改成 `j <= N`)
- **效果**:越界读写 1 个元素,破坏相邻内存
- **触发条件**:每次 LayerNorm forward 都触发

## 延迟显现机制

```
LayerNorm CUDA kernel 越界读写 → 破坏相邻 tensor 的 1 个元素
    ↓ 该 tensor 参与后续计算(Linear + ReLU)
误差在后续层传播 → 梯度计算受影响
    ↓ 累积 ~90 个 iteration
梯度爆炸 → loss NaN
```

**关键**:前 89 个 iteration 都"看起来正常",第 90 个才 NaN。
Agent 不能一眼看到 bug。

## 为什么难

1. **跨语言**:症状在 Python,bug 在 CUDA,需要读两种语言
2. **症状模糊**:NaN 可能是 100 种原因(learning rate / 数据 / 模型 / bug)
3. **无 stack trace**:NaN 不指向 CUDA kernel,指向 Python 层的 loss 计算
4. **需要理解 CUDA 编程**:线程索引、越界访问、内存布局
5. **延迟显现**:bug 执行后跑了大量操作才出现症状

## 预估轨迹

- **步数**:100-200 步
- **时间**:1-2 小时
- **关键转折点**:
  - 步骤 20:发现 NaN,开始排查
  - 步骤 50:定位到 LayerNorm 相关
  - 步骤 80:找到自定义 CUDA 扩展
  - 步骤 100:读懂 CUDA 内核逻辑
  - 步骤 120:找到 off-by-one bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `isnan` / `nan_to_num` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 换用原生 LayerNorm | grep `nn.LayerNorm` |
| 改训练参数 | diff 检查 |
| 注释掉越界代码 | diff 检查 |

## Oracle

```bash
#!/bin/bash
# 撤销 off-by-one: j <= N → j < N
sed -i 's/for (int j = threadIdx.x; j <= N;/for (int j = threadIdx.x; j < N;/g' \
    layernorm_cuda/layernorm_cuda_kernel.cu
# 清除编译缓存,强制重新编译
rm -rf /root/.cache/torch_extensions/py310_cu126/layernorm_cuda
```

## 资源需求

- **GPU**:1× RTX A6000(4GB+)
- **编译**:30-60 秒(CUDA 扩展编译)
- **训练验证**:1-2 分钟(100 iteration)
- **磁盘**:~500MB

---

## 开发踩坑记录

> 以下是在开发 Task 1 过程中遇到的问题和解决方案,供后续题目参考。

### 坑 1:从源码编译 PyTorch 极其复杂

**问题**:最初计划 clone PyTorch 源码 → 注入 bug → 从源码编译。但:
- `git submodule update --init --recursive` 经常失败(网络问题、子模块嵌套)
- `python setup.py develop` 需要大量依赖(CMake、ninja、MKL 等)
- 全量编译需要 30-60 分钟,且容易因缺少某个子模块失败
- `third_party/ideep` / `third_party/kineto` 等子模块经常 clone 失败

**解决**:放弃从源码编译 PyTorch,改用**自定义 CUDA 扩展**(`torch.utils.cpp_extension.load()`)。
- 只编译一个 `.cu` + 一个 `.cpp` 文件(30 秒)
- 不需要 clone PyTorch 源码
- 不影响已安装的 PyTorch

**结论**:后续题目如果需要在 CUDA 层注入 bug,**优先用自定义 CUDA 扩展**,不要试图从源码编译整个框架。

### 坑 2:`torch.utils.cpp_extension.load()` 有缓存

**问题**:修改 `.cu` 文件后,`load()` 不会自动重新编译,仍用缓存的旧版本。

**解决**:修改源码后,必须清除缓存:
```bash
rm -rf /root/.cache/torch_extensions/py310_cu126/<extension_name>
```

**注意**:缓存路径格式 `/root/.cache/torch_extensions/<python_version>_<cuda_version>/<extension_name>`。

### 坑 3:`set -e` 导致测试脚本提前退出

**问题**:测试脚本用 `set -e`,但 `grep` 在找不到匹配时返回非零,导致脚本退出。

**解决**:测试脚本**不要用 `set -e`**。用显式错误处理:
```bash
# 不要这样:
set -e
if echo "$result" | grep -q "pattern"; then ... fi  # grep 失败时脚本退出

# 要这样:
if echo "$result" | grep -q "pattern"; then ... fi  # grep 失败不影响脚本
```

### 坑 4:`$(dirname "$0")` 路径解析不可靠

**问题**:测试脚本用 `$(dirname "$0")/..` 获取任务目录,但在某些调用方式下解析错误(如 `bash /tmp/test.sh`)。

**解决**:用 `cd && pwd` 获取绝对路径:
```bash
# 不要这样:
TASK_DIR="$(dirname "$0")/.."

# 要这样:
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
```

### 坑 5:训练脚本必须调用 buggy kernel

**问题**:最初 train.py 用 `nn.ReLU()`,但 bug 在 LayerNorm kernel 里。bug 永远不会被触发。

**解决**:train.py 必须用**自定义的 CudaLayerNorm**(调用 buggy kernel),不能用 PyTorch 原生的 `nn.LayerNorm`。

**检查**:anti-hack 检测应包含 `grep "nn.LayerNorm"` 来防止 agent 换回原生实现。

### 坑 6:CUDA 扩展的 `__init__.py` 不是必须的

**问题**:最初写了 `__init__.py` 来封装 CUDA 扩展加载,但 `torch.utils.cpp_extension.load()` 可以直接从 `.cu` 文件编译,不需要 Python 包装。

**解决**:train.py 直接用 `load()` 编译并加载,不需要 `__init__.py`:
```python
from torch.utils.cpp_extension import load
m = load(name='my_ext', sources=['my_ext.cpp', 'my_kernel.cu'], verbose=False)
```

### 坑 7:off-by-one bug 在第 0 步就 NaN(不是延迟 90 步)

**问题**:最初设计的 bug(`j <= N`)在第 0 步就导致 NaN,而不是延迟 90 步。原因是越界读取的内存恰好包含 NaN/Inf 值。

**解决**:这是可接受的 —— bug 的"延迟"取决于内存布局。不同 seed 可能在不同步数触发 NaN。关键是**bug 存在时一定触发 NaN,修复后一定不触发**。

**启示**:延迟显现的时间**不可精确控制**,取决于:
- 越界读写的内存内容
- 模型结构(LayerNorm 层数)
- batch size
- 随机种子

### 坑 8:测试脚本的 anti-hack 检查需要区分"任务初始化"和"agent 修改"

**问题**:测试脚本检查 `git diff` 来判断是否修改了 train.py。但 train.py 是任务初始化时写的,不是 agent 改的。

**解决**:不要用 `git diff` 检查 train.py。改用**语义检查**:
```bash
# 检查是否换用了原生 LayerNorm
if grep -q "nn.LayerNorm" "$WORKSPACE/train.py"; then HACK=1; fi

# 检查是否有 NaN hack
if grep -q "isnan\|nan_to_num\|clip_grad" "$WORKSPACE/train.py"; then HACK=1; fi
```

### 通用经验总结

| 经验 | 说明 |
|---|---|
| **用 CUDA 扩展,不从源码编译框架** | 30 秒 vs 30 分钟,且不需要 clone 整个框架 |
| **修改源码后清缓存** | `rm -rf /root/.cache/torch_extensions/...` |
| **测试脚本不用 `set -e`** | grep 等命令返回非零会中断脚本 |
| **路径用绝对路径** | `$(cd "$(dirname "$0")" && pwd)` |
| **确保 buggy kernel 被调用** | 训练脚本必须用自定义实现,不能用框架原生的 |
| **anti-hack 用语义检查** | 检查代码内容,不检查 git diff |
| **延迟显现不可精确控制** | 接受不同 seed 触发步数不同,只要"有 bug 一定触发" |

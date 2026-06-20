# Task 1: PyTorch LayerNorm CUDA rsqrt Bug → 训练 NaN

## 概述

在 PyTorch 源码的 LayerNorm CUDA forward kernel 中注入一个 bug:去掉 `rsqrt` 的 `eps` 保护。
当方差 = 0 时(`rsqrt(0)` = Inf → `0 * Inf` = NaN),训练 loss 变 NaN。
每 20 步注入零方差 batch 触发,约第 90 步 NaN 出现。

## Bug 设计

- **位置**: `aten/src/ATen/native/cuda/layer_norm_kernel.cu` 第 246 行
- **类型**: `rsqrt(wd.sigma2 + eps)` → `rsqrt(wd.sigma2)` (去掉 eps 保护)
- **注入方式**: `solution/inject_bug.py` 注入 1 个真 bug + 19 个诱饵
- **效果**: 当方差 = 0 时,`rsqrt(0)` = Inf,`(x - mean) * Inf` = NaN
- **触发条件**: 零方差输入(batch 内所有特征相同)

## 延迟显现机制

```
每 20 步:注入零方差 batch(所有特征 = 相同值)
    ↓ LayerNorm 看到方差 = 0
rsqrt(0) = Inf → (x - mean) * Inf = NaN
    ↓ NaN 传播到 loss → NaN 梯度 → 模型权重变 NaN
    ↓ 后续所有步骤都是 NaN
第 ~90 步:loss 检测到 NaN
```

## 为什么难

1. **无提示**:train.py 和 instruction.md 不提"LayerNorm"、"rsqrt"、"零方差",agent 要自己定位
2. **19 个诱饵**:20 个 CUDA 文件被修改,只有 1 个是真 bug
3. **git diff 无效**:改动已提交到 git,agent 用 `git diff` 看不到
4. **跨语言**:症状在 Python(loss NaN),bug 在 CUDA(rsqrt)
5. **无 stack trace**:NaN 不指向 CUDA kernel
6. **需要理解 CUDA 编程**:Welford 算法、rsqrt 数值稳定性
7. **需要增量重编**:修复后要 ninja 重编 + cp .so 到 site-packages

## 预估轨迹

- **步数**: 200-400 步
- **时间**: 2-4 小时
- **关键转折点**:
  - 步骤 30: 发现 NaN,开始排查
  - 步骤 80: 锁定 LayerNorm 是问题(写测试排除其他 op)
  - 步骤 130: 在 PyTorch 源码中找到 layer_norm_kernel.cu
  - 步骤 180: 读懂 CUDA 内核逻辑(Welford 算法)
  - 步骤 230: 找到 rsqrt 缺少 eps 的 bug
  - 步骤 280: 修复 + 增量重编 + 验证

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `isnan` / `nan_to_num` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 改训练参数 | diff 检查 |
| 用 try/catch 吞错误 | grep `try:` |
| 用 git diff 找答案 | 改动已提交,`git diff` 为空 |

## Oracle

`solution/solve.sh`:
```bash
# 恢复 eps 保护
sed -i 's/rsqrt(wd\.sigma2)/rsqrt(wd.sigma2 + eps)/' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
# 增量重编
cd /build/pytorch/build && ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

## 资源需求

- **GPU**: 1× RTX A6000(4GB+)
- **fat base 构建**: 1-2 小时(一次性)
- **task1 增量构建**: 5-15 分钟
- **训练验证**: 1-2 分钟
- **磁盘**: ~30GB(fat base) + ~1GB(task1)

## 架构

```
pytorch-2.5.0-fat-base (一次性构建,1-2 小时):
  PyTorch 2.5.0 源码 + .git + third_party + build 目录
  支持 ninja 增量编译

task1 (增量构建,5-15 分钟):
  FROM fat-base
  → inject_bug.py 注入 1 真 bug + 19 诱饵
  → git commit 隐藏改动(防 agent 用 git diff 找)
  → ninja 增量编译(只重编 1 个 .cu → 重新链接 .so)
  → cp .so 到 site-packages
  → 安装 Kimi Code + Claude Code
  → 复制 train.py
```

## Agent 评测

容器内预装了 Kimi Code 和 Claude Code,可通过 `run.sh` 和 `calibrate.sh` 评测:

```bash
# 单次运行(Kimi Code)
./run.sh kimi-for-coding 10 42

# 校准(3 次运行)
./calibrate.sh kimi-for-coding 10 3
```

**限制**:
- 预算: $10/次
- 步数: 500 步/轮
- 时间: 1 小时超时

---

## 开发踩坑记录

### 坑 1: 自定义 CUDA 扩展方案太简单

**问题**:最初用自定义 CUDA 扩展(`layernorm_override.so`)实现 buggy LayerNorm。agent 一眼就能看到 `layernorm_cuda/` 目录,难度太低。

**解决**:改为在 PyTorch 源码中注入 bug,用 ninja 增量编译。agent 需要在几百万行代码中定位。

### 坑 2: setup.py install 不支持增量编译

**问题**:改了 .cu 源码后,`setup.py install` 检测到已安装就跳过编译。删掉 .o 和 .so 后重跑也没用。

**解决**:直接用 `ninja` 命令:
```bash
cd /build/pytorch/build
ninja -j32 lib/libtorch_cuda.so   # 只重编改过的 .o + 链接
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

### 坑 3: .so 必须手动复制到 site-packages

**问题**:ninja 编译的 .so 在 `build/lib/`,但 Python 用的是 `site-packages/torch/lib/`。两者不同步。

**解决**:编译后手动 `cp`。

### 坑 4: rsqrt 去掉 eps 在随机数据下不触发 NaN

**问题**:`rsqrt(sigma2)` 和 `rsqrt(sigma2 + eps)` 在随机数据下几乎没区别(方差从不精确等于 0)。

**解决**:在 train.py 里每 20 步注入零方差 batch(所有特征设为相同值),强制触发 `rsqrt(0)`。

### 坑 5: BatchNorm 破坏零方差

**问题**:模型有 BatchNorm,它会把零方差输入"修"成非零方差,导致 LayerNorm 看不到零方差。

**解决**:去掉 BatchNorm,让 LayerNorm 成为第一层。

### 坑 6: rsqrt 改错位置

**问题**:最初 sed 替换 `rsqrt(m2 + eps)`,但改的是 `RowwiseMomentsCUDAKernel`(算方差的),不是 `LayerNormForwardCUDAKernel`(用方差的)。两者变量名不同。

**解决**:精确匹配 `rsqrt(wd.sigma2 + eps)` (forward kernel 的变量名是 `wd.sigma2`,不是 `m2`)。

### 坑 7: git diff 暴露 bug

**问题**:agent 跑 `git diff` 就能看到改动,绕过调试过程。

**解决**:注入后 `git add -A && git commit`,让 `git diff` 为空。同时配置 git user.email/user.name。

### 坑 8: Docker secret 层缓存不刷新

**问题**:`--mount=type=secret` 的层被 Docker 缓存,API key 改了也不会重新构建。

**解决**:用 `--no-cache` 或把 secret 放在 Dockerfile 末尾(只重建末尾层)。

### 坑 9: Kimi Code 需要 provider 前缀

**问题**:Kimi Code 的 config.toml 格式比预期复杂,provider 名必须是 `"managed:kimi-code"`,model 引用必须带前缀 `"kimi-code/kimi-for-coding"`。

**解决**:查官方文档 `moonshotai.github.io/kimi-code/zh/configuration/config-files`。

### 坑 10: train.py 注释不能暴露 bug

**问题**:train.py 的注释里写了"LayerNorm + rsqrt + 零方差",agent 直接看到就找到答案了。

**解决**:删掉所有暗示 bug 机制的注释,train.py 只描述"训练出现 NaN",不解释原因。

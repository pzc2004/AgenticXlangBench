# Task 1: PyTorch LayerNorm CUDA rsqrt Bug → 训练 NaN

## 概述

在 PyTorch 源码的 LayerNorm CUDA forward kernel 中注入一个 bug:去掉 `rsqrt` 的 `eps` 保护。
当方差 = 0 时(`rsqrt(0)` = Inf → `0 * Inf` = NaN),训练 loss 变 NaN。
通过 train.py 的 pre-hook 每 40 步注入零方差输入触发,约第 90-180 步 NaN 出现。

## Bug 设计

- **位置**: `/build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu` 第 246 行
- **类型**: `rsqrt(wd.sigma2 + eps)` → `rsqrt(wd.sigma2)` (去掉 eps 保护)
- **注入方式**: `solution/inject_bug.py` 注入 1 个真 bug + 19 个诱饵(注释/宏)
- **效果**: 当方差 = 0 时,`rsqrt(0)` = Inf,`(x - mean) * Inf` = NaN
- **触发条件**: train.py 的 pre-hook 每 40 步将 LayerNorm 输入替换为零方差数据

## 模型结构(LayerNorm 藏在中间,不是第一层)

```
Conv2d → BatchNorm → ReLU → Conv2d → GELU → MaxPool → Flatten
    → Linear → LayerNorm(第 8 层) → ReLU
    → Linear → LayerNorm → Dropout
    → Linear → 输出
```

agent 需要读 `model.py` 才能知道模型用了哪些 op。LayerNorm 藏在第 8 层,NaN 出现在 step ~90,
agent 不能通过"NaN 出现在最早"直接锁定 LayerNorm。

## 零方差注入机制(train.py pre-hook)

```python
def _zero_variance_pre_hook(module, input):
    _step_counter[0] += 1
    if _step_counter[0] % 40 == 0 and _step_counter[0] > 0:
        x = input[0]
        val = torch.rand(x.shape[0], 1, device=x.device)
        uniform = val.expand_as(x).clone()
        return (uniform,)  # 修改后的输入,直接送 CUDA kernel
    return None
```

**关键**:用 `register_forward_pre_hook` 而不是 `register_forward_hook`。
pre-hook 修改输入后,CUDA kernel 直接处理 uniform 数据 → rsqrt(0) = Inf → NaN。
post-hook 会替换输出,绕过 CUDA kernel,NaN 不触发。

## 为什么难

1. **模型用 10+ 种 op**:Conv/BN/ReLU/GELU/MaxPool/Linear/LayerNorm/Dropout,agent 不知道哪个有问题
2. **19 个诱饵**:20 个 CUDA 文件被修改,只有 1 个是真 bug
3. **git diff 无效**:改动已提交到 git,agent 用 `git diff` 看不到
4. **跨语言**:症状在 Python(loss NaN),bug 在 CUDA(rsqrt)
5. **无 stack trace**:NaN 不指向 CUDA kernel
6. **hook 注入**:零方差通过 pre-hook 注入,agent 需要理解 hook 机制
7. **需要增量重编**:修复后要 ninja 重编 + cp .so 到 site-packages

## 预估轨迹

- **步数**: 100-300 步
- **时间**: 1-3 小时
- **关键转折点**:
  - 步骤 20: 发现 NaN,开始排查
  - 步骤 50: 读 model.py,列出所有可能的 op
  - 步骤 80: 读 train.py,发现 pre-hook 注入零方差到 LayerNorm
  - 步骤 100: 在 PyTorch 源码中找到 layer_norm_kernel.cu
  - 步骤 130: 读懂 CUDA 内核逻辑,找到 rsqrt 缺 eps
  - 步骤 150: 修复 + 增量重编 + 验证

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `nan_to_num` / `clip_grad` / `clamp` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 换用原生 LayerNorm | grep `nn.LayerNorm`(train.py 不应改) |
| 改训练参数 | diff 检查 |
| 用 try/catch 吞错误 | grep `try:` |
| 用 git diff 找答案 | 改动已提交,`git diff` 为空 |
| 去掉 pre-hook | train.py 不应被修改 |

## Oracle

`solution/solve.sh`:
```bash
sed -i 's/rsqrt(wd\.sigma2)/rsqrt(wd.sigma2 + eps)/' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
cd /build/pytorch/build && ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

## 校准结果(Kimi Code)

| Seed | Reward | 轨迹大小 | 步数 | 是否修了 bug |
|---|---|---|---|---|
| 1 | 0.60 | 55KB | 8 步 | ✅ 修了 |
| 2 | 0.60 | 63KB | ~10 步 | ✅ 修了 |
| 3 | 0.60 | 58KB | ~8 步 | ✅ 修了 |

Kimi Code 的推理链(8 步):
1. Read(train.py + model.py) → 发现用了 LayerNorm
2. Grep("layer_norm") + Glob(*layer_norm*) → 找到 CUDA kernel
3. Read(layer_norm_kernel.cu) → 读源码
4. Bash(train.py) → 看到 NaN
5. Edit(修 rsqrt) → 修 bug
6. Bash(ninja + cp) → 重编译
7. Bash(test.sh) → 验证通过

**问题**:Kimi 通过 model.py 直接看到 `nn.LayerNorm`,一步锁定目标。19 个诱饵没发挥作用。

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
  干净源码(无 bug),支持 ninja 增量编译

task1 (增量构建,5-15 分钟):
  FROM fat-base
  → inject_bug.py 注入 1 真 bug + 19 诱饵
  → git commit 隐藏改动
  → ninja 增量编译(只重编 layer_norm_kernel.cu.o → 链接 libtorch_cuda.so)
  → cp .so 到 site-packages
  → 安装 Kimi Code + Claude Code
  → 复制 train.py + model.py

Agent 进入容器后:
  /workspace/train.py          ← 训练脚本(含 pre-hook 注入)
  /workspace/model.py          ← 模型定义(LayerNorm 藏在中间)
  /build/pytorch/              ← PyTorch 源码(20 个文件被修改)
```

## 文件结构

```
task/
├── task.toml              ← 任务元数据
├── instruction.md         ← 发给 agent 的 prompt
├── environment/
│   ├── Dockerfile         ← 构建 task1 镜像
│   └── Dockerfile.base    ← fat base 镜像定义
├── workspace/
│   ├── train.py           ← 训练脚本(含 pre-hook)
│   └── model.py           ← 模型定义(10+ 种 op,LayerNorm 在中间)
├── solution/
│   ├── inject_bug.py      ← 注入 1 真 bug + 19 诱饵
│   └── solve.sh           ← Oracle(撤销 bug)
└── tests/
    └── test.sh            ← 判题脚本(分层评分 + anti-hack)
```

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

**解决**:通过 pre-hook 注入零方差输入到 LayerNorm,强制触发 `rsqrt(0)`。

### 坑 5: register_forward_hook 绕过 CUDA kernel

**问题**:用 `register_forward_hook` 修改 LayerNorm 输出时,手动调了 `torch.rsqrt`(正确的 Python 实现),绕过了 CUDA kernel 的 buggy rsqrt,NaN 不触发。

**解决**:改用 `register_forward_pre_hook`,只修改输入不替换输出。CUDA kernel 直接处理 uniform 数据 → 触发 bug。

### 坑 6: rsqrt 改错位置

**问题**:最初 sed 替换 `rsqrt(m2 + eps)`,但改的是 `RowwiseMomentsCUDAKernel`(算方差的),不是 `LayerNormForwardCUDAKernel`(用方差的)。两者变量名不同。

**解决**:精确匹配 `rsqrt(wd.sigma2 + eps)` (forward kernel 的变量名是 `wd.sigma2`,不是 `m2`)。

### 坑 7: git diff 暴露 bug

**问题**:agent 跑 `git diff` 就能看到改动,绕过调试过程。

**解决**:注入后 `git add -A && git commit`,让 `git diff` 为空。

### 坑 8: Docker secret 层缓存不刷新

**问题**:`--mount=type=secret` 的层被 Docker 缓存,API key 改了也不会重新构建。

**解决**:用 `--build-arg CACHE_BUST=$(date +%s)` 强制重跑最后几层。

### 坑 9: API key 不应烤进镜像

**问题**:最初把 SSH key 和 API key 通过 Dockerfile 的 RUN 步骤写入镜像,不安全。

**解决**:SSH key 通过 `--mount=type=secret` 挂载(用完即消失)。API key 通过 `docker run -e` 或 `-v` 运行时注入。

### 坑 10: LayerNorm 是第一层 → agent 一眼锁定

**问题**:最初模型是 `LayerNorm → Linear → ReLU`,NaN 在 step 20 出现,agent 一眼锁定 LayerNorm。

**解决**:把 LayerNorm 移到模型中间(Conv → BN → ReLU → GELU → MaxPool → Linear → **LayerNorm**),NaN 出现时机不直接指向 LayerNorm。19 个诱饵分布在其他 op 的 CUDA kernel 里,agent 需要逐个排查。

### 坑 11: inject_bug.py 需要先恢复再注入

**问题**:Docker 缓存可能导致源码已经被改成 buggy 版,再次注入时找不到原始代码。

**解决**:inject_bug.py 先恢复干净版 `rsqrt(wd.sigma2 + eps)`,再注入 bug `rsqrt(wd.sigma2)`。

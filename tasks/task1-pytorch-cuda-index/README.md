# Task 1: PyTorch LayerNorm CUDA rsqrt 负号 Bug → 训练 NaN

## 概述

在 PyTorch 源码的 LayerNorm CUDA forward kernel 中注入一个 bug:在 `rsqrt` 公式里加一个负号。
`rsqrt(wd.sigma2 + eps)` → `rsqrt(-wd.sigma2 + eps)`,当方差 > eps 时参数为负 → `rsqrt(负数)` = NaN。
**训练第 0 步就 NaN**,不需要任何 hook 或特殊输入。

## Bug 设计

- **位置**: `/build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu` 第 246 行
- **类型**: `rsqrt(wd.sigma2 + eps)` → `rsqrt(-wd.sigma2 + eps)` (负号 typo)
- **注入方式**: `solution/inject_bug.py` 注入 1 个真 bug + 19 个诱饵(注释/宏)
- **效果**: 当 `var > eps` 时,`-var + eps < 0` → `rsqrt(负数)` = NaN
- **触发条件**: 正常随机数据即可触发(方差 > eps 恒成立)

## 模型结构(FeatureNorm 封装 LayerNorm,不直接暴露)

```
Conv2d → BatchNorm → ReLU → Conv2d → GELU → MaxPool → Flatten
    → Linear → FeatureNorm(第 8 层) → ReLU
    → Linear → FeatureNorm → Dropout
    → Linear → 输出
```

- `FeatureNorm` 是自定义类,内部调用 `F.layer_norm`
- agent 需要读 `model.py` 才能知道 `FeatureNorm` 底层是 LayerNorm
- NaN 在第 0 步就出现,agent 不能通过"出现时间"推断是哪个 op

## 训练脚本(无 hook,纯随机数据)

```python
def make_batch(batch_size, device):
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y
```

**没有 hook、没有注入、没有特殊处理**。NaN 完全由 CUDA kernel bug 导致。

## 为什么难

1. **无提示**:train.py、model.py、test.sh 都不暴露 bug 细节
2. **模型用 10+ 种 op**:Conv/BN/ReLU/GELU/MaxPool/Linear/FeatureNorm/Dropout
3. **19 个诱饵**:20 个 CUDA 文件被修改,只有 1 个是真 bug
4. **git diff 无效**:改动已提交到 git
5. **跨语言**:症状在 Python(loss NaN),bug 在 CUDA(rsqrt)
6. **无 stack trace**:NaN 不指向 CUDA kernel
7. **FeatureNorm 封装**:agent 需要读 model.py 才能知道底层是 LayerNorm
8. **需要增量重编**:修复后要 ninja 重编 + cp .so

## 预估轨迹

- **步数**: 100-300 步
- **时间**: 1-3 小时
- **关键转折点**:
  - 步骤 10: 跑 train.py,看到 NaN
  - 步骤 30: 读 model.py,列出所有可能的 op
  - 步骤 50: 读 FeatureNorm 实现,发现底层是 F.layer_norm
  - 步骤 80: 在 PyTorch 源码中找到 layer_norm_kernel.cu
  - 步骤 100: 排查 19 个诱饵文件
  - 步骤 120: 读懂 CUDA 内核,找到负号 bug
  - 步骤 140: 修复 + 增量重编 + 验证

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `nan_to_num` / `clip_grad` / `clamp` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 修改 train.py | 检查关键内容(ImageClassifier)是否还在 |
| 修改 model.py | 检查关键内容(FeatureNorm / F.layer_norm)是否还在 |
| 用 git diff 找答案 | 改动已提交,`git diff` 为空 |

## Oracle

`solution/solve.sh`:
```bash
# 恢复: rsqrt(-wd.sigma2 + eps) → rsqrt(wd.sigma2 + eps)
sed -i 's/rsqrt(-wd\.sigma2 + eps)/rsqrt(wd.sigma2 + eps)/' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
cd /build/pytorch/build && ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

## 校准结果(Kimi Code)

| Seed | Reward | 轨迹大小 | 步数 | 修了什么 |
|---|---|---|---|---|
| 1 | **1.0** | 66KB | 14 步 | 修了 CUDA kernel 的负号 ✅ |

Kimi Code 推理链(14 步):
1. Read(train.py + model.py + test.sh) → 看到 ImageClassifier + FeatureNorm
2. 跑 train.py → NaN 在 step 0
3. Python 测试 × 4 → 理解 FeatureNorm 底层是 LayerNorm
4. Glob(*layer_norm*) → 找到 CUDA 文件
5. Read(layer_norm_kernel.cu) → 读源码
6. **Grep(-wd.sigma2)** → 找到负号 bug
7. Edit → 修了
8. ninja + cp → 重编译
9. 跑 train.py → 无 NaN
10. 跑 test.sh → **reward = 1.0**

## 资源需求

- **GPU**: 1× RTX A6000(4GB+)
- **fat base 构建**: 1-2 小时(一次性)
- **task1 增量构建**: 5-15 分钟
- **训练验证**: < 1 分钟(第 0 步就 NaN)
- **磁盘**: ~30GB(fat base) + ~1GB(task1)

## 架构

```
pytorch-2.5.0-fat-base (一次性构建,1-2 小时):
  PyTorch 2.5.0 源码 + .git + third_party + build 目录
  干净源码(无 bug),支持 ninja 增量编译

task1 (增量构建,5-15 分钟):
  FROM fat-base
  → inject_bug.py 注入 1 真 bug(负号) + 19 诱饵
  → git commit 隐藏改动
  → ninja 增量编译
  → cp .so 到 site-packages
  → 安装 Kimi Code + Claude Code
  → 复制 train.py + model.py

Agent 进入容器后:
  /workspace/train.py          ← 训练脚本(纯随机数据,无 hook)
  /workspace/model.py          ← 模型定义(FeatureNorm 封装 LayerNorm)
  /build/pytorch/              ← PyTorch 源码(20 个文件被修改)
```

## 文件结构

```
task/
├── task.toml              ← 任务元数据
├── instruction.md         ← 发给 agent 的 prompt(不暴露 bug 细节)
├── environment/
│   ├── Dockerfile         ← 构建 task1 镜像
│   └── Dockerfile.base    ← fat base 镜像定义
├── workspace/
│   ├── train.py           ← 训练脚本(纯随机数据)
│   └── model.py           ← 模型定义(FeatureNorm 封装 LayerNorm)
├── solution/
│   ├── inject_bug.py      ← 注入 1 真 bug + 19 诱饵
│   └── solve.sh           ← Oracle(撤销负号)
└── tests/
    └── test.sh            ← 判题脚本(不暴露 bug 细节)
```

---

## 开发踩坑记录

### 坑 1: 自定义 CUDA 扩展方案太简单

**问题**:最初用自定义 CUDA 扩展(`layernorm_override.so`)实现 buggy LayerNorm。agent 一眼就能看到 `layernorm_cuda/` 目录,难度太低。

**解决**:改为在 PyTorch 源码中注入 bug,用 ninja 增量编译。agent 需要在几百万行代码中定位。

### 坑 2: setup.py install 不支持增量编译

**问题**:改了 .cu 源码后,`setup.py install` 检测到已安装就跳过编译。

**解决**:直接用 `ninja` 命令:
```bash
cd /build/pytorch/build
ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

### 坑 3: .so 必须手动复制到 site-packages

**问题**:ninja 编译的 .so 在 `build/lib/`,但 Python 用的是 `site-packages/torch/lib/`。

**解决**:编译后手动 `cp`。

### 坑 4: 去掉 eps 的 bug 在随机数据下不触发 NaN

**问题**:`rsqrt(sigma2)` 和 `rsqrt(sigma2 + eps)` 在随机数据下几乎没区别(方差从不精确等于 0)。

**解决(旧)**:通过 pre-hook 注入零方差输入。

**解决(新)**:改成负号 typo `rsqrt(-wd.sigma2 + eps)`,随机数据下 `var > eps` 恒成立 → `rsqrt(负数)` = NaN,不需要 hook。

### 坑 5: register_forward_hook 绕过 CUDA kernel

**问题**:用 `register_forward_hook` 修改 LayerNorm 输出时,手动调了 `torch.rsqrt`(正确的 Python 实现),绕过了 CUDA kernel 的 buggy rsqrt。

**解决(旧)**:改用 `register_forward_pre_hook`。

**解决(新)**:去掉 hook,用始终触发的 bug(负号 typo)。

### 坑 6: rsqrt 改错位置

**问题**:最初 sed 替换 `rsqrt(m2 + eps)`,但改的是 `RowwiseMomentsCUDAKernel`(算方差的),不是 `LayerNormForwardCUDAKernel`(用方差的)。

**解决**:精确匹配 `rsqrt(wd.sigma2 + eps)` (forward kernel 的变量名是 `wd.sigma2`,不是 `m2`)。

### 坑 7: git diff 暴露 bug

**问题**:agent 跑 `git diff` 就能看到改动。

**解决**:注入后 `git add -A && git commit`,让 `git diff` 为空。

### 坑 8: Docker secret 层缓存不刷新

**问题**:`--mount=type=secret` 的层被 Docker 缓存,API key 改了也不会重新构建。

**解决**:用 `--build-arg CACHE_BUST=$(date +%s)` 强制重跑。

### 坑 9: API key 不应烤进镜像

**问题**:最初把 SSH key 和 API key 写入镜像,不安全。

**解决**:SSH key 通过 `--mount=type=secret` 挂载。API key 通过 `docker run -e` 或 `-v` 运行时注入。

### 坑 10: LayerNorm 是第一层 → agent 一眼锁定

**问题**:最初模型是 `LayerNorm → Linear → ReLU`,agent 一眼锁定 LayerNorm。

**解决**:把 LayerNorm 移到模型中间,用 `FeatureNorm` 封装,不直接暴露 `nn.LayerNorm`。

### 坑 11: inject_bug.py 需要先恢复再注入

**问题**:Docker 缓存可能导致源码已经被改成 buggy 版。

**解决**:inject_bug.py 先恢复干净版,再注入新 bug。

### 坑 12: hook 暴露问题所在 op

**问题**:pre-hook 只作用于 FeatureNorm,agent 读 train.py 就知道问题在 LayerNorm。

**解决(新)**:去掉 hook,用始终触发的 bug(负号 typo),agent 需要自己推理哪个 op 有问题。

### 坑 13: test.sh 暴露 bug 细节

**问题**:test.sh 里写了 `rsqrt(wd.sigma2 + eps)` 和 `rsqrt(wd.sigma2)`,agent 直接看到答案。

**解决**:test.sh 只检查结果(有没有 NaN),不暴露 bug 细节。

### 坑 14: docker commit 保留 agent 修复

**问题**:run.sh 在 Kimi 容器退出后启动新容器跑 test.sh,新容器没有 Kimi 的修复。

**解决**:用 `docker commit` 保存容器状态,在快照镜像上跑 test.sh。

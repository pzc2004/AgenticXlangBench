# Task 1: PyTorch LayerNorm 复合 Bug → 训练异常

## 概述

在 PyTorch 源码的 LayerNorm CUDA kernel 中注入 **3 个复合 bug**，每个在不同条件下触发：
- Bug 1：backward 梯度符号翻转（所有数据）
- Bug 2：forward eps 条件错误（方差 < 0.1 时）
- Bug 3：forward NaN 注入（blockIdx.x > 32 时）

单一 bug 容易被秒，复合 bug 让 agent 修了 A 还有 B，修了 B 还有 C。

## Bug 设计

| Bug | 位置 | 触发条件 | 表现 | 定位难度 |
|-----|------|---------|------|---------|
| Bug 1 | vectorized backward | 所有数据 | 梯度方向错误 | 需要分析 backward 公式 |
| Bug 2 | vectorized forward | var < 0.1 | 归一化异常（eps 变大） | 需要理解 eps 含义 |
| Bug 3 | vectorized forward | blockIdx.x > 32 | NaN（1/8 的行） | 需要理解 CUDA 索引 |

## 为什么复合 bug 更难

1. **修了 A 还有 B**：agent 修完 Bug 1 后，测试仍然失败（Bug 2+3 还在）
2. **修了 B 还有 C**：agent 修完 Bug 2 后，大 batch 仍然 NaN（Bug 3）
3. **每个 bug 需要不同的调试技能**：
   - Bug 1：分析 backward 梯度公式
   - Bug 2：理解 eps 的数学含义
   - Bug 3：理解 CUDA blockIdx 索引
4. **agent 容易以为修完了**：修完 Bug 1+2 后，小 batch 测试通过，但大 batch 仍然 NaN

## 实测结果

| 版本 | Kimi 消息数 | Reward | 说明 |
|------|-----------|--------|------|
| 单一符号翻转 | 41 | 1.0 | 被秒 |
| eps * 1000 + 多层归一化 | 78 | 1.0 | 还是被秒 |
| **3 个复合 bug** | **246** | **0.15** | Kimi 只修了 2/3 |

## 调试轨迹分析（Kimi，246 步）

Kimi 的调试路径：
1. 读 train.py / model.py / test.sh（步 0-10）
2. 对比 CPU vs CUDA 梯度，定位到 LayerNorm（步 10-30）
3. 读 layer_norm_kernel.cu 源码（步 30-40）
4. **修复 Bug 2（条件 eps）**（步 32）
5. **修复 Bug 1（梯度符号）**（步 34）
6. 重编译 + 测试（步 38-50）
7. 发现大 batch 仍然 NaN，开始排查（步 50-200）
8. 尝试理解 Dropout 行为差异（步 200-246）
9. **始终没找到 Bug 3**（blockIdx.x > 32 的 NaN 注入）

## 模型结构（多层归一化，增加定位难度）

```
Conv2d → BatchNorm2d → ReLU → Conv2d → GroupNorm → GELU → MaxPool → Flatten
    → Linear → FeatureNorm(LayerNorm) → ReLU
    → Linear → GroupNorm → Dropout
    → Linear → 输出
```

- `FeatureNorm` 是自定义类，内部调用 `F.layer_norm`
- 模型同时使用 BatchNorm2d、GroupNorm、LayerNorm 三种归一化
- 梯度检查显示多个层都有差异，无法精确定位到 LayerNorm

## 训练脚本（固定数据模式）

```bash
python train.py --steps 50 --seed 42 --device cuda --eval_fixed_data
```

- `--eval_fixed_data`：用固定 seed 生成训练数据（使 CPU/CUDA 结果可比较）
- 训练后输出 `accuracy {correct} {total}` 和 `final_accuracy`
- 小 batch（16）：无 NaN，accuracy ~18%（Bug 1+2）
- 大 batch（64）：NaN，accuracy ~10%（Bug 3）

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `nan_to_num` / `clip_grad` / `isnan` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试（GPU 应比 CPU 快 1.5x+） |
| 修改 train.py | 检查关键内容（ImageClassifier）是否还在 |
| 修改 model.py | 检查关键内容（FeatureNorm / F.layer_norm / BatchNorm / GroupNorm）是否还在 |
| 用 git diff/show 找答案 | `.git` 目录已被删除 |

## Oracle

`solution/solve.sh`：
```bash
# Bug 1: f_grad_input += ... → f_grad_input -= ...
sed -i '/T_ACC f_grad_input = fH \* gamma_val \* dy;/{n;s/f_grad_input += .../f_grad_input -= .../}' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu

# Bug 2+3: 恢复 forward 代码
# 删除条件 eps 和 NaN 注入
python3 -c "
old = '''T_ACC _eps = (wd.sigma2 < T_ACC(0.1)) ? T_ACC(0.01) : eps;
    T_ACC rstd_val = rsqrt(wd.sigma2 + _eps);
    if (blockIdx.x > 32) rstd_val = T_ACC(0.0) / T_ACC(0.0);'''
new = 'T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);'
content = content.replace(old, new)
" /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu

cd /build/pytorch/build && ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

## 资源需求

- **GPU**: 1× RTX A6000（4GB+）
- **fat base 构建**: 1-2 小时（一次性）
- **task1 增量构建**: 5-15 分钟
- **训练验证**: 2-3 分钟
- **磁盘**: ~30GB（fat base） + ~1GB（task1）

## 踩坑记录

### 1. 单一 bug 太容易被秒

**问题**：单一符号翻转 bug，Kimi 41 步就修完了。

**解决**：用复合 bug（3 个 bug 在不同条件下触发），Kimi 246 步只修了 2/3。

### 2. eps * N 类 bug 对训练没影响

**问题**：`eps * 1000.0f` 甚至 `eps * 100000.0f`，模型仍然能 100% accuracy。

**原因**：模型有 BatchNorm 和 GroupNorm，能补偿 LayerNorm 的弱点。

**解决**：用符号翻转（影响梯度方向），而非 eps 修改（只影响归一化强度）。

### 3. CPU+GPU 同时改的方案行不通

**问题**：想让 CPU 和 GPU 都有同样的 bug，使梯度检查无法检测。

**原因**：模型太鲁棒，即使 LayerNorm 完全坏了，其他归一化层能补偿。

**解决**：用 GPU-only bug + 梯度检查测试（CPU vs CUDA 对比）。

### 4. 诱饵用注释太容易排除

**问题**：诱饵用纯注释，agent 可以直接忽略。

**解决**：诱饵用看起来像真 bug 的代码（如 `float eps = 0.01f;`），但放在不影响功能的位置。

### 5. .git 必须删除

**问题**：agent 用 `git show <commit>` 直接看到所有改动。

**解决**：Dockerfile 最后加 `rm -rf /build/pytorch/.git`。

### 6. 多层归一化增加定位难度

**问题**：模型只有 LayerNorm，agent 通过梯度对比直接定位到目标。

**解决**：模型同时使用 BatchNorm + GroupNorm + LayerNorm，梯度检查显示多个层都有差异。

### 7. test.sh 必须测试多种场景

**问题**：单一测试场景（如只测 accuracy）无法覆盖所有 bug。

**解决**：测试多种场景（小 batch accuracy + 大 batch 无 NaN），每个场景覆盖不同的 bug。

### 8. Bug 触发条件必须不同

**问题**：如果所有 bug 都在同一条件下触发，agent 修一个就全修了。

**解决**：每个 bug 的触发条件不同（所有数据 / 方差小 / blockIdx 大），agent 必须分别修复。

## 文件结构

```
task/
├── task.toml              ← 任务元数据
├── instruction.md         ← 发给 agent 的 prompt（不暴露 bug 细节）
├── environment/
│   ├── Dockerfile         ← 构建 task1 镜像
│   └── Dockerfile.base    ← fat base 镜像定义
├── workspace/
│   ├── train.py           ← 训练脚本（支持 --eval_fixed_data）
│   └── model.py           ← 模型定义（FeatureNorm 封装 LayerNorm）
├── solution/
│   ├── inject_bug.py      ← 注入 3 个真 bug + 20 个诱饵
│   └── solve.sh           ← Oracle（修复 3 个 bug）
└── tests/
    └── test.sh            ← 判题脚本（多场景测试）
```

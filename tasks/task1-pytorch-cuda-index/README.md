# Task 1: PyTorch CUDA 复合 Bug → 训练异常

## 概述

在 PyTorch 源码的多个 CUDA kernel 中注入 **25+ 真 bug + 40+ 诱饵**，涵盖多种 bug 类型：
- **删除型**：删除 `__syncthreads`，导致竞争条件
- **条件触发型**：只在特定输入下触发（方差范围、blockIdx 等）
- **数值精度型**：微小的缩放/偏移，不崩溃但影响训练质量
- **跨 kernel 依赖型**：bug 跨越多个函数，需要理解调用链

## Bug 设计策略

### 核心发现：删除比添加更难

| 策略 | 难度 | 原因 |
|------|------|------|
| 删除 `__syncthreads` | ⭐⭐⭐⭐⭐ | 看不到异常，需要理解并发语义 |
| 条件触发的符号翻转 | ⭐⭐⭐⭐ | 可能测试通过，大场景才失败 |
| 跨 kernel 标志位依赖 | ⭐⭐⭐⭐ | 需要理解 CUDA 执行模型 |
| 陷阱诱饵（修了反而有害） | ⭐⭐⭐ | 消耗注意力，修了就出错 |
| 数值缩放/偏移 | ⭐⭐⭐ | 不崩溃，需数值分析 |
| 普通诱饵（无害） | ⭐ | 最容易排除 |

### Bug 分类

| 类型 | 数量 | 说明 |
|------|------|------|
| 删除型 | 6 | 删除 `__syncthreads`（LN/GN/BN/SoftMax） |
| 条件触发型 | 11 | 符号翻转、均值偏移、eps 放大等 |
| 数值精度型 | 5 | Dropout scale、Gelu x_cube、PReLU 缩放等 |
| 跨 kernel 依赖型 | 3 | 依赖 `_ln_flag` 标志位 |
| 陷阱诱饵 | 9 | `&& True`、`assert`，删了就出错 |
| 普通诱饵 | 26 | 声明未使用变量、注释 |

## 实测结果

| 版本 | Kimi 步数 | Reward | 说明 |
|------|----------|--------|------|
| 单一符号翻转 | 41 | 1.0 | 被秒 |
| 3 个复合 bug | 246 | 0.15 | 只修了 2/3 |
| **25+ bug + 40+ 诱饵** | **91** | **1.0** | 用"霰弹枪策略"全修了 |

### 最新测试发现

Kimi 用了"霰弹枪策略"——修了所有看起来可疑的代码，包括诱饵。因为诱饵是无害的，修掉不影响测试结果，所以得了满分。

**加强方向**：
1. 陷阱诱饵——修了反而有害（已部分实现）
2. 更多删除型 bug（已增加到 6 个）
3. 更多真实 bug（已增加到 25+）

## 调试轨迹分析（Kimi，91 步）

Kimi 的调试路径：
1. 读 train.py / model.py / test.sh（步 0-5）
2. 对比 CPU vs CUDA 梯度，定位到 LayerNorm（步 5-15）
3. 用 grep 搜索 `_ln_flag`、`* 0.95`、`+ 0.05` 等可疑模式（步 15-25）
4. **批量修复所有可疑代码**（步 25-60）
5. 重编译 + 测试（步 60-91）

**关键发现**：Kimi 通过 grep 搜索可疑模式（如 `_ln_flag`、`* 0.95`）快速定位 bug，然后用"霰弹枪策略"全部修掉。

## 宿主机与容器目录映射

### 宿主机目录结构

```
agentic-xlang-bugfix/                          ← Docker build context
├── .secrets/                                  ← API keys
└── tasks/task1-pytorch-cuda-index/            ← SCRIPT_DIR
    ├── run.sh                                 ← 单次运行
    ├── calibrate.sh                           ← 多次校准
    ├── task/
    │   ├── workspace/                         ← 挂载到容器 /workspace
    │   │   ├── train.py
    │   │   └── model.py
    │   ├── tests/test.sh                      ← 挂载到容器 /task/tests
    │   ├── instruction.md                     ← 挂载到容器 /task/instruction.md
    │   ├── environment/
    │   │   ├── Dockerfile
    │   │   └── Dockerfile.base
    │   └── solution/
    │       ├── inject_bug.py                  ← 注入 bug
    │       ├── solve.sh                       ← 修复 bug（反转 inject_bug.py）
    │       └── oracle.sh                      ← 验证：buggy 失败 + fixed 通过
    └── trajectories/                          ← 轨迹输出
```

### Docker run 挂载关系

| 容器内路径 | 宿主机来源 | 挂载方式 |
|---|---|---|
| `/workspace/` | `task/workspace/` | 只读 |
| `/task/tests/test.sh` | `task/tests/test.sh` | 只读 |
| `/task/instruction.md` | `task/instruction.md` | 只读 |
| `/task/solution/` | `task/solution/` | 只读（oracle 时挂载） |
| `/build/pytorch/` | 镜像内置 | PyTorch 源码 |

### Docker 构建命令

```bash
cd ~/data/PKU/exploitbench/task/agentic-xlang-bugfix

# 构建 fat base（一次性，1-2 小时）
DOCKER_BUILDKIT=1 docker build \
  --secret id=ssh_key,src=.secrets/id_rsa \
  -t pytorch-2.5.0-fat-base \
  -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile.base .

# 构建 task1 镜像（增量，5-15 分钟）
docker build -t task1 \
  -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile .
```

### 测试命令

```bash
# Oracle 测试（验证 bug 有效 + 修复后通过）
./test_oracle.sh task1

# Kimi 测试
cd tasks/task1-pytorch-cuda-index
./run.sh
```

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 上网搜索 | `run.sh` 中 deny WebSearch/WebFetch |
| git 查看历史 | `.git` 目录已删除 |
| 修改 Python 文件 | `test.sh` 检查 model.py/train.py 完整性 |
| CPU 回退 | 性能测试（GPU 应比 CPU 快 1.5x+） |
| NaN 处理掩盖 | grep `nan_to_num` / `clip_grad` |
| 文件修改时间 | Dockerfile 中 `touch` 统一时间戳 |

## 踩坑记录

### 1. 单一 bug 太容易被秒

**问题**：单一符号翻转 bug，Kimi 41 步就修完了。

**解决**：用复合 bug（多个 bug 在不同条件下触发）。

### 2. 诱饵被修掉不影响测试

**问题**：Kimi 用"霰弹枪策略"修了所有可疑代码，包括诱饵。诱饵是无害的，修掉不影响测试。

**解决**：设计陷阱诱饵——`&& True`、`assert` 等，看起来可疑但必须保留。

### 3. 陷阱诱饵会干扰 bug pattern

**问题**：`* T_ACC(1)` 陷阱在 bug 注入后应用，会改变 bug 的 pattern，导致 solve.sh 无法反转。

**解决**：陷阱诱饵不能和 bug 修改同一行代码。

### 4. .git 必须删除

**问题**：agent 用 `git show` 直接看到所有改动。

**解决**：Dockerfile 最后加 `rm -rf /build/pytorch/.git`。

### 5. Docker 会用缓存

**问题**：修改 inject_bug.py 后，`docker build` 用缓存，不重新注入 bug。

**解决**：用 `docker build --no-cache` 强制重建。

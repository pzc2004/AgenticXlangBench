# Task 4: JAX Batching Rule 错误 → vmap+grad 梯度错误

## 概述

在 JAX 的 batching transform 中注入 **21+ 真 bug + 30+ 诱饵**，涵盖多种 bug 类型：
- **删除型**：删除 early return、条件检查，最难发现
- **维度偏移型**：batch_dims +1/-1，导致形状转置
- **条件反转型**：if 条件取反，逻辑完全错误
- **陷阱诱饵**：`&& True`、`assert`，删了就出错

## Bug 设计策略

### 核心发现：删除 early return 最难

| 策略 | 难度 | 原因 |
|------|------|------|
| 删除 early return | ⭐⭐⭐⭐⭐ | 看不到异常，需要理解控制流 |
| 删除条件检查 | ⭐⭐⭐⭐ | 看起来像优化，实际是保护 |
| batch_dims +1 | ⭐⭐⭐ | 有迹可循，对比可发现 |
| 条件反转 | ⭐⭐⭐ | 逻辑错误，需要理解语义 |
| 陷阱诱饵 | ⭐⭐⭐ | 消耗注意力 |
| 普通诱饵 | ⭐ | 最容易排除 |

### Bug 分类

| 类型 | 数量 | 说明 |
|------|------|------|
| 删除型 | 8 | 删除 early return、src==dst 检查、nzs_out 过滤等 |
| 维度偏移型 | 7 | batch_dims +1、bdim_out +1、axes -1 等 |
| 条件反转型 | 5 | fancy check、isinstance、bdim None 等 |
| lax.py batching | 3 | reshape dims、transpose perm、res_bdim |
| slicing.py batching | 2 | gather bdim、offset_dims |
| ad.py | 2 | nzs_out、is_vjp |
| 陷阱诱饵 | 15 | `&& True`、`assert` |
| 普通诱饵 | 23 | 注释、调试代码 |

## 实测结果

| 版本 | Kimi 步数 | Reward | 说明 |
|------|----------|--------|------|
| 2 个 bug（原始） | 50 | 1.0 | 被秒 |
| 50 个 bug + 100 诱饵 | 299 | 1.0 | 用"霰弹枪策略"全修了 |
| **21+ bug + 30+ 诱饵** | 待测 | 待测 | 加了删除型 bug + 陷阱诱饵 |

### Kimi 调试路径（299 步版本）

1. 跑 test_vmap.py → 看到 XLA shape mismatch 错误
2. 写调试脚本追踪 `BatchTrace.process_primitive`
3. 测试 `vmap(sin)(x).shape` → 发现形状被转置 (4,8) → (8,4)
4. 定位到 `vectorized_batcher` 和 `process_primitive`
5. 用"霰弹枪策略"修了所有可疑代码

**关键发现**：Kimi 通过 `vmap(sin)(x).shape` 快速验证形状，然后 grep 搜索 `batch_dims` 相关代码，批量修复。

## 宿主机与容器目录映射

### 宿主机目录结构

```
agentic-xlang-bugfix/                          ← Docker build context
├── .secrets/                                  ← API keys
└── tasks/task4-jax-vmap-batch/                ← SCRIPT_DIR
    ├── run.sh                                 ← 单次运行
    ├── calibrate.sh                           ← 多次校准
    ├── task/
    │   ├── workspace/                         ← 挂载到容器 /workspace
    │   │   └── test_vmap.py
    │   ├── tests/test.sh                      ← 挂载到容器 /task/tests
    │   ├── instruction.md                     ← 挂载到容器 /task/instruction.md
    │   ├── environment/
    │   │   ├── Dockerfile
    │   │   └── Dockerfile.base
    │   └── solution/
    │       ├── inject_bug.py                  ← 注入 bug（支持 --reverse）
    │       ├── solve.sh                       ← 调用 inject_bug.py --reverse
    │       └── oracle.sh                      ← 验证
    └── trajectories/                          ← 轨迹输出
```

### Docker run 挂载关系

| 容器内路径 | 宿主机来源 | 挂载方式 |
|---|---|---|
| `/workspace/` | `task/workspace/` | 只读 |
| `/task/tests/test.sh` | `task/tests/test.sh` | 只读 |
| `/task/instruction.md` | `task/instruction.md` | 只读 |
| `/task/solution/` | `task/solution/` | 只读（oracle 时挂载） |
| `/build/jax/` | 镜像内置 | JAX 源码 |

### Docker 构建命令

```bash
cd ~/data/PKU/exploitbench/task/agentic-xlang-bugfix

# 构建 fat base（一次性）
DOCKER_BUILDKIT=1 docker build \
  --secret id=ssh_key,src=.secrets/id_rsa \
  -t jax-fat-base \
  -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile.base .

# 构建 task4 镜像
docker build --no-cache -t task4 \
  -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .
```

### 测试命令

```bash
# Oracle 测试
./test_oracle.sh task4

# Kimi 测试
cd tasks/task4-jax-vmap-batch
./run.sh
```

## inject_bug.py 设计

### 支持 --reverse

`inject_bug.py` 支持 `--reverse` 参数，用于修复 bug：

```bash
# 注入 bug
python3 inject_bug.py

# 修复 bug（反转所有修改）
python3 inject_bug.py --reverse
```

实现方式：
```python
REVERSE = "--reverse" in sys.argv

def apply_bug(filepath, old, new, name):
    if REVERSE:
        old, new = new, old  # 反转
    ...
```

### solve.sh 直接调用 inject_bug.py

```bash
#!/bin/bash
python3 /task/solution/inject_bug.py --reverse
```

不需要手动定义每个 bug 的修复模式。

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 上网搜索 | `run.sh` 中 deny WebSearch/WebFetch |
| git 查看历史 | `.git` 目录已删除 |
| 绕过 vmap | `test.sh` 检查是否使用 `jax.vmap` |
| 修改测试脚本 | `test.sh` 检查完整性 |
| 文件修改时间 | Dockerfile 中 `touch` 统一时间戳 |

## 踩坑记录

### 1. 陷阱诱饵会干扰 bug pattern

**问题**：`* 1` 陷阱在 bug 注入后应用，会改变 bug 的 pattern（如 `batch_dims[0] + 1` 变成 `batch_dims[0] * 1 + 1`），导致 `--reverse` 无法修复。

**解决**：禁用 `* 1` 陷阱，只用 `&& True` 和 `assert` 陷阱。

### 2. Docker 用缓存导致 bug 没注入

**问题**：修改 inject_bug.py 后，`docker build` 用缓存，不重新注入。

**解决**：用 `docker build --no-cache` 强制重建。

### 3. sed 命令会破坏函数定义

**问题**：用 `sed` 批量替换时，误伤了函数定义（如 `def inject_deletion_bugs():` 变成 `def inject_def inject_deletion_bugs():_bugs():`）。

**解决**：用 Python 的 `py_compile` 检查语法，避免 sed 的坑。

### 4. solve.sh 只需调用 inject_bug.py --reverse

**问题**：之前手动定义每个 bug 的修复模式，容易遗漏。

**解决**：inject_bug.py 支持 `--reverse`，solve.sh 直接调用。

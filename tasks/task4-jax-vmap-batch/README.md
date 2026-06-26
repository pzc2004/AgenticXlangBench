# Task 4: JAX Batching Rule 错误 → vmap+grad 梯度错误

## 概述

在 JAX 的 batching transform 中注入 **26 个真 bug**，涵盖多种 bug 类型：
- **删除型**：删除 early return、条件检查，最难发现
- **维度偏移型**：batch_dims +1/-1，导致形状转置
- **条件反转型**：if 条件取反，逻辑完全错误

当前状态：bug 注入机制已改为 `unified diff patch + 注入校验`，但 `test.sh` 对单个 bug 的覆盖度不足，正在加强测试中。

## Bug 设计策略

### 核心发现：删除 early return 最难

| 策略 | 难度 | 原因 |
|------|------|------|
| 删除 early return | ⭐⭐⭐⭐⭐ | 看不到异常，需要理解控制流 |
| 删除条件检查 | ⭐⭐⭐⭐ | 看起来像优化，实际是保护 |
| batch_dims +1 | ⭐⭐⭐ | 有迹可循，对比可发现 |
| 条件反转 | ⭐⭐⭐ | 逻辑错误，需要理解语义 |

### Bug 分类

| 类型 | 数量 | 说明 |
|------|------|------|
| 删除型 | 6 | 删除 early return、src==dst 检查、nzs_out 过滤等 |
| 维度偏移型 | 6 | batch_dims +1、bdim_out +1、axes -1 等 |
| 条件反转型 | 3 | fancy check、broadcast size、nzs_out 等 |
| lax.py batching | 8 | reshape/transpose/concat/select_n/reduce/dot_general 等 |
| slicing.py batching | 2 | gather bdim、offset_dims |
| ad.py | 2 | nzs_out、is_vjp |

## 注入与验证机制（新版）

### 文件说明

| 文件 | 作用 |
|------|------|
| `solution/decoys.patch` | 诱饵 patch（clean → +decoys），build 时固定 |
| `solution/generate_decoys.py` | 从 JAX 源码生成 `decoys.patch` |
| `solution/bugs.patch` | 完整 bug patch（+decoys → +decoys+bugs） |
| `solution/per_bug_patches/Bug_*.patch` | 单个 bug patch（+decoys → +decoys+单 bug） |
| `solution/generate_per_bug_patches.py` | 从 JAX 源码重新生成上述 patch |
| `solution/inject_bug.py` | 应用/回退 `bugs.patch`，带注入校验 |
| `solution/solve.sh` | 调用 `inject_bug.py --reverse` 修复 |
| `solution/oracle.sh` | 验证 buggy 版失败、修复后通过 |
| `solution/oracle_per_bug.py` | 逐个验证每个 bug 都能被检测到 |

### 诱饵设计

- **目标**：比真 bug 更像 bug，分散 Kimi 注意力，占满上下文。
- **最强诱饵**：看起来可疑、但**改动它反而会引入新 bug** 的代码（如必要 guard、dtype cast、边界检查）。
- **生效方式**：`decoys.patch` 在 Dockerfile 中于 build 时应用，solve.sh 不会回退，因此 Kimi 始终看到诱饵。
- **示例**：在 `batching.py`/`ad.py`/`lax.py`/`slicing.py` 中加入带 `FIXME`/`WARNING`/`BUG_CANDIDATE` 注释的 assert、类型检查、axis 边界 guard 等。

### inject_bug.py

```bash
# 注入 bug
python3 /task/solution/inject_bug.py

# 修复 bug
python3 /task/solution/inject_bug.py --reverse
```

实现方式：内部调用 `patch -d /build/jax -p0 < bugs.patch`，并通过已知 bug marker 校验是否成功。

### solve.sh

```bash
#!/bin/bash
set -e
python3 /task/solution/inject_bug.py --reverse
bash /task/tests/test.sh
```

## 宿主机与容器目录映射

### 宿主机目录结构

当前会话运行在宿主机上的容器内：
- 容器内项目路径：`/workspace/work/PKU/exploitbench/`
- 宿主机对应路径：`<PROJECT_PATH>/`

```
agentic-xlang-bugfix/                          ← Docker build context
├── .secrets/                                  ← API keys / SSH key
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
    │       ├── decoys.patch                     ← 诱饵 patch
    │       ├── generate_decoys.py               ← 诱饵 patch 生成器
    │       ├── bugs.patch                       ← 完整 bug patch
    │       ├── per_bug_patches/                 ← 单个 bug patch
    │       ├── generate_per_bug_patches.py      ← patch 生成器
    │       ├── inject_bug.py                    ← patch 应用/回退
    │       ├── solve.sh                         ← 修复脚本
    │       ├── oracle.sh                        ← oracle 验证
    │       └── oracle_per_bug.py                ← per-bug oracle
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
ssh pzc@162.105.87.147
cd <PROJECT_PATH>/task/agentic-xlang-bugfix

# 构建 fat base（一次性）
DOCKER_BUILDKIT=1 docker build \
  --secret id=ssh_key,src=.secrets/id_rsa \
  -t jax-fat-base \
  -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile.base .

# 构建 task4 镜像
DOCKER_BUILDKIT=1 docker build \
  --secret id=ssh_key,src=.secrets/id_rsa \
  -t task4 \
  -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .
```

### 测试命令

```bash
# 在容器内直接判题
bash /task/tests/test.sh

# Oracle 测试
bash /task/solution/oracle.sh

# Per-bug Oracle
python3 /task/solution/oracle_per_bug.py

# Kimi 测试（宿主机执行）
cd tasks/task4-jax-vmap-batch
./run.sh
```

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 上网搜索 | `run.sh` 中 deny WebSearch/WebFetch |
| git 查看历史 | `.git` 目录已删除 |
| 绕过 vmap | `test.sh` 检查是否使用 `jax.vmap` |
| 修改测试脚本 | `test.sh` 检查完整性 |
| 文件修改时间 | Dockerfile 中 `touch` 统一时间戳 |

## 当前问题与下一步

### test.sh 覆盖度不足

per-bug oracle 显示：绝大多数 bug 单独存在时分数仍为 1.0，说明当前 `test_vmap.py` 没有触发这些 bug 对应的代码路径。

下一步：针对每个 bug 设计最小触发用例，重写 `test_vmap.py` / `test.sh`。

## 踩坑记录

### 1. 文本匹配注入非常脆弱

**问题**：用 `str.replace` 注入 bug，PyTorch/JAX 源码一换行或版本升级就失效，且多处相同代码容易改错。

**解决**：改为 `unified diff patch` 注入，从实际镜像源码生成 patch，应用时带校验。

### 2. patch 路径与 JAX_PKG 不一致

**问题**：`jax._src.lax.slicing.__file__` 得到 `/build/jax/jax/_src/...`，所以 `JAX_PKG=/build/jax/jax`，但 patch 路径是 `jax/_src/...`，需要用 `patch -d /build/jax` 而不是 `patch -d /build/jax/jax`。

**解决**：`inject_bug.py` 中 patch base 取 `os.path.dirname(JAX_PKG)`。

### 3. Docker 用缓存导致 bug 没注入

**问题**：修改 inject_bug.py 或 bugs.patch 后，`docker build` 用缓存，不重新注入。

**解决**：用 `docker build --no-cache` 强制重建，或在 Dockerfile 中加入 build stamp。

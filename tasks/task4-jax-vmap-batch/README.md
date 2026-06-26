# Task 4: JAX Batching Rule 错误 → vmap+grad 梯度错误

## 概述

在 JAX 的 batching transform 中注入 **26 个真 bug**，涵盖多种 bug 类型：
- **删除型**：删除 early return、条件检查，最难发现
- **维度偏移型**：batch_dims +1/-1，导致形状转置
- **条件反转型**：if 条件取反，逻辑完全错误

当前状态：bug 注入机制已改为 `unified diff patch + 注入校验`；`test_vmap.py` 已加强到 76 个有效测试（fixed 100% 通过，buggy 0% 通过），Oracle 与 per-bug Oracle 均已通过。

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
| 维度偏移型 | 6 | batch_dims +1、bdim_out +1、axes -1、expand_dims 输出 bdim 等 |
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
| `solution/generate_bugs_patch.py` | 从 `decoys.patch` + `per_bug_patches` 生成 `bugs.patch` |
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

实现方式：内部调用 `patch -d /build/jax -p0 < bugs.patch`，并通过 5 个已知 bug marker 校验是否成功。

### 生成 patch

如果修改了某个 bug 的定义（`generate_per_bug_patches.py` 中的 `BUGS` 列表），需要重新生成 patch：

```bash
# 1. 启动 fat-base 容器（内含 clean JAX 源码）
docker run -d -v $(pwd)/tasks/task4-jax-vmap-batch/task/solution:/task/solution \
  --name task4_gen jax-fat-base sleep 3600

# 2. 重新生成 per-bug patches（基于 decoys 状态）
docker exec task4_gen python3 /task/solution/generate_per_bug_patches.py \
  /build/jax /task/solution/decoys.patch /task/solution/per_bug_patches

# 3. 重新生成 bugs.patch
docker exec task4_gen python3 /task/solution/generate_bugs_patch.py /build/jax \
  /task/solution/decoys.patch /task/solution/bugs.patch

# 4. 清理
docker stop task4_gen && docker rm task4_gen
```

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

## 当前状态

- `test_vmap.py` 已二阶段加强：覆盖 30+ 种操作/模式 × 多种 shape/axis = **455 个有效测试用例**。
  - 一阶段：支持非零 `in_axes`（axis=0/1/2），覆盖 batch 维在不同位置的情况。
  - 二阶段：新增混合 batched/unbatched 参数、JVP/linearize、显式 reshape dimensions、正轴 transpose、identity moveaxis、ragged dot_general、显式 gather、链式操作等定向测试。
- 启用 **fail-fast**：遇到第一个失败立即退出，避免 buggy 版本在大量测试上卡住。
- Oracle 结果：buggy 版 0.10，fixed 版 1.0。
- Per-bug Oracle 结果：**26/26 个 bug 单独注入时均能被检测到**（`oracle_per_bug.py` 已加 60 秒超时防卡死，判定阈值 `< 1.0`）。

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

### 4. 过于破坏性的 bug 会让所有 vmap 失效

**问题**：某个 bug 把 `if p in fancy_primitive_batchers` 改成 `if p not in fancy_primitive_batchers`，导致任何 vmap 调用都报 `Batching rule for 'jit' not implemented`，模型一步都走不通。

**解决**：把该 bug 替换为对 `expand_dims_batcher` 输出 batch_dim 的轻微偏移，既能被测试捕获，又不会让框架完全崩溃。

### 5. skipped 测试不应计入 passed

**问题**：早期 `test_vmap.py` 把 `skipped`（维度不足跳过）也算进 `passed`，导致 buggy 版分数虚高（24%）。

**解决**：统计时 `skipped` 单独计数，仅 `passed / meaningful_total` 参与 accuracy 计算。

### 6. 个别 bug 会让测试无限卡住

**问题**：某些 batching rule bug（如 transpose axes 错误、reduce dim 错误）不会立即报错，而是让 JAX 进入极慢/无限的编译或执行路径；跑大量测试时会被拖死。

**解决**：`test_vmap.py` 默认启用 fail-fast，遇到第一个失败立即退出；`oracle_per_bug.py` 对每个 bug 的测试设 60 秒超时，超时视为检测到 bug。

### 7. Dockerfile 删除了 solve/oracle 需要的脚本

**问题**：早期 Dockerfile 把 `inject_bug.py` 复制到 `/tmp` 并在 RUN 后删除，导致容器内没有 `/task/solution/inject_bug.py`，`solve.sh` / `oracle.sh` / `oracle_per_bug.py` 无法运行。

**解决**：Dockerfile 中将 `inject_bug.py` 保留在 `/task/solution/`，并 COPY 整个 `solution/` 目录到容器，确保所有脚本和 patch 都可用。

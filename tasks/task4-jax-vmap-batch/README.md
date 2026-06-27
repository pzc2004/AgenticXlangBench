# Task 4: JAX Batching Rule 错误 → vmap+grad 梯度错误

## 概述

在 JAX 的 batching transform 中注入 **26 个真 bug**，涵盖多种 bug 类型：

- **删除型**：删除 early return、条件检查、nzs_out 过滤等，最难发现
- **维度偏移型**：batch_dims +1/-1、result_batch_dim +1、operand_bdim +1 等，导致形状/转置错误
- **条件反转型**：if 条件取反、nzs_out 判断取反，逻辑完全错误
- **Zero tangent 型**（AD）：linearize 中 Zero / non-Zero 判断反转，JVP/VJP 路径出错

当前状态：
- bug 注入机制为 `unified diff patch + fuzz=0 注入校验`
- `test_vmap.py` 已加强到 **245 个有效测试**（fixed 100% 通过，buggy 0% 通过）
- Oracle 与 per-bug Oracle 均已通过
- 已启用 **protected grading**：agent 非 root，判分脚本锁在 `/opt/judge`，只能执行 `grade` 拿总分

## Bug 设计策略

### 核心发现：删除型与 AD 类 bug 最难

| 策略 | 难度 | 原因 |
|------|------|------|
| 删除 early return / 条件检查 | ⭐⭐⭐⭐⭐ | 看不到异常，需要理解控制流 |
| AD/Linearize Zero tangent 反转 | ⭐⭐⭐⭐⭐ | 只在 JVP/VJP/linearize 路径暴露 |
| Lax batch rule 维度偏移 | ⭐⭐⭐⭐ | 跨多个 primitive，需对齐 batch dim |
| batch_dims +1 | ⭐⭐⭐ | 有迹可循，对比可发现 |
| 条件反转 | ⭐⭐⭐ | 逻辑错误，需要理解语义 |

### Bug 分类

| 类型 | 数量 | 所在文件 | 说明 |
|------|------|----------|------|
| Batching 维度偏移/轴交换/删除 | 13 | `jax/_src/interpreters/batching.py` | matchaxis、vectorized_batcher、reducer_batcher、expand_dims_batcher 等 |
| Lax batch rule 维度偏移 | 8 | `jax/_src/lax/lax.py` | reshape/transpose/concat/select_n/reduce/dot_general/iota 等 |
| AD/Linearize Zero tangent | 3 | `jax/_src/interpreters/ad.py` | linearize 中 nzs_out、is_vjp、out_zeros 等 |
| Slicing batch rule | 2 | `jax/_src/lax/slicing.py` | gather bdim、offset_dims |

## 注入与验证机制

### 文件说明

| 文件 | 作用 |
|------|------|
| `solution/decoys.patch` | 诱饵 patch（clean → +decoys），build 时固定 |
| `solution/generate_decoys.py` | 从 JAX 源码生成 `decoys.patch` |
| `solution/bugs.patch` | 完整 bug patch（+decoys → +decoys+bugs） |
| `solution/generate_bugs_patch.py` | 从 `decoys.patch` + `per_bug_patches` 生成 `bugs.patch` |
| `solution/per_bug_patches/Bug_*.patch` | 单个 bug patch（+decoys → +decoys+单 bug） |
| `solution/generate_per_bug_patches.py` | 从 JAX 源码重新生成上述 patch，是 BUGS 的 source of truth |
| `solution/inject_bug.py` | 应用/回退 `bugs.patch`，带 fuzz=0 注入校验和 marker 校验 |
| `solution/solve.sh` | 调用 `inject_bug.py --reverse` 修复（开发/oracle 用，agent 看不到） |
| `solution/oracle.sh` | 验证 buggy 版失败、修复后通过（开发用，agent 看不到） |
| `solution/oracle_per_bug.py` | 逐个验证每个 bug 都能被检测到（开发用） |
| `solution/analyze_trajectory.py` | 自动分析 Kimi 轨迹：修对/漏修/难度分级/反 hack 行为 |
| `environment/grade.c` | setuid-root 判分入口，agent 执行 `grade` 只拿到总分 |

### inject_bug.py

```bash
# 注入 bug
python3 /task/solution/inject_bug.py

# 修复 bug
python3 /task/solution/inject_bug.py --reverse
```

实现方式：
1. 调用 `patch -d /build/jax -p0 -F0 < bugs.patch`（fuzz=0，任何 hunk 不匹配就 reject）
2. 检查没有 `.rej` 残留
3. 用 5 个已知 bug marker 二次校验是否成功

### 生成 patch

如果修改了某个 bug 的定义（`generate_per_bug_patches.py` 中的 `BUGS` 列表），需要重新生成 patch：

```bash
# 1. 启动 fat-base 镜像（内含 clean JAX 源码）
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

## Protected Grading（判分隔离）

为了防止 agent 直接读答案/测试，task4 启用了 task1 同款的 protected grading：

| 机制 | 实现 | 效果 |
|------|------|------|
| 非 root agent | Dockerfile 创建 `agent` 用户 uid 1500 | agent 无法读取 root-only 文件 |
| 判分脚本锁在 `/opt/judge` | `root:root 0700` | `cat /opt/judge/test.sh` → Permission denied |
| `grade` setuid | `environment/grade.c` 编译为 `/usr/local/bin/grade`，权限 4755 | agent 执行 `grade` 只拿到 `score=X.XX` |
| `/task/tests` 指向 `/opt/judge` | symlink | 测试路径对 agent 不可读 |
| `/task/solution` 对 agent 不可见 | run.sh 挂载空目录覆盖 | agent 看不到 `bugs.patch` / `solve.sh` |
| `.git` 已清理 | Dockerfile 删除 `/build/jax/.git` | `git diff` / `git log` 不可用 |
| 统一时间戳 | Dockerfile `touch` 所有 `.py` | 无法通过 `stat` / `ls -lt` 定位注入点 |

## Docker 构建与运行

### 目录结构

```
agentic-xlang-bugfix/                          ← Docker build context
├── .secrets/                                  ← API keys / SSH key
└── tasks/task4-jax-vmap-batch/                ← SCRIPT_DIR
    ├── run.sh                                 ← 单次运行
    ├── calibrate.sh                           ← 多次校准
    ├── trajectories/                          ← 轨迹输出
    └── task/
        ├── workspace/                         ← 运行时挂载到 /workspace
        │   └── test_vmap.py
        ├── tests/test.sh                      ← 判分脚本，build 时复制到 /opt/judge
        ├── tests/test_vmap.py                 ← 真实测试，build 时复制到 /opt/judge
        ├── instruction.md                     ← 运行时挂载到 /task/instruction.md
        ├── environment/
        │   ├── Dockerfile
        │   ├── Dockerfile.base
        │   └── grade.c                        ← setuid-root 判分入口
        └── solution/
            ├── decoys.patch
            ├── bugs.patch
            ├── per_bug_patches/
            ├── inject_bug.py
            ├── generate_per_bug_patches.py
            ├── generate_bugs_patch.py
            ├── solve.sh
            ├── oracle.sh
            ├── oracle_per_bug.py
            └── analyze_trajectory.py          ← 轨迹分析
```

### Docker run 挂载关系（agent 运行时）

| 容器路径 | 来源 | 挂载方式 | 说明 |
|---|---|---|---|
| `/workspace/` | `task/workspace/` | 只读 | 开发自测 stub |
| `/task/instruction.md` | `task/instruction.md` | 只读 | agent prompt |
| `/task/solution/` | 空目录 | 只读 | 隐藏 solution，防 patch 作弊 |
| `/task/tests/` | `/opt/judge` symlink | root-only | 真实测试对 agent 不可读 |
| `/trajectories/` | `trajectories/$RUN_ID/` | 可写 | 轨迹输出 |
| `/home/agent/.kimi-code/config.toml` | 运行时生成 | 只读 | 权限规则（deny WebSearch/WebFetch） |

### Docker 构建命令

```bash
# 在项目根目录执行
cd <项目路径>

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

Agent 运行时：

```bash
# agent 只能执行 grade 拿总分
grade
```

开发/oracle 用（需 root + 挂载 solution）：

```bash
# Oracle 测试
docker run --rm --gpus all --user 0 \
  -v $(pwd)/tasks/task4-jax-vmap-batch/task/solution:/task/solution:ro \
  task4 bash /task/solution/oracle.sh

# Per-bug Oracle
docker run --rm --gpus all --user 0 \
  -v $(pwd)/tasks/task4-jax-vmap-batch/task/solution:/task/solution:ro \
  task4 python3 /task/solution/oracle_per_bug.py
```

## 反 hack 措施

| Hack 路径 | 检测/阻断方法 |
|---|---|
| 读 `bugs.patch` / `solve.sh` | `/task/solution` 挂载为空目录 |
| 读 `/task/tests/test.sh` | `/opt/judge` root-only 0700 |
| 读 `/task/tests/test_vmap.py` | `/opt/judge` root-only 0700 |
| 读 `/opt/judge` 判分逻辑 | `/opt/judge` root-only；`grade` 只回显总分 |
| 上网搜索 | `run.sh` deny WebSearch/WebFetch |
| git 查看历史 | `.git` 目录已删除 |
| 绕过 vmap | `test.sh` 检查是否使用 `jax.vmap` / `jax.grad` |
| 修改测试脚本 | `test.sh` 检查完整性；agent 无写权限 |
| 文件修改时间定位 | Dockerfile 中 `touch` 统一时间戳 |
| 通过 `/logs/verifier` 偷分 | `/logs/verifier` root-only 0700 |

## 当前状态

- `test_vmap.py` 已加强到 **245 个有效测试用例**，覆盖单参数、多参数、零参数、常量输出、pytree 输出、linalg、vjp、ragged_dot、多种 shape/axis 组合。
- 启用 **fail-fast**：遇到第一个失败立即退出，避免 buggy 版本卡住。
- Oracle 结果：buggy 版 0.10，fixed 版 **1.0**。
- Per-bug Oracle 结果：**26/26 个 bug 单独注入时均能被检测到**。
- Protected grading 验证通过：agent（uid 1500）无法读取 `/opt/judge/test.sh`、`/task/tests/test_vmap.py`、`/task/solution/bugs.patch`。
- 轨迹分析工具 `analyze_trajectory.py` 已就位，可输出：修对/漏修、难度分级、修复时间线、反 hack 行为检测。

## Kimi 测试结果

| 时间 | 设置 | Reward | Turns | Tool calls | 真 bug 修对 | 备注 |
|---|---|---|---|---|---|---|
| 2026-06-27 | 未加 protected grading | **1.0** | ~10 | ~10 | 26/26 | agent 找到 `/task/solution/bugs.patch` 并用 `patch -R` 作弊 |
| 2026-06-28 | protected grading + 3600s timeout | **0.10** | 341 | 341 | 9/26 | 无法读 patch/测试；AD/删除型/Lax offset 类 0% 修复；时间到被迫停止 |

结论：**protected grading 有效阻止作弊后，task4 对 Kimi 构成实质性难度**。

## 踩坑记录

### 1. 文本匹配注入非常脆弱

**问题**：用 `str.replace` 注入 bug，PyTorch/JAX 源码一换行或版本升级就失效，且多处相同代码容易改错。

**解决**：改为 `unified diff patch` 注入，从实际镜像源码生成 patch，应用时 fuzz=0 + marker 校验。

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

### 7. Agent 直接读到 `bugs.patch` 作弊

**问题**：早期 Dockerfile COPY 整个 `solution/` 进镜像，agent 发现 `bugs.patch` 后直接 `patch -R` 拿 1.0。

**解决**：
- Dockerfile 不再 COPY `solution/`
- run.sh 用空目录挂载覆盖 `/task/solution`
- 真实判分脚本锁进 `/opt/judge`（root-only）
- agent 只能执行 `grade` 拿总分

### 8. 输出缓冲导致 timeout 误判

**问题**：`test.sh` 用管道捕获 python 输出时，python 缓冲不刷新，timeout 杀死前看不到任何输出。

**解决**：`PYTHONUNBUFFERED=1 python -u` 强制无缓冲。

### 9. `linearize_subtrace_2` 是死代码

**问题**：Bug 7/18/19 最初改在 `linearize_subtrace_2` 上，但该函数实际未执行，导致 oracle 检测不到。

**解决**：把改动移到顶层 `linearize` 函数，确保在真实 JVP/VJP 路径触发。

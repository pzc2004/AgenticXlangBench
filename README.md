# AgenticXlangBench

面向 agentic RL 后训练的跨语言跨抽象层 bug-fix 任务集 —— 底层 bug,上层症状,延迟显现。

## 核心设计

**延迟显现(Delayed Manifestation)**:bug 在底层(C / CUDA / Rust / ASM),症状在上层(Python / SQL / CLI),两者之间跨越多个抽象层和数百个操作。agent 需要从症状反推、穿越语言边界、理解框架内部,才能定位根因。

```
bug 代码执行(CUDA kernel / C++ / Rust)
    ↓ 正常返回,没有报错
大量下游操作(autograd / 图像管线 / 查询执行...)
    ↓ 每一步都"看起来正常"
症状出现(NaN / accuracy 下降 / 结果偏差 / segfault)
    ↓ 症状远离 bug,stack trace 指向错误位置
```

## 任务总览

| # | 任务 | Bug 层 | 症状 | 延迟机制 | 语言 |
|---|---|---|---|---|---|
| 1 | PyTorch CUDA index | CUDA | 训练 NaN | 累积误差 | Python+CUDA |
| 2 | PyTorch autograd sign | C++ | accuracy↓ | 累积误差 | Python+C++ |
| 3 | NumPy BLAS register | x86 ASM | 结果偏差 | 触发条件 | Python+ASM |
| 4 | JAX vmap batching | Python 内部 | 梯度错 | 级联传播 | Python+内部 |
| 5 | CPython refcount | C | segfault | 概率性 | Python+C |
| 6 | TF shape inference | C++ | shape err | 级联传播 | Python+C++ |
| 7 | cuDNN conv backward | CUDA | 效果差 | 累积+触发 | Python+CUDA |
| 8 | Rust FFI lifetime | Rust | segfault | 概率性 | Python+Rust |
| 9 | PG executor NULL | C | 行数不对 | 触发条件 | SQL+C |
| 10 | Redis module write | C | 数据乱码 | 级联传播 | Redis+C |
| 11 | LLVM codegen | C++ | -O2 输出错 | 级联传播 | C+ASM |
| 12 | Python GIL race | C | 偶发错 | 概率性 | Python+C |
| 13 | OpenCV CUDA resize | CUDA | 像素偏差 | 级联传播 | Python+CUDA |
| 14 | NumPy dtype overflow | C | 溢出 | 触发条件 | Python+C |
| 15 | SQLite optimizer | C | 少行 | 触发条件 | SQL+C |

每道题的详细设计(bug 类型 / 延迟机制 / 为什么难 / anti-hack 措施 / oracle)见各任务目录下的 `README.md`。

## 最新进展

### 2026-06-25: Task 1 & Task 4 重大突破

**Task 1 (PyTorch CUDA)**:
- Bug 数量：3 → **25+ 真 bug + 40+ 诱饵**
- 新增删除型 bug（删 `__syncthreads`，最难发现）
- 新增陷阱诱饵（`&& True`、`assert`，修了反而有害）
- Oracle 测试通过：Buggy 0.15, Fixed 1.0
- Kimi 测试：91 步修完（用"霰弹枪策略"）

**Task 4 (JAX vmap)**:
- Bug 数量：2 → **21+ 真 bug + 30+ 诱饵**
- 新增删除型 bug（删 early return、条件检查）
- inject_bug.py 支持 `--reverse`，solve.sh 直接调用
- Oracle 测试通过：Buggy 0.10, Fixed 1.0
- Kimi 测试：299 步修完（用"霰弹枪策略"）

**关键发现**：
1. 删除代码比添加代码更难被 agent 发现
2. 陷阱诱饵（修了反而有害）是有效的反 hack 手段
3. "霰弹枪策略"（修所有可疑代码）是 agent 的常见行为

### 反 hack 措施汇总

| 措施 | 说明 |
|------|------|
| 禁止上网搜索 | `run.sh` 中 deny WebSearch/WebFetch |
| 删除 `.git` | 防止 `git diff` 看到改动 |
| 统一文件修改时间 | Dockerfile 中 `touch` 所有文件 |
| 陷阱诱饵 | `&& True`、`assert`，删了就出错 |
| Anti-hack 检查 | test.sh 中检查完整性 |

详见 `skills/anti-hack.md`。

## 快速开始

```bash
# SSH 到宿主机
ssh pzc@162.105.87.147

# 进入项目目录
cd ~/data/PKU/exploitbench/task/agentic-xlang-bugfix

# 构建镜像
docker build --no-cache -t task1 -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile .
docker build --no-cache -t task4 -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .

# Oracle 测试
./test_oracle.sh task1
./test_oracle.sh task4

# Kimi 测试
cd tasks/task1-pytorch-cuda-index && ./run.sh
cd tasks/task4-jax-vmap-batch && ./run.sh

# 校准难度(3 模型 × 3 seed)
./calibrate.sh 10
```

## 项目架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      开发环境 (容器内)                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Claude Code │  │   Skills    │  │   Memory    │         │
│  │  (AI 助手)  │  │ (自动化工具) │  │ (持久记忆)  │         │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘         │
│         │                │                                  │
│         ▼                ▼                                  │
│  ┌─────────────────────────────────────────┐               │
│  │           工作流 (Workflow)              │               │
│  │  generate-task → calibrate → select     │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      宿主机 (162.105.87.147)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Docker    │  │   Kimi Code │  │  轨迹存储   │         │
│  │  (构建镜像)  │  │  (测试 Agent)│  │ (trajectories)│        │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘         │
│         │                │                                  │
│         ▼                ▼                                  │
│  ┌─────────────────────────────────────────┐               │
│  │         评测管线 (Pipeline)              │               │
│  │  build → oracle → kimi → analyze        │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 自动化管线

#### 1. 任务生成 (`/generate-task`)

```
输入: CVE / bug 描述 / 框架
    ↓
Phase 1: 分析 bug 类型和触发条件
    ↓
Phase 2: 设计用户接口 (train.py / model.py)
    ↓
Phase 3: 设计诱饵 (陷阱诱饵 + 普通诱饵)
    ↓
Phase 4: 编写 inject_bug.py (支持 --reverse)
    ↓
Phase 5: 编写 solve.sh (调用 inject_bug.py --reverse)
    ↓
Phase 6: 编写 oracle.sh (验证 buggy 失败 + fixed 通过)
    ↓
Phase 7: 编写 test.sh (多场景 + anti-hack)
    ↓
Phase 8: 编写 Dockerfile (注入 bug + touch 统一时间戳)
    ↓
输出: tasks/taskN-<name>/ 完整目录
```

#### 2. 镜像构建

```bash
# Fat base (一次性, 1-2 小时)
DOCKER_BUILDKIT=1 docker build \
  --secret id=ssh_key,src=.secrets/id_rsa \
  -t pytorch-2.5.0-fat-base \
  -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile.base .

# Task 镜像 (增量, 5-15 分钟)
docker build --no-cache -t task1 \
  -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile .
```

**关键步骤**:
1. `COPY inject_bug.py /tmp/` — 复制注入脚本
2. `RUN python3 /tmp/inject_bug.py` — 注入 bug + 诱饵
3. `RUN find ... -exec touch {} +` — 统一文件修改时间
4. `RUN ninja -j32 lib/libtorch_cuda.so` — 重编译
5. `RUN rm -rf /build/pytorch/.git` — 删除 git 历史

#### 3. Oracle 测试

```bash
./test_oracle.sh task1
```

**流程**:
```
1. 启动容器 (挂载 workspace/tests/solution)
    ↓
2. 运行 test.sh (buggy 版本)
    ↓
3. 检查分数 < 1.0 (确认 bug 有效)
    ↓
4. 运行 solve.sh (调用 inject_bug.py --reverse)
    ↓
5. 运行 test.sh (fixed 版本)
    ↓
6. 检查分数 = 1.0 (确认修复正确)
```

#### 4. Kimi 测试

```bash
cd tasks/task1-pytorch-cuda-index
./run.sh
```

**流程**:
```
1. 生成 kimi_config.toml (API key + 权限规则)
    ↓
2. 启动容器 (挂载 workspace/tests/instruction)
    ↓
3. 运行 kimi -p "任务 prompt"
    ↓
4. 保存 trajectory.jsonl (完整轨迹)
    ↓
5. docker commit (保存修复状态)
    ↓
6. 在快照容器中运行 test.sh
    ↓
7. 输出 result.jsonl (reward/turns/tool_calls)
```

#### 5. 轨迹分析

```bash
# 查看结果
cat trajectories/*/result.jsonl

# 分析轨迹
python3 -c "
import json
with open('trajectories/*/trajectory.jsonl') as f:
    lines = [json.loads(l) for l in f]
    # 分析 agent 的调试路径
"
```

**分析要点**:
- 定位方式: 系统搜索 / 偶然发现 / 提示引导
- 关键转折: 在哪一步发现了 bug
- 失败原因: 卡在哪里, 为什么
- hack 尝试: 有没有尝试绕过约束

### Skills 系统

```
.claude/skills/
├── README.md           ← Skills 索引
├── generate-task.md    ← 任务生成 (9 个 Phase)
├── calibrate-task.md   ← 难度校准
├── select-tasks.md     ← 选题
└── anti-hack.md        ← 反 hack 措施 (7 种)
```

| Skill | 用途 | 关键功能 |
|-------|------|---------|
| `/generate-task` | 生成评测题 | Bug 构造策略、诱饵设计、anti-hack |
| `/calibrate-task` | 校准难度 | 多次运行、轨迹分析、调整策略 |
| `/select-tasks` | 选题 | 按成功率排序、淘汰太难/太简单 |
| `/anti-hack` | 反 hack | 禁止上网、删除 .git、陷阱诱饵 |

### Bug 构造策略

| 策略 | 难度 | 说明 |
|------|------|------|
| 删除关键代码 | ⭐⭐⭐⭐⭐ | 删 `__syncthreads`、删 early return |
| 条件触发 | ⭐⭐⭐⭐ | 只在特定输入下触发 |
| 跨函数依赖 | ⭐⭐⭐⭐ | 需要理解调用链 |
| 陷阱诱饵 | ⭐⭐⭐ | 修了反而有害 (`&& True`、`assert`) |
| 数值精度 | ⭐⭐⭐ | 不崩溃, 需数值分析 |
| 符号翻转 | ⭐⭐ | 有迹可循 |
| 普通诱饵 | ⭐ | 最容易排除 |

详见 `skills/generate-task.md` Phase 4.5。

## 项目结构

```
agentic-xlang-bugfix/
├── README.md                          ← 本文件
├── .claude/
│   ├── skills/                        ← Claude Code Skills
│   │   ├── generate-task.md
│   │   ├── calibrate-task.md
│   │   ├── select-tasks.md
│   │   └── anti-hack.md
│   └── settings.local.json
├── .secrets/                          ← API keys (不提交)
│   ├── claude_api_key
│   ├── kimi_api_key
│   └── id_rsa
└── tasks/                             ← 15 道任务
    ├── task1-pytorch-cuda-index/
    │   ├── README.md                  ← 设计思路 + 踩坑记录
    │   ├── run.sh                     ← 单次运行 (含 deny WebSearch)
    │   ├── calibrate.sh               ← 多模型校准
    │   ├── trajectories/              ← 轨迹存储
    │   └── task/
    │       ├── task.toml              ← 任务元数据
    │       ├── instruction.md         ← 发给 agent 的 prompt
    │       ├── environment/
    │       │   ├── Dockerfile         ← 注入 bug + touch + 编译
    │       │   └── Dockerfile.base    ← fat base 镜像
    │       ├── workspace/             ← 用户代码 (train.py, model.py)
    │       ├── solution/
    │       │   ├── inject_bug.py      ← 注入 bug (支持 --reverse)
    │       │   ├── solve.sh           ← 调用 inject_bug.py --reverse
    │       │   └── oracle.sh          ← 验证 buggy 失败 + fixed 通过
    │       └── tests/
    │           └── test.sh            ← 判题脚本 (多场景 + anti-hack)
    └── task4-jax-vmap-batch/
        └── ... (同上结构)
```

## 评测流程

```
┌──────────────────────────────────────────────────────────────┐
│                    完整评测流程                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 构建镜像                                                  │
│     docker build --no-cache -t task1 -f Dockerfile .         │
│         ↓                                                    │
│  2. Oracle 验证                                               │
│     ./test_oracle.sh task1                                   │
│     ├── buggy 版本: 分数 < 1.0 ✓                             │
│     └── fixed 版本: 分数 = 1.0 ✓                             │
│         ↓                                                    │
│  3. Kimi 测试                                                 │
│     cd tasks/task1-pytorch-cuda-index && ./run.sh            │
│     ├── kimi 执行任务                                         │
│     ├── 保存 trajectory.jsonl                                │
│     └── 输出 result.jsonl (reward/turns)                     │
│         ↓                                                    │
│  4. 轨迹分析                                                  │
│     分析 agent 的调试路径、关键转折、失败原因                    │
│         ↓                                                    │
│  5. 调整难度                                                  │
│     太简单 → 增加 bug / 删除型 bug / 陷阱诱饵                  │
│     太难 → 减少 bug / 加提示                                   │
│         ↓                                                    │
│  6. 重复 2-5 直到成功率 ~50%                                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 反 Reward Hack

每道题的 `tests/test.sh` 包含多层反 hack 检查:

| 检查 | 防什么 |
|---|---|
| 多 seed / 多参数测试 | 防"只过一个 case" |
| 性能回归测试 | 防"CPU 回退绕过 CUDA bug" |
| diff 只允许底层文件 | 防"改 Python 不改 C++" |
| 静态分析(isnan / try/catch) | 防"Python 层打补丁掩盖" |
| Mutation testing | 防"测试套件本身弱" |

详见 `skills/anti-hack.md`。

## 反 Reward Hack

每道题的 `tests/test.sh` 包含多层反 hack 检查:

| 检查 | 防什么 |
|---|---|
| 多 seed / 多参数测试 | 防"只过一个 case" |
| 性能回归测试 | 防"CPU 回退绕过 CUDA bug" |
| diff 只允许底层文件 | 防"改 Python 不改 C++" |
| 静态分析(isnan / try/catch) | 防"Python 层打补丁掩盖" |
| Mutation testing | 防"测试套件本身弱" |

详见 `OVERVIEW.md` §6。

## 选题策略

设计了 15 道题,校准后**选成功率最接近 50% 的 10 道**:

- 成功率 40-60% → 选入(难度合适)
- 成功率 >60% → 淘汰或加难度
- 成功率 <30% → 淘汰或加提示

## Claude Code Skills

本项目包含 Claude Code skills,用于自动化任务生成和校准:

| Skill | 用途 | 用法 |
|---|---|---|
| `/generate-task` | 生成一道跨语言 bug-fix 评测题 | `/generate-task --cve CVE-2024-xxxx --framework pytorch` |
| `/calibrate-task` | 校准一道题的难度 | `/calibrate-task --task-dir tasks/taskN-xxx --runs 3` |
| `/select-tasks` | 从 N 道题里选最优 M 道 | `/select-tasks --n 15 --m 10` |

详见 `.claude/skills/` 目录。

## 相关项目

- [ExploitBench](https://github.com/exploitbench/exploitbench) — V8 JavaScript 引擎漏洞利用评测基准(本项目的灵感来源)
- [harbor](https://github.com/harbor-framework/harbor) — 任务格式框架

## 开发工具

本项目使用 **Claude Code** 作为 AI 编程助手，接入以下模型：

- **MiMo-V2.5-Pro**

## 许可证

[MIT License](LICENSE)

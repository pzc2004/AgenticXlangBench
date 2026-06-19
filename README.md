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

## 快速开始

```bash
# 克隆仓库
git clone git@github.com:pzc2004/AgenticXlangBench.git
cd AgenticXlangBench

# 跑单个任务(用 Claude Code)
cd tasks/task1-pytorch-cuda-index
./run.sh claude-sonnet-4-6 10

# 校准难度(3 模型 × 3 seed)
./calibrate.sh 10
```

## 项目结构

```
AgenticXlangBench/
├── README.md                          ← 本文件
├── OVERVIEW.md                        ← 详细设计文档
├── LICENSE                            ← MIT 许可证
└── tasks/                             ← 15 道任务
    └── taskN-<name>/
        ├── README.md                  ← 该任务的设计思路
        ├── run.sh                     ← 单次运行
        ├── calibrate.sh               ← 多模型校准
        ├── trajectories/              ← 轨迹存储(运行时生成)
        └── task/                      ← 任务本身(harbor 格式)
            ├── task.toml              ← 任务元数据
            ├── instruction.md         ← 发给 agent 的 prompt
            ├── environment/Dockerfile ← 运行环境
            ├── tests/test.sh          ← 判题脚本
            └── solution/solve.sh      ← Oracle 参考解
```

`task/` 目录是标准 harbor 格式,可独立使用。`run.sh` / `calibrate.sh` / `trajectories/` 是评测基础设施。

## 评测流程

```
run.sh / calibrate.sh
    ↓
Claude Code 执行任务(工具调用:exec / write_file / read_file)
    ↓
trajectory.jsonl 保存完整轨迹
    ↓
tests/test.sh 给分(写入 /logs/verifier/reward.txt)
    ↓
calibrate.sh 汇总:平均分 / 成功率 / 判定(太简单 / 太难 / 合适)
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

详见 `OVERVIEW.md` §6。

## 选题策略

设计了 15 道题,校准后**选成功率最接近 50% 的 10 道**:

- 成功率 40-60% → 选入(难度合适)
- 成功率 >60% → 淘汰或加难度
- 成功率 <30% → 淘汰或加提示

## 相关项目

- [ExploitBench](https://github.com/exploitbench/exploitbench) — V8 JavaScript 引擎漏洞利用评测基准(本项目的灵感来源)
- [harbor](https://github.com/harbor-framework/harbor) — 任务格式框架

## 开发工具

本项目使用 **Claude Code** 作为 AI 编程助手，接入以下模型：

- **MiMo-V2.5-Pro**

## 许可证

[MIT License](LICENSE)

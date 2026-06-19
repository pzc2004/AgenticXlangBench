# 跨语言跨仓库 Bug-Fix 出题项目

> 目标:出 ~15 道题(选最优 10 道),前沿模型成功率 ~50%,轨迹步数 >200
> 核心设计:bug 在底层(C/CUDA/Rust/ASM),症状在上层(Python/SQL/CLI),且**延迟显现**

## 1. 核心设计洞察:延迟显现(Delayed Manifestation)

普通 bug-fix:bug 执行 → 立刻报错 → agent 一眼看到 → 10 步修完

**本项目的题**:
```
bug 代码执行(CUDA kernel / C++ / Rust)
    ↓ 正常返回,没有报错
大量下游操作(PyTorch autograd / 图像管线 / 查询执行...)
    ↓ 每一步都"看起来正常"
症状出现(loss NaN / accuracy 下降 / 结果偏差 / segfault)
    ↓ 症状距离 bug 几百个操作,stack trace 指向错误位置
agent 需要:从症状反推 → 理解框架内部 → 穿越语言边界 → 找到真 bug
```

**延迟显现的 4 种手法**:

| 手法 | 题目 | 机制 |
|---|---|---|
| 累积误差 | 1/2/7/13 | 每次误差小,几百次后才明显 |
| 触发条件 | 3/7/9/14/15 | 特定 shape/size/value/dtype 才触发 |
| 级联传播 | 4/6/10/11/13 | 错误经过多个抽象层才到症状 |
| 概率性 | 5/8/12 | 内存/线程时序不确定,不是每次触发 |

## 2. 十五道题总览

| # | 任务 | Bug 层 | 症状 | 延迟 | 语言 | 可运行 |
|---|---|---|---|---|---|---|
| 1 | PyTorch CUDA index | CUDA | 训练 NaN | 累积 | Python+CUDA | ✅ |
| 2 | PyTorch autograd sign | C++ | accuracy↓ | 累积 | Python+C++ | ✅ |
| 3 | NumPy BLAS register | x86 ASM | 结果偏差 | 触发 | Python+ASM | ✅ |
| 4 | JAX vmap batching | Python | 梯度错 | 级联 | Python+内部 | ❌需装JAX |
| 5 | CPython refcount | C | segfault | 概率 | Python+C | ✅ |
| 6 | TF shape inference | C++ | shape err | 级联 | Python+C++ | ❌需装TF |
| 7 | cuDNN conv backward | CUDA | 效果差 | 累积+触发 | Python+CUDA | ✅ |
| 8 | Rust FFI lifetime | Rust | segfault | 概率 | Python+Rust | ✅ |
| 9 | PG executor NULL | C | 行数不对 | 触发 | SQL+C | ❌需装PG |
| 10 | Redis module write | C | 数据乱码 | 级联 | Redis+C | ❌需装Redis |
| 11 | LLVM codegen | C++ | -O2 输出错 | 级联 | C+ASM | ❌需编译LLVM |
| 12 | Python GIL race | C | 偶发错 | 概率 | Python+C | ✅ |
| 13 | OpenCV CUDA resize | CUDA | 像素偏差 | 级联 | Python+CUDA | ❌需编译OpenCV |
| 14 | NumPy dtype overflow | C | 溢出 | 触发 | Python+C | ✅ |
| 15 | SQLite optimizer | C | 少行 | 触发 | SQL+C | ❌需编译SQLite |

**本机可直接运行**:1/2/3/5/7/8/12/14(8 道)
**需要额外安装**:4/6/9/10/11/13/15(7 道)

## 3. 难度梯度

| 难度 | 题目 | 理由 |
|---|---|---|
| ⭐⭐⭐⭐⭐ | 1(CUDA) 7(cuDNN) 11(LLVM) 13(OpenCV) | 跨 3 层 + GPU/编译器 + 症状模糊 |
| ⭐⭐⭐⭐ | 2(autograd) 4(JAX) 5(refcount) 12(GIL) | 跨 2 层 + 需理解框架内部/并发 |
| ⭐⭐⭐ | 3(BLAS) 6(TF) 8(Rust) 14(dtype) | 跨 2 层 + 触发条件隐蔽 |
| ⭐⭐ | 9(PG) 10(Redis) 15(SQLite) | 跨 2 层但症状相对明确 |

## 4. 每个任务的结构

```
taskN-<name>/
├── README.md              ← 设计思路(bug / 延迟 / 为什么难 / anti-hack / oracle)
├── run.sh                 ← 单次运行:./run.sh [model] [budget]
├── calibrate.sh           ← 多模型校准:./calibrate.sh [budget]
├── trajectories/          ← 轨迹存储(自动创建)
│   ├── <timestamp>_<model>/
│   │   ├── trajectory.jsonl   ← 完整轨迹(stream-json)
│   │   ├── result.jsonl       ← 单次结果
│   │   └── stderr.log
│   └── calibration_results.jsonl  ← 校准汇总
└── task/                  ← 任务本身(可发布到 harbor)
    ├── task.toml          ← 任务元数据
    ├── instruction.md     ← 发给 agent 的 prompt
    ├── environment/
    │   └── Dockerfile     ← 运行环境
    ├── tests/
    │   └── test.sh        ← 判题脚本(分层评分)
    └── solution/
        └── solve.sh       ← Oracle(撤销注入)
```

`task/` 文件夹是干净的 harbor 格式,可直接发布。外围是开发/评测基础设施。

## 5. 使用方式

```bash
cd tasks/task1-pytorch-cuda-index

# 单次跑(用 Claude Code)
./run.sh claude-sonnet-4-6 10

# 校准(3 模型 × 3 seed = 9 次)
./calibrate.sh 10
```

`calibrate.sh` 跑完自动判定:
- 平均分 > 0.7 → "太简单"
- 平均分 < 0.3 → "太难"
- 0.3-0.7 → "合适 ✓"

## 6. Anti-hack 措施(通用)

跨语言 bug-fix 特有的 hack 路径:

| Hack | 检测 |
|---|---|
| Python 层加 NaN 检查 / 梯度裁剪 | grep `isnan` / `clip_grad` / `nan_to_num` |
| CPU 回退绕过 CUDA bug | 性能测试(GPU 应快 5x+) |
| 改训练参数避开触发 | 多参数组合测试 |
| try/catch 吞错误 | grep `try:` |
| 注释掉 / 禁用代码 | diff 行数检查 + 功能测试 |
| 只改上层不改底层 | diff 只允许 .cu/.cpp/.c/.rs/.h |
| monkey-patch 替换实现 | grep `setattr` |
| 改测试断言 | diff 检查 tests/ 改动 |

每项检查都写进 `tests/test.sh`,是 reward 的一部分。

## 7. 选题策略

15 道出完后,校准时**选成功率最接近 50% 的 10 道**:

```
校准后:
  成功率 40-60% → 选入(难度合适)
  成功率 >60%   → 淘汰或加难度(太简单)
  成功率 <30%   → 淘汰或加提示(太难)
```

## 8. Oracle 设计

所有题的 oracle 都是**撤销注入**(恢复原始代码):

```bash
#!/bin/bash
cd /path/to/project
git checkout HEAD -- path/to/buggy/file.c
# 增量重编(如需要)
```

Oracle 成本极低(几行脚本),因为 bug 是注入的不是真实存在的。

## 9. Bug 注入原则

1. **改动最小**(1-3 行)—— 避免引入多个 bug
2. **真实可信**(改动类型跟真实 CVE 一致)
3. **不在 git history 里**(避免 agent 直接搜 diff)
4. **注释/commit message 不留痕迹**

## 10. 落地优先级

1. **先做 Task 1 prototype**(PyTorch CUDA) — 最具代表性,环境就绪
2. **跑 baseline 校准** — 用 `calibrate.sh` 跑 3 Claude 模型 × 3 seed
3. **复制模式做其余 14 道** — 同一套 inject → test → oracle 流程
4. **选最优 10 道** — 淘汰偏离 50% 太远的 5 道
5. **完善 instruction.md 和 test.sh** — 加欺骗性约束、多场景测试

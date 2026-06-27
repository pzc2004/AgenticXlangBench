# /generate-task — 生成一道跨语言 bug-fix 评测题

## 用途

生成一道面向 agentic RL 后训练的跨语言 bug-fix 评测题。
目标：前沿模型成功率 ~50%，轨迹步数 >200。

## 适用框架

本 skill 适用于任何"上层语言调用底层实现"的场景：

| 上层 | 底层 | 典型 bug 来源 |
|------|------|-------------|
| Python (PyTorch) | CUDA/C++ | kernel 逻辑错误、数值精度 |
| Python (TensorFlow) | CUDA/C++ | shape inference、op 实现 |
| Python (NumPy) | C/BLAS | 数组操作、线性代数 |
| Python (JAX) | XLA/C++ | jit 编译、vmap 批处理 |
| Python (CPython 扩展) | C | refcount、GIL、内存管理 |
| Python (Rust FFI) | Rust | 生命周期、unsafe 代码 |
| SQL (PostgreSQL) | C | 查询执行、优化器 |
| SQL (SQLite) | C | 查询计划、索引 |
| Redis 命令 | C | 数据结构、序列化 |
| C/C++ (LLVM) | ASM | 代码生成、优化 pass |

## 执行流程

### Phase 1： 分析 Bug

1. **定位 bug 文件和行号**
   - 如果提供 CVE：读 CVE 描述 → 找到对应源码文件 → 确定 bug 位置
   - 如果提供 bug-file：直接使用

2. **分析 bug 类型**
   - 数值稳定性(NaN/Inf/精度损失)
   - 越界访问(off-by-one/缓冲区溢出)
   - 竞态条件(数据竞争/死锁)
   - 类型错误(溢出/截断/隐式转换)
   - 逻辑错误(错误公式/错误比较/错误分支)
   - 内存问题(泄漏/use-after-free/double-free)

3. **确定触发条件**
   - 随机数据就能触发(最理想)
   - 需要特定输入才能触发(需要设计数据注入)
   - 需要特定环境才能触发(需要设计环境)

### Phase 2： 设计用户可见的接口

**核心原则**：bug 所在的底层代码不能直接暴露给用户。用户通过上层接口交互。

**通用模式**：
```
用户脚本(train.py / query.sql / redis-cli)
    ↓ 调用
上层封装(model.py / ORM / client library)
    ↓ 调用
底层实现(CUDA kernel / C extension / SQL executor)
    ↑ bug 在这里
```

**关键技巧**：用自定义封装类隐藏底层实现。例如：
- PyTorch: `class FeatureNorm` 封装 `F.layer_norm`
- NumPy： 自定义 `my_svd` 封装 `np.linalg.svd`
- SQL： 视图或存储过程封装底层查询
- Redis： 自定义命令封装底层操作

**文件组织**：
- 用户脚本： `train.py` / `query.sql` / `benchmark.sh`
- 封装层： `model.py` / `schema.sql` / `client.py`
- 底层源码： 在 Docker 镜像中(不直接可见)

### Phase 3： 设计用户脚本

**核心原则**：不暴露任何 bug 细节。

```python
# 通用结构(以 ML 训练为例)
from model import MyModel  # 不直接 import 底层库

def make_data(batch_size, device):
    # 随机数据,不需要特殊输入
    return random_data, random_labels

# 没有 hook、没有特殊注入、纯随机数据
# bug 由底层代码自身触发
```

**禁止**：
- ❌ 不要在用户脚本里提到 bug 所在的底层函数名
- ❌ 不要加 hook / 注入 / 特殊处理
- ❌ 不要加错误检查 / 异常捕获 / 降级逻辑

### Phase 4： 设计诱饵

**目标**：多个底层文件被修改，只有 1-3 个是真 bug。

**诱饵类型**(优先用真实代码改动)：
1. 改常量(精度、阈值、缓冲区大小)
2. 改比较符(< → <=， >= → >， == → ！=)
3. 改循环边界(off-by-one 但不触发)
4. 改数值精度(cast 类型、舍入方式)
5. 改错误处理(添加死代码分支)
6. 改内存对齐(看起来像性能优化)

**诱饵要求**：
- ✅ 编译通过
- ✅ 运行不崩溃
- ✅ 看起来像真 bug
- ❌ 不能是纯注释(太容易排除)

**注入方式：统一用 unified diff patch(task1/task4 现行架构)**

不再用 inline 字符串替换(`str.replace`/精确匹配源码)注入，改为 patch 应用/回退。
`solution/` 目录结构：

```
solution/
├── clean_src/                  ← 干净源码(patch 生成的输入,取自 fat-base 镜像)
├── generate_decoys.py          ← 诱饵定义        → decoys.patch
├── generate_per_bug_patches.py ← BUGS 单一事实来源 → per_bug_patches/Bug_N.patch
├── generate_bugs_patch.py      ← (clean+decoys)→+bugs 合成 → bugs.patch
├── decoys.patch                ← 诱饵层:build 永久应用、solve 不回退
├── bugs.patch                  ← bug 层:build 应用、solve 回退
├── per_bug_patches/            ← 每个 bug 一个 patch(供审查 / per-bug oracle)
├── inject_bug.py               ← 应用/回退 bugs.patch(`--reverse`)
└── solve.sh                    ← inject_bug.py --reverse + 增量编译
```

**双层 patch 的关键**：
- 注入链路：`clean → patch decoys.patch → inject_bug.py(应用 bugs.patch) → 编译`
- 修复链路：`solve.sh → inject_bug.py --reverse → 增量编译`(只回退 bug，诱饵保留 → 精确等于纯诱饵态)
- bug 改动全部由 `BUGS` 列表(`generate_per_bug_patches.py`)单一事实来源生成，改 bug 只动这一处
- **patch 文件 build 时用完即删**(`rm bugs.patch decoys.patch`)，agent 读不到 → 替代旧的 `git commit` 隐藏法

> 为什么 patch 优于 inline：锚点由 `clean_src` 真实源码生成，天生对齐，不会出现 inline 时代"模式拼错导致 bug 没注入"的哑改(见教训 15)。

**patch 生成步骤(一次性)**：
1. **取干净源码**：从 fat-base 镜像把目标文件拷进 `clean_src/`(patch 生成的基线，之后不手动改)。
2. **写定义**：诱饵写进 `generate_decoys.py`；真 bug 写进 `generate_per_bug_patches.py` 的 `BUGS` 列表(单一事实来源，每条 = 文件 + old 片段 + new 片段)。
3. **生成三份 patch**：
   - `generate_decoys.py` → `decoys.patch`
   - `generate_per_bug_patches.py` → `per_bug_patches/Bug_N.patch`(逐 bug，供审查 / per-bug oracle)
   - `generate_bugs_patch.py` → 在 **(clean+decoys)** 之上合成 `bugs.patch`(保证 bug 层叠在诱饵层上仍精确可逆)
4. 之后改 bug/诱饵**只动定义脚本再重新生成**，绝不手编 `.patch`。

**注入校验：三关，缺一不可**(实现于 `inject_bug.py`):
1. **fuzz=0 精确应用(主防线)**：`patch -p0 -F0 --no-backup-if-mismatch`。`-F0` 禁用模糊匹配，任一 hunk 上下文不精确匹配即 reject + 非 0 退出 → 等于对**全部 hunk** 做精确落地校验。注：`-F0` 只禁 fuzz，仍容忍纯行号偏移，源码小幅变动不会误杀。
2. **无 `.rej` 残留**：解析 patch 的 `+++` 目标文件，逐个确认没有 `<file>.rej`(双保险)。
3. **marker 冒烟**：注入后断言几个 buggy 态独有串在、回退后消失。**只作抽样冒烟**——前两关已全检测，且删除型 bug 删完无独有新文本、天生写不出可靠 marker，绝不能把 marker 当主防线。

**往返一致**：`apply → reverse` 后源码必须精确等于纯诱饵态(用 oracle 验证：solve 回退 + 重编译后满分，reverse 后逐字节等于 decoys 态)。

### Phase 4.5： Bug 构造策略

**核心发现**：删除代码比添加代码更难被 agent 修复。

#### 策略 1： 删除关键代码(最难)

删掉一行必要的代码，agent 需要理解缺失了什么并补回来。

```c
// 原始代码
__syncthreads();
return WelfordDataLN(sigma2, mean);

// Bug: 删除 __syncthreads
return WelfordDataLN(sigma2, mean);
```

**为什么难**：
- agent 看到的是"正常代码"，没有可疑的修改
- 需要理解并发语义才知道少了 `__syncthreads`
- `git diff` 无法使用(已删除 .git)
- 需要对比参考实现才能发现缺失

**Task 1 实测**：删除 `__syncthreads` 的 race condition bug，Kimi 246 步都没修完。

#### 策略 2： 条件触发(次难)

bug 只在特定条件下触发，agent 需要找到触发条件。

```c
// Bug: 只在方差很小时翻转符号
T_ACC rstd_val = (sigma2 > 0.5 && sigma2 < 2.0)
    ? -rsqrt(sigma2 + eps)
    : rsqrt(sigma2 + eps);
```

**为什么难**：
- 小 batch 测试可能通过(方差不在范围内)
- 大 batch 才触发，agent 容易以为修完了
- 需要理解数值范围才能定位

#### 策略 3： 跨 kernel 依赖(较难)

bug 跨越多个函数/文件，需要理解调用链。

```c
// kernel A: 设置标志
if (threadIdx.x == 0 && blockIdx.x == 0) {
    _ln_flag = true;
}

// kernel B: 使用标志(但标志可能还没设置)
b[index] = -scale * (mean[ng] + 0.1 * _ln_flag);
```

**为什么难**：
- 单独看每个 kernel 都"正确"
- 需要理解 CUDA 执行模型(不同 block 并行)
- 需要追踪跨函数的数据流

#### 策略 4： 诱饵比真 bug 更多(干扰)

注入 30+ 个诱饵，只有 3-5 个是真 bug。

```python
# 诱饵(看起来像 bug 但不影响功能)
float _sigma2_floor = 1e-6f;  # 声明但未使用
float _eps_override = 1e-5f;  # 声明但未使用

# 真 bug(隐藏在诱饵中)
T_ACC rstd_val = (sigma2 > 0.5 && sigma2 < 2.0)
    ? -rsqrt(sigma2 + eps)
    : rsqrt(sigma2 + eps);
```

**为什么难**：
- agent 需要逐个检查每个修改
- 诱饵消耗 agent 的注意力和预算
- 真 bug 藏在大量诱饵中

#### 策略 5： 数值精度陷阱(隐蔽)

微小的数值变化，不会崩溃但影响训练质量。

```c
// Bug: eps 放大 100 倍
rsqrt(var + eps * 100.0f)  // 应该是 eps

// Bug: 梯度累加方向错误
stats_x2 -= c_loss * gamma * (c_h - mean) * rstd;  // 应该是 +=
```

**为什么难**：
- 不会崩溃，不会 NaN
- 只是训练精度下降
- 需要数值分析才能发现

#### Bug 类型难度排名

| 难度 | 类型 | 原因 |
|------|------|------|
| ⭐⭐⭐⭐⭐ | 删除关键代码 | 看不到异常,需要理解缺失 |
| ⭐⭐⭐⭐ | 条件触发 | 可能测试通过,大场景才失败 |
| ⭐⭐⭐⭐ | 跨函数/文件依赖 | 需要理解调用链 |
| ⭐⭐⭐ | 诱饵淹没 | 消耗注意力,但能逐个排除 |
| ⭐⭐⭐ | 数值精度 | 不崩溃,需要数值分析 |
| ⭐⭐ | 符号翻转 | 有迹可循,对比即可发现 |
| ⭐ | 添加多余代码 | 最容易发现和删除 |

**实测修复率(task1， kimi seed42， reward 0.98， 22/35 修对)** —— 给上面星级一个经验锚点：

| bug 类型 | 修复率 | 解读 |
|---|---|---|
| 激活/Dropout 数值(`*0.9`/`+0.01`/`x_cube`) | ~100% | **免费分**:有可见可疑常量,一波 grep 全清,几乎不贡献难度 |
| 数值缩放(`*0.95`/`*0.8`) | 100% | 同上 |
| 条件触发(符号翻转 / eps 放大) | ~80% | 能 grep 到的会修,藏在冷门算子里的会漏 |
| 跨 kernel(`_ln_flag`) | ~100% | kimi 能跟标志位,没想象中难 |
| **删除型 `__syncthreads`** | **~19%(3/16)** | **难度主引擎**:只修对"主战场算子(LN backward)"里位置明显的;**SoftMax / GroupNorm 的删除型全漏** |

**关键教训**(调控难度时直接照用)：
- **删除型是唯一真正拉开难度的类型**，但仅当它落在 agent **必看的主算子之外**(如 SoftMax/GN)才漏修；放在 agent 反复试错的算子(LN)里会被顺手修掉。
- 数值/常量型修复率近 100% → **占步数但不构成难度**，适合做"基底分 + 消耗注意力"，别指望它拉难度。
- **隐形难度陷阱**：删除型大量漏修、reward 却仍 0.98 → test 对这些 race 命中不足，等于"难且测不出"。加难度时**必须同步加强 test 对删除型的带电命中**，否则白注入(见 calibrate-task.md Step 4 隐形难度警告 + `analyze_trajectory.py`)。

#### 实战组合建议

```
最佳组合(3 个真 bug + 30 个诱饵):
1. 删除 __syncthreads (race condition)
2. 条件触发的符号翻转 (sigma2 范围)
3. 跨 kernel 的标志位依赖

诱饵分布:
- 10 个声明但未使用的变量
- 10 个不影响功能的常量修改
- 10 个注释/死代码
```

### Phase 5： 设计 instruction.md

**核心原则**：不暴露任何 bug 细节。

```markdown
# 任务:修复 [框架名] [症状描述]

## 背景
我们使用了一个从源码编译的 [框架]。运行时出现 [症状]。

## 你的任务
1. 理解 bug
2. 定位 bug(在 [源码目录] 中)
3. 修复 bug(只允许改 [允许的文件类型])
4. 重新编译(如需要)
5. 验证(bash /task/tests/test.sh)
6. 检查分数(≥ 0.6 才通过,< 0.6 继续排查)

## 约束条件
- 只允许改 [底层语言] 文件
- 不允许改 [上层语言] 文件
- 不允许加 [hack 类型]
```

**禁止在 instruction.md 里提到**：
- ❌ bug 所在的具体文件名
- ❌ bug 的具体类型
- ❌ bug 的修复方法
- ❌ 任何 hook / 注入机制

### Phase 6： 设计 test.sh

**核心原则**：只检查结果，不暴露 bug 细节。

**通用评分标准**(满分 1.0):
```
0.10  基础:框架/库可导入
0.05  基础:运行环境可用(GPU/数据库/等)
0.40  核心:功能测试(多组输入/多场景)
0.15  边界测试(不同参数/不同数据量)
0.10  性能测试(与基线对比)
0.20  Anti-hack:无掩盖 + 关键文件未被修改
```

**测试策略选择**：

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 崩溃检查 | bug 导致 crash/NaN | 简单直接 | 太容易,agent 秒杀 |
| 结果对比 | bug 导致错误结果 | 直观 | 需要设计合适的阈值 |
| 基线对比 | bug 影响性能/质量 | 精确检测 | 需要固定数据模式 |
| **多场景复合** | **多个 bug** | **最难** | **设计复杂** |

**推荐：多场景复合测试**：
```bash
# 场景 1: 基本功能测试(覆盖 Bug A+B)
for seed in 1 2 3 4 5; do
    run_with_params --seed $seed --small_batch
    # 检查结果质量 >= 阈值
done

# 场景 2: 边界条件测试(覆盖 Bug C)
for param in large_value special_value edge_case; do
    run_with_params --param $param
    # 检查不崩溃/不 NaN
done
```

**禁止在 test.sh 里暴露**：
- ❌ bug 所在的文件名
- ❌ bug 的具体模式
- ❌ 修复方法

**Anti-hack 检查**(通用)：
- 结果掩盖(替换 NaN/Inf/错误值)
- 降级回退(绕过底层直接用上层实现)
- 修改用户脚本(改变调用方式)
- 修改封装层(改变底层调用)

### Phase 7： 设计 Dockerfile

**两阶段构建**：

```dockerfile
# Dockerfile.base (一次性,耗时较长):
# - 从源码编译框架
# - 保留 .git / third_party / build 目录
# - 支持增量编译

# Dockerfile (增量,较快):
FROM <base-image>
# 注入用 patch(见 Phase 4 patch 架构):先永久应用诱饵,再应用 bug
COPY solution/decoys.patch solution/bugs.patch solution/inject_bug.py /tmp/
RUN patch -d <源码目录> -p0 < /tmp/decoys.patch && \
    PYTORCH_DIR=<源码目录> BUGS_PATCH=/tmp/bugs.patch python /tmp/inject_bug.py && \
    rm /tmp/decoys.patch /tmp/bugs.patch /tmp/inject_bug.py   # 用完即删,防 agent 读到
RUN <编译命令> && <安装命令>
RUN rm -rf <源码目录>/.git  # 删除 git 历史
RUN find <源码目录> -name "*.<ext>" -exec touch {} +  # 统一时间戳(防 stat 定位)
CMD ["/bin/bash"]
```

**关键**：
- 用 `--mount=type=secret` 挂载 SSH key(不烤进镜像)
- 用 `--build-arg CACHE_BUST=$(date +%s)` 强制重编
- API key 运行时注入(`docker run -e` 或 `-v`)
- **必须删除 .git** 防止 agent 用 git diff/show
- **patch 文件用完即删** > 旧的 `git commit` 隐藏法；删 patch + 删 .git + touch 时间戳 三件套缺一不可
- 判分逻辑防读(setuid grade + 非 root agent)见 `anti-hack.md` 第 8 条

### Phase 8： 验证 + 校准

**⚠️ 每次运行前必须用 oracle 验证 test.sh！**

在跑 Kimi/Claude 测试之前，必须先用 oracle fix 验证 test.sh 能正确给分：

```bash
# 1. 启动镜像(不运行 agent)
docker run --rm --gpus all \
  -v $(pwd)/task/workspace:/workspace:ro \
  -v $(pwd)/task:/task:ro \
  task_image bash -c "
    # 2. 应用 oracle fix
    bash /task/solution/solve.sh
    
    # 3. 运行 test.sh
    bash /task/tests/test.sh
    
    # 4. 检查分数应为 1.0
    cat /logs/verifier/reward.txt
  "
```

**如果 oracle 分数不是 1.0，说明 test.sh 有 bug，必须先修复再跑 agent 测试！**

常见问题：
- 梯度检查阈值太严/太松 → 调整阈值
- accuracy 阈值不合理 → 用梯度检查替代
- anti-hack 误报 → 检查 grep 模式
- 超时 → 减少训练步数

验证通过后，再执行：
1. **验证 bug 生效**：运行用户脚本，确认症状出现
2. **验证基线正常**：在无 bug 环境运行，确认症状不出现
3. **验证 git 不可用**：在运行环境内运行 `git log`，应失败
4. **跑校准**：用目标模型跑 3 次，统计 reward
5. **检查 reward**：目标 40-60%(太简单 > 60%，太难 < 40%)

### Phase 9： 输出

最终输出：
```
tasks/taskN-<name>/
├── README.md
├── run.sh / calibrate.sh
└── task/
    ├── task.toml / instruction.md
    ├── environment/Dockerfile / Dockerfile.base
    ├── workspace/  (用户脚本 + 封装层)
    ├── solution/   (注入脚本 + Oracle)
    └── tests/      (判题脚本)
```

## 参考实现(canonical 样例，直接引用不另维护模板)

本 skill **不内置脚本模板**，而是把已被 oracle 实测跑通的真实任务当作单一事实来源。
出新题时直接 `cp` 对应文件再改少数几处，避免模板副本与真实代码漂移。

**两个 canonical 样例，按框架类型选**：
- **编译型 / 底层(CUDA、C/C++ 扩展、BLAS)** → `tasks/task1-pytorch-cuda-index/`
- **纯 Python / JIT(JAX、TF、NumPy)** → `tasks/task4-jax-vmap-batch/`

> ⚠️ 别拿 CUDA 样例去套 Python 题：编译型每改一处要重编译(影响 per-bug oracle 取舍、solve.sh 内容)，纯 Python 是秒级迭代。

**一套题的标准交付物**(以 task1 目录为准)：
```
task/
├── instruction.md                    ← 任务描述(不泄露 bug,见 Phase 5)
├── workspace/{train.py, model.py}    ← 用户脚本+封装层(只读挂载,见 Phase 2-3)
├── tests/test.sh                     ← 判分(分项+带电+HACK,见 Phase 6 / 教训 19)
├── environment/{Dockerfile, Dockerfile.base, grade.c}  ← 构建+判分防读(Phase 7 / anti-hack 第8条)
└── solution/
    ├── clean_src/                    ← 干净源码基线(patch 生成输入)
    ├── generate_decoys.py            ← 诱饵定义 → decoys.patch
    ├── generate_per_bug_patches.py   ← BUGS 定义 → per_bug_patches/
    ├── generate_bugs_patch.py        ← 合成 bugs.patch
    ├── inject_bug.py                 ← 应用/回退 + 校验三关
    ├── solve.sh / oracle.sh          ← 修复 / 双向验证
    └── oracle_per_bug.py             ← 逐 bug 验证(秒级迭代任务才用,见教训 20)
run.sh / calibrate.sh                 ← 运行 / 校准(模板见下文)
analyze_trajectory.py                 ← 轨迹分析(消费 trajectories/,出修对/漏修/难度报告)
```

**引用索引：每个环节抄哪个文件、改哪几处**

| 出题环节 | 参考文件(task1) | 移植说明 |
|---|---|---|
| patch 注入 + 校验三关 | `solution/inject_bug.py` | **骨架照抄**(fuzz=0/.rej/幂等),只改 `PYTORCH_DIR`、`BUG_MARKERS` 冒烟串 |
| bug 定义 | `solution/generate_per_bug_patches.py` 的 `BUGS` | **看格式**(文件+old+new 三元组),内容按本题全换 |
| 诱饵定义 | `solution/generate_decoys.py` | 看四类诱饵写法(ln_flag/陷阱/普通/迷惑),内容全换 |
| patch 合成 | `solution/generate_bugs_patch.py` | **照抄**(clean+decoys 之上合成,保证可逆) |
| 修复 + 回退 | `solution/solve.sh` | 改**编译命令**(ninja→对应框架的 build) |
| oracle 双向验证 | `solution/oracle.sh` | 照抄(buggy<1.0 且 fixed=1.0,见教训 16) |
| per-bug 验证 | `solution/oracle_per_bug.py` | 仅秒级迭代任务用;编译型跳过(教训 20) |
| 分项 + 带电 test | `tests/test.sh` | **看结构**(分项给分/带电检查/reward.txt/HACK 减半),检查项按本题重写(教训 19) |
| 判分防读 | `environment/grade.c` + Dockerfile 层 | 照抄,见 `anti-hack.md` 第 8 条 |
| 轨迹分析 | `solution/analyze_trajectory.py` | 照抄,`--bugs` 指向本题 BUGS;`classify_bug` 按本题模式微调 |
| 运行 / 校准 | `run.sh` / `calibrate.sh` | 见下文模板,改任务名/镜像名 |

> 维护约定：task1/task4 兼任"活样例"，其脚本须保持**通用骨架与任务特有内容(具体 bug 片段)分离清晰**，不要塞实验性改动。

---

## run.sh 模板

每个 task 需要一个 `run.sh` 用于单次运行。**必须使用 Docker 运行环境**，不能直接在本地运行。

关键设计：
1. **Docker 运行环境**：agent 在镜像内执行，与本地系统隔离
2. **docker commit**：保存 agent 的修复状态
3. **快照镜像测试**：在 commit 后的镜像里跑 test.sh
4. **API key 注入**：通过环境变量或挂载，不写入镜像

```bash
#!/bin/bash
# run.sh — 用 Kimi Code 跑单次任务 + 保存轨迹 + 拿 reward
# 用法: ./run.sh [model] [budget_usd] [seed] [timeout]

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
SEED="${3:-42}"
TIMEOUT="${4:-3600}"

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$SCRIPT_DIR/task"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_SAFE}_seed${SEED}"
TRAJ_DIR="$SCRIPT_DIR/trajectories/$RUN_ID"
mkdir -p "$TRAJ_DIR"

TASK_NAME="$(basename "$SCRIPT_DIR")"

# 读取 API keys
SECRETS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/.secrets"
ANTHROPIC_KEY=$(cat "$SECRETS_DIR/claude_api_key" 2>/dev/null | tr -d '[:space:]')
KIMI_KEY=$(cat "$SECRETS_DIR/kimi_api_key" 2>/dev/null | tr -d '[:space:]')

# 生成 Kimi config.toml(临时)
KIMI_CONFIG="$TRAJ_DIR/kimi_config.toml"
cat > "$KIMI_CONFIG" << EOF
default_model = "$MODEL"
default_permission_mode = "yolo"

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"
api_key = "$KIMI_KEY"

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 262144

[loop_control]
max_steps_per_turn = 500
max_retries_per_step = 3

[[permission.rules]]
decision = "allow"
pattern = "Read"

[[permission.rules]]
decision = "allow"
pattern = "Write"

[[permission.rules]]
decision = "allow"
pattern = "Bash"
EOF

# 构造任务 prompt
TASK_PROMPT="Seed: $SEED.

$(cat "$TASK_DIR/instruction.md")

## 完成后的验证步骤

修复 bug 后,运行以下命令验证:

\`\`\`bash
bash /task/tests/test.sh
\`\`\`

这会输出分数(0-1)。确保在修复后运行这一步。"

# Docker 运行参数
CONTAINER_NAME="${TASK_NAME}_$(date +%s)"
SNAPSHOT_IMAGE="${TASK_NAME}_snapshot_$(date +%s)"

# [1/3] 启动 Kimi Code
echo ">>> [1/3] 启动 Kimi Code (超时 ${TIMEOUT}s)..."
timeout "$TIMEOUT" \
docker run --name $CONTAINER_NAME --gpus all \
  -v "$TASK_DIR/workspace:/workspace:ro" \
  -v "$TASK_DIR:/task:ro" \
  -v "$TRAJ_DIR:/trajectories" \
  -v "$KIMI_CONFIG:/root/.kimi-code/config.toml:ro" \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_KEY" \
  TASK_IMAGE_NAME \
  kimi -p "$TASK_PROMPT" \
    --model "$MODEL" \
    --output-format stream-json \
    2>"$TRAJ_DIR/stderr.log" \
    | grep '^{.*}$' > "$TRAJ_DIR/trajectory.jsonl" || true

# [2/3] 保存修复状态,跑测试
echo ">>> [2/3] 保存修复状态..."
docker commit "$CONTAINER_NAME" "$SNAPSHOT_IMAGE" > /dev/null 2>&1
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1

echo ">>> 运行测试..."
docker run --rm --gpus all \
  -v "$TASK_DIR/workspace:/workspace:ro" \
  -v "$TASK_DIR:/task:ro" \
  "$SNAPSHOT_IMAGE" \
  bash -c "
    bash /task/tests/test.sh 2>/dev/null || true
    cat /logs/verifier/reward.txt 2>/dev/null || echo '0.0'
  " > "$TRAJ_DIR/reward.txt" 2>/dev/null || true

docker rmi "$SNAPSHOT_IMAGE" > /dev/null 2>&1 || true

# [3/3] 汇总结果
REWARD=$(tail -1 "$TRAJ_DIR/reward.txt" 2>/dev/null | tr -d '[:space:]')
REWARD="${REWARD:-0.0}"
TURNS=$(grep -c '"role":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
TOOL_CALLS=$(grep -c '"tool_calls"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
rm -f "$KIMI_CONFIG"

echo "{\"task\":\"$TASK_NAME\",\"model\":\"$MODEL\",\"seed\":$SEED,\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS}" | tee "$TRAJ_DIR/result.jsonl"
```

**注意**：
- `--gpus all` 只在需要 GPU 的 task 使用(CUDA/OpenCV 等)
- `TASK_IMAGE_NAME` 替换为实际镜像名(如 `task1-pytorch-cuda-index`)
- grep 模式是 `"role":"assistant"` 和 `"tool_calls"`，不是 `"type":"assistant"` 和 `"type":"tool_use"`

---

## calibrate.sh 模板

每个 task 需要一个 `calibrate.sh` 用于多次校准运行。

```bash
#!/bin/bash
# calibrate.sh — 多 seed 校准难度
# 用法: ./calibrate.sh [model] [budget] [num_runs] [timeout]

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
NUM_RUNS="${3:-3}"
TIMEOUT="${4:-3600}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/trajectories/calibration_results.jsonl"

mkdir -p "$SCRIPT_DIR/trajectories"
> "$RESULTS_FILE"

TASK_NAME="$(basename "$SCRIPT_DIR")"

echo "========================================="
echo " 校准任务: $TASK_NAME"
echo " 模型:     $MODEL"
echo " 运行次数: $NUM_RUNS"
echo "========================================="

for SEED in $(seq 1 $NUM_RUNS); do
    echo ""
    echo ">>> Run $SEED / $NUM_RUNS (seed=$SEED)"
    echo "-----------------------------------------"

    RESULT=$("$SCRIPT_DIR/run.sh" "$MODEL" "$BUDGET" "$SEED" "$TIMEOUT" 2>&1 | tee /dev/stderr | grep '^{' | tail -1 || true)

    if [ -n "$RESULT" ]; then
        echo "$RESULT" >> "$RESULTS_FILE"
    fi
done

echo ""
echo "========================================="
echo " 校准结果汇总"
echo "========================================="

python3 -c "
import json, sys
from collections import defaultdict

results = []
for line in open('$RESULTS_FILE'):
    line = line.strip()
    if line:
        try:
            results.append(json.loads(line))
        except:
            pass

if not results:
    print('无有效结果')
    sys.exit(0)

print(f'有效运行: {len(results)}')
print()

by_model = defaultdict(list)
for r in results:
    by_model[r['model']].append(r)

print(f'{\"模型\":<20} {\"平均分\":<10} {\"修复率\":<10} {\"平均轮数\":<10}')
print('-' * 50)
for model in sorted(by_model.keys()):
    rs = by_model[model]
    avg_reward = sum(r['reward'] for r in rs) / len(rs)
    fix_rate = sum(1 for r in rs if r.get('reward', 0) >= 0.8) / len(rs) * 100
    avg_turns = sum(r.get('turns', 0) for r in rs) / len(rs)
    print(f'{model:<20} {avg_reward:<10.2f} {fix_rate:<10.0f}% {avg_turns:<10.0f}')

print()
avg_all = sum(r['reward'] for r in results) / len(results)
if avg_all > 0.7:
    print(f'判定: 平均分 {avg_all:.2f} > 0.7 → 太简单,需要加难度')
elif avg_all < 0.3:
    print(f'判定: 平均分 {avg_all:.2f} < 0.3 → 太难,需要加提示或降低难度')
else:
    print(f'判定: 平均分 {avg_all:.2f} 在 0.3-0.7 → 难度合适 ✓')
"
```

---

## Docker 镜像构建经验

### 构建顺序

1. **先构建基础镜像**(如果需要)：`Dockerfile.base` → `docker build -t xxx-base`
2. **再构建 task 镜像**：`Dockerfile` → `docker build -t taskN`
3. **验证镜像**：运行简单命令确认 bug 注入正确

### 常见构建失败及解决

| 失败原因 | 解决方法 |
|---------|---------|
| git clone HTTPS 失败 | 改用 SSH: `git clone git@github.com:...` |
| 镜像拉取超时 | 用镜像源: `docker.1ms.run/xxx` |
| apt-get 包不存在 | 添加 APT 源或换基础镜像 |
| pip install 编译失败 | 用预编译包或简化依赖 |
| inject_bug.py 找不到文件 | 用 `glob.glob()` 动态搜索 |
| .so 导入失败 | 用 `python setup.py build_ext --inplace` |
| 自定义基础镜像不存在 | 先创建 `Dockerfile.base` |

### 并行构建

机器配置足够时，可以并行构建多个镜像：

```bash
DOCKER_BUILDKIT=1 docker build -t task1 ... > /tmp/build_task1.log 2>&1 &
DOCKER_BUILDKIT=1 docker build -t task2 ... > /tmp/build_task2.log 2>&1 &
wait
```

### 镜像大小优化

- 使用 `python:3.11-slim` 而非 `python:3.11`
- 删除 `.git` 目录(节省 50-80%)
- 使用 `--no-cache-dir` 安装 pip 包
- 合并 RUN 命令减少层数

### 验证清单

构建后必须验证：
1. ✅ Bug 注入正确(grep 检查)
2. ✅ .git 已删除
3. ✅ 基本功能可用(import / 运行测试)
4. ✅ train.py / model.py 可访问(通过挂载)

---

## 通用出题技巧：如何根据轨迹增加难度

### 原则：分析 agent 轨迹，找到"秒杀"路径，然后堵死它

当 agent 太快解决题目时，不要改 bug 本身，而是改 agent 的"捷径"。

### 技巧 1：删除 git 历史

**问题**：agent 用 `git show <commit>` 直接看到所有改动。

**解决**：Dockerfile 最后加 `rm -rf <源码目录>/.git`。

**效果**：agent 必须靠阅读源码和分析来定位 bug。

**适用**：所有框架(Python/C++/Rust/SQL/。。。)

### 技巧 2：用复合 bug 替代单一 bug

**问题**：单一 bug 太容易，agent 找到就修完了。

**解决**：注入 2-3 个 bug，每个在不同条件下触发：
- Bug A：所有输入都触发(如符号错误)
- Bug B：特定输入触发(如边界条件)
- Bug C：特定状态触发(如大数据量)

**效果**：agent 修了 A 还有 B，修了 B 还有 C。

**适用**：所有框架。示例：
- PyTorch：梯度符号 + eps 条件 + NaN 注入
- NumPy：数组索引 + 精度损失 + 内存泄漏
- SQL：查询计划 + 索引选择 + 溢出

### 技巧 3：用多层封装模糊定位

**问题**：模型只有单一底层调用，agent 直接定位到目标。

**解决**：用户脚本调用多个不同的底层函数，bug 藏在其中一个。

**效果**：agent 需要逐一排查每个底层调用。

**适用**：所有框架。示例：
- PyTorch:BatchNorm + GroupNorm + LayerNorm
- NumPy：多个 BLAS 调用
- SQL：多个 JOIN / 子查询

### 技巧 4：用条件触发替代始终触发

**问题**：bug 始终触发，agent 通过简单对比就能发现。

**解决**：bug 只在特定条件下触发：
- 输入大小/形状满足特定条件
- 数据值在特定范围内
- 特定的执行路径/分支

**效果**：agent 需要测试多种场景才能发现所有 bug。

**适用**：所有框架。示例：
- CUDA: `if (blockIdx.x > 32)` / `if (var < 0.1)`
- C: `if (size > PAGE_SIZE)` / `if (ptr == NULL)`
- SQL: `WHERE` 条件满足时 / 大表 JOIN 时

### 技巧 5：用隐晦症状替代显眼症状

**问题**：崩溃/NaN/segfault 太显眼，agent 一轮检查就能找到。

**解决**：用隐晦的症状：
- 训练收敛慢(不是不收敛)
- 结果精度差(不是完全错误)
- 性能下降(不是崩溃)
- 偶发错误(不是必现)

**效果**：agent 需要更深入的分析才能发现问题。

**适用**：所有框架。示例：
- ML：accuracy 从 95% 降到 80%(不是 NaN)
- SQL：查询结果少了几行(不是报错)
- Redis：数据偶尔乱码(不是每次)

### 技巧 6：用常量修改替代符号修改

**问题**：符号修改(如 `+=` → `-=`)太明显，grep 一步找到。

**解决**：常量修改(如 `eps * 1.001` / `buffer_size - 1`)看起来像正常代码。

**效果**：agent 需要理解数学公式/业务逻辑才能发现异常。

**适用**：所有框架。示例：
- 数值计算：精度常量、阈值
- 内存管理：缓冲区大小、对齐方式
- SQL：优化器参数、索引选择

### 技巧 7：用可编译诱饵替代注释诱饵

**问题**：注释诱饵太容易排除。

**解决**：诱饵用看起来像真 bug 的代码：
```
// 不好的诱饵(纯注释):
// float eps = 0.01f;  // FIXME

// 好的诱饵(看起来像真 bug):
float bn_eps = 0.01f;  // 在不影响功能的位置
```

**适用**：所有编译型语言(C/C++/Rust/CUDA/。。。)

### 技巧 8：测试多种场景覆盖多个 bug

**问题**：单一测试场景无法覆盖所有 bug。

**解决**：测试多种场景：
- 不同输入大小/形状
- 不同参数组合
- 不同执行路径
- 边界条件

**效果**：每个场景覆盖不同的 bug，agent 必须全部修复。

### 技巧 9：Bug 触发条件必须不同

**问题**：如果所有 bug 都在同一条件下触发，agent 修一个就全修了。

**解决**：每个 bug 的触发条件不同：
- Bug A：所有输入
- Bug B：特定输入大小
- Bug C：特定数据值

**效果**：agent 必须分别测试每个条件才能发现所有 bug。

### 技巧 10：用质量指标替代崩溃检查

**问题**：崩溃检查太容易通过(只要不崩溃就行)。

**解决**：用质量指标(如 accuracy、查询正确率、数据完整性)，agent 必须真正修复问题。

**适用**：所有框架。示例：
- ML:accuracy ≥ 80%
- SQL：结果行数正确率 ≥ 95%
- Redis：数据一致性 100%

---

## 难度调节公式

```
难度 = bug 数量 × 触发条件差异 × 定位难度 × 诱饵质量
```

| 因子 | 低难度 | 高难度 |
|------|--------|--------|
| bug 数量 | 1 个 | 2-3 个 |
| 触发条件 | 所有输入 | 不同条件 |
| 定位难度 | 崩溃(显眼) | 精度差(隐晦) |
| 诱饵质量 | 纯注释 | 可编译代码 |
| git 历史 | 可用 | 已删除 |
| 封装层数 | 单一 | 多层 |
| 测试场景 | 单一 | 多种 |

**调节方法**：
- 太简单 → 增加 bug 数量、删除 git、用隐晦症状替代崩溃
- 太难 → 减少 bug 数量、提供提示、简化触发条件

---

## 各框架特有的技巧

### PyTorch / TensorFlow (ML 框架)

- **多层归一化**：BatchNorm + GroupNorm + LayerNorm，模糊梯度定位
- **梯度检查**：对比 CPU vs CUDA 梯度，检测 GPU-only bug
- **固定数据模式**：用固定 seed 使结果可复现
- **前向 vs 反向**：bug 在 backward 更隐蔽(forward hook 无效)

### NumPy / BLAS (数值计算)

- **精度对比**：float32 vs float64 结果差异
- **内存布局**：C-order vs Fortran-order
- **BLAS 调用**：OpenBLAS vs MKL 行为差异

### SQL 数据库

- **查询计划**：EXPLAIN 分析优化器选择
- **索引覆盖**：有索引 vs 无索引结果不同
- **事务隔离**：并发场景下的数据一致性

### C/C++ 扩展

- **内存工具**：Valgrind / AddressSanitizer 检测内存问题
- **线程工具**：ThreadSanitizer 检测竞态条件
- **UB 检测**：UndefinedBehaviorSanitizer 检测未定义行为

### Rust FFI

- **unsafe 代码**：bug 藏在 unsafe 块中
- **生命周期**：悬垂引用 / use-after-free

### SQL 数据库 (补充：诱饵设计)

SQL 类任务不能用编译型诱饵。替代方案：

**诱饵类型**：
1. **错误的索引提示**：添加无用的 `USE INDEX` / `FORCE INDEX`
2. **死代码 SQL**：添加永远不会执行的子查询分支
3. **冗余条件**：添加 `WHERE 1=1` 或 `AND col IS NOT NULL`（不影响结果但看起来可疑）
4. **错误的类型转换**：添加不必要的 `CAST` / `::type`
5. **可疑的注释**：SQL 注释中的"修复"标记

**示例**：
```sql
-- 不好的诱饵(纯注释):
-- FIXME: index selection might be wrong

-- 好的诱饵(看起来像真 bug):
SELECT * FROM t USE INDEX (idx_col) WHERE ...  -- 添加了无用的索引提示
SELECT * FROM t WHERE col = 1 AND col IS NOT NULL  -- 冗余条件
```

### JAX / XLA (补充：诱饵设计)

JAX 类任务的 bug 在 Python batching rule 或 XLA lowering 中。诱饵设计：

**诱饵类型**：
1. **错误的 axis 参数**：在 `np.moveaxis` / `np.expand_dims` 中用错误的 axis
2. **冗余 reshape**：添加不必要的 `reshape` 操作
3. **死代码分支**：添加 `if False:` 包裹的代码
4. **可疑的常量**：在计算中添加不影响结果的常量

**示例**：
```python
# 不好的诱饵(纯注释):
# FIXME: batch dimension might be wrong

# 好的诱饵(看起来像真 bug):
x = jnp.moveaxis(x, 0, -1)  # 在不影响结果的位置添加
x = x.reshape(x.shape)  # 冗余 reshape
```

---

## 概率性 Bug 的测试策略

有些 bug 不是每次都触发(如 refcount、GIL、内存时序)。这类 bug 的测试需要特殊设计。

### 问题：偶发错误如何检测？

如果 bug 只在 1% 的运行中触发，单次测试无法可靠检测。

### 策略 1：多次运行取统计值

```bash
# 运行 100 次,检查失败率
fail_count=0
for i in $(seq 1 100); do
    result=$(run_test 2>&1)
    if echo "$result" | grep -q "error\|segfault\|wrong"; then
        fail_count=$((fail_count + 1))
    fi
done
# 如果失败率 > 10%,说明 bug 存在
if [ $fail_count -gt 10 ]; then
    echo "FAIL: $fail_count/100 runs failed"
fi
```

### 策略 2：增加触发概率

如果 bug 依赖特定条件(如 GC 时序、线程调度)，可以通过以下方式增加触发概率：

- **增加迭代次数**：从 1000 次增加到 100000 次
- **增加并发度**：从 2 线程增加到 16 线程
- **减少内存**：让 GC 更频繁触发
- **添加延迟**：在关键位置添加 `sleep` 或 `yield`

### 策略 3：检查中间状态

即使 bug 不每次都触发，可以检查中间状态是否异常：

```bash
# 检查内存使用是否持续增长(内存泄漏)
for i in $(seq 1 100); do
    run_test
    mem=$(get_memory_usage)
    if [ $mem -gt $threshold ]; then
        echo "FAIL: memory leak detected at iteration $i"
    fi
done
```

### 策略 4：使用检测工具

对于 C/C++ 类 bug，可以使用工具检测：
- **Valgrind**：检测内存错误
- **AddressSanitizer**：检测 use-after-free、buffer overflow
- **ThreadSanitizer**：检测数据竞争
- **UndefinedBehaviorSanitizer**：检测未定义行为

```bash
# 用 AddressSanitizer 编译并运行
gcc -fsanitize=address -o test test.c
./test  # 如果有内存错误,会直接报错
```

### 策略 5：对比多次运行结果

如果 bug 导致结果不确定(如数据竞争)，可以对比多次运行的结果：

```bash
# 运行 10 次,检查结果是否一致
result1=$(run_test)
result2=$(run_test)
if [ "$result1" != "$result2" ]; then
    echo "FAIL: results differ between runs"
fi
```

### 各任务的概率性 bug 策略

| 任务 | Bug 类型 | 触发条件 | 推荐策略 |
|------|---------|---------|---------|
| task5 (refcount) | use-after-free | GC 时序 | 策略 2(增加迭代) + 策略 4(ASan) |
| task8 (Rust FFI) | 生命周期错误 | 内存布局 | 策略 2(增加迭代) + 策略 4(Miri) |
| task12 (GIL) | 数据竞争 | 线程调度 | 策略 2(增加并发) + 策略 5(对比结果) |

---

## 从 task1 验证的关键教训

以下是 task1 开发过程中反复验证的教训，适用于所有题目。

### 教训 1：单一 bug 必然被秒杀

**task1 数据**：单一符号翻转 bug，Kimi 41 步修完(步数 246 的 17%)。

**原因**：agent 的梯度对比 + 源码搜索能力太强，单一 bug 无法制造足够的排查路径。

**通用原则**：任何单一 bug，无论多隐蔽，都会被 agent 快速定位。必须用复合 bug。

### 教训 2：复合 bug 让难度指数级上升

**task1 数据**：3 个复合 bug，Kimi 246 步只修了 2/3(步数增加 6 倍)。

**原因**：
- 修了 A 还有 B，agent 以为修完了但测试仍然失败
- 每个 bug 需要不同的调试技能
- agent 容易在修完前 2 个后放弃

**通用原则**：2-3 个 bug 是最佳数量。太少(1 个)被秒，太多(4+)agent 直接放弃。

### 教训 3：Bug 的触发条件必须不同

**task1 数据**：
- Bug 1：所有数据都触发(梯度符号翻转)
- Bug 2：方差 < 0.1 时触发(条件 eps)
- Bug 3:blockIdx.x > 32 时触发(条件 NaN)

**效果**：agent 修完 Bug 1+2 后，小 batch 测试通过，但大 batch 仍然 NaN。

**通用原则**：如果所有 bug 都在同一条件下触发，agent 修一个就全修了。必须让每个 bug 在不同的输入/状态/条件下触发。

### 教训 4：修改参数(eps * N)对训练没有影响

**task1 数据**：eps * 1000 甚至 eps * 100000，模型仍然 100% accuracy。

**原因**：模型有多个归一化层(BatchNorm + GroupNorm)，能补偿 LayerNorm 的弱点。修改 eps 只影响归一化强度，不影响梯度方向。

**通用原则**：修改常量/参数类 bug 往往无效，因为系统的其他部分能补偿。必须用影响核心逻辑的 bug(如符号错误、条件错误)。

**适用场景**：
- ML：修改 eps/learning_rate/batch_size 无效(其他层能补偿)
- SQL：修改优化器参数无效(可能走其他执行计划)
- C：修改缓冲区大小可能无效(其他路径能处理)

### 教训 5：CPU/GPU 同时修改的方案行不通

**task1 数据**：尝试让 CPU 和 GPU 都有同样的 bug(使梯度检查无法检测)，但模型仍然 100% accuracy。

**原因**：模型太鲁棒，即使底层实现完全错误，其他层/路径能补偿。

**通用原则**：不能靠"让所有路径都有同样的 bug"来隐藏 bug。系统的鲁棒性会让 bug 无效。必须用 GPU-only / 特定路径的 bug。

**适用场景**：
- ML：CPU 和 GPU 都有 bug → 模型仍能训练
- SQL：所有执行计划都有 bug → 查询仍能返回结果
- C：所有代码路径都有 bug → 程序仍能运行

### 教训 6：诱饵必须看起来像真 bug

**task1 数据**：纯注释诱饵，agent 直接忽略。

**原因**：agent 能区分"代码"和"注释"，注释不会被当作 bug。

**通用原则**：诱饵必须是看起来像真 bug 的代码，放在不影响功能的位置。

**好的诱饵**：
- 声明但不使用的变量(`float eps = 0.01f;`)
- 在死代码分支中的修改(`if (false) { ... }`)
- 在不影响结果的位置的常量修改

**坏的诱饵**：
- 纯注释(`// FIXME: this might be wrong`)
- 在明显正确位置的修改
- 与真 bug 无关的修改

### 教训 7：测试必须覆盖多种场景

**task1 数据**：只测小 batch → agent 修了 Bug 1+2 就认为修完了(但 Bug 3 还在)。

**通用原则**：每种测试场景覆盖不同的 bug。必须测试：
- 不同输入大小/形状
- 不同参数组合
- 边界条件
- 不同执行路径

**task1 示例**：
- 小 batch(16) accuracy 检查 → 覆盖 Bug 1+2
- 大 batch(64) 无 NaN 检查 → 覆盖 Bug 3

### 教训 8：用质量指标替代崩溃检查

**task1 数据**：NaN 检查太容易通过(只要不 NaN 就行)。改为 accuracy ≥ 30% 后，agent 必须真正修复训练问题。

**通用原则**：崩溃/NaN/segfault 是显眼的症状，agent 能快速定位。用质量指标(accuracy、正确率、性能)更难检测。

**适用场景**：
- ML：accuracy ≥ 80% 替代"无 NaN"
- SQL：结果正确率 ≥ 95% 替代"不报错"
- Redis：数据一致性 100% 替代"不崩溃"

### 教训 9：删除 .git 是必须的

**task1 数据**：有 .git 时，Kimi 用 `git show <commit>` 直接看到所有改动。

**通用原则**：任何版本控制系统的历史都会暴露 bug。必须在构建后删除 .git / .svn / .hg。

**实现**：Dockerfile 最后加 `rm -rf <源码目录>/.git`。

### 教训 10：分析轨迹找到"秒杀"路径

**task1 数据**：
- 第 1 版(单一 bug)：Kimi 41 步修完 → 分析轨迹发现用 git show 秒杀
- 第 2 版(删除 .git)：Kimi 78 步修完 → 分析轨迹发现用梯度对比秒杀
- 第 3 版(复合 bug)：Kimi 246 步只修 2/3 → 成功

**通用原则**：每次校准后，分析 agent 轨迹，找到最快的解决路径，然后堵死它。

**分析方法**：
1. 看 agent 的前 10 步做了什么(通常是读文件、运行脚本)
2. 找到 agent 定位 bug 的关键步骤(通常是 grep / 对比 / 测试)
3. 在那个步骤加障碍(删除 git / 增加诱饵 / 改变触发条件)

### 教训 11：每次运行前用 oracle 验证 test.sh

**task1 数据**：test.sh 从 accuracy 检查改为梯度检查，发现 accuracy 检查不可靠(随机数据方差太大)。oracle fix 后 accuracy 也只有 18-34%，无法区分 bug 存在和修复。

**通用原则**：每次跑 agent 测试前，必须先用 oracle fix 验证 test.sh 能给满分。

**验证方法**：
```bash
docker run --rm --gpus all \
  -v ... \
  task_image bash -c "
    bash /task/solution/solve.sh  # 应用 oracle fix
    bash /task/tests/test.sh      # 应该输出 1.0
  "
```

**如果 oracle 分数不是 1.0，说明 test.sh 有 bug，必须先修复！**

常见问题：
- accuracy 阈值不合理(随机数据方差大)→ 改用梯度检查
- 梯度检查阈值太严/太松 → 用 oracle 测试确定合适阈值
- anti-hack 误报 → 检查 grep 模式是否过于宽泛
- 超时 → 减少训练步数或增加超时时间

### 教训 12：禁止 agent 访问 GitHub 下载参考代码

**task4 数据**：Kimi 用 `curl` 从 GitHub 下载 JAX 源码对比，直接找到所有 bug。

**通用原则**：agent 可能用 `curl`/`wget` 从 GitHub 下载参考实现，对比找到 bug。必须在 Docker 运行时封锁外部网络。

**实现**：
```bash
docker run \
  --add-host="github.com:127.0.0.1" \
  --add-host="raw.githubusercontent.com:127.0.0.1" \
  --add-host="codeload.github.com:127.0.0.1" \
  --add-host="objects.githubusercontent.com:127.0.0.1" \
  --add-host="pypi.org:127.0.0.1" \
  --add-host="files.pythonhosted.org:127.0.0.1" \
  ...
```

**注意**：需要同时封锁 `raw.githubusercontent.com` 和 `objects.githubusercontent.com`，否则 agent 可以绕过。

### 教训 13：删除 .gitignore 防止时间戳泄露

**task4 数据**：Kimi 用 `find . -name '*.py' -newer .gitignore` 找到被修改的文件。

**通用原则**：`.gitignore` 的时间戳可以作为参考点，agent 可以用 `find -newer` 找到被修改的文件。必须删除 `.gitignore`。

**实现**：
```dockerfile
RUN rm -rf /build/source/.git /build/source/.gitignore
```

### 教训 14：touch 所有文件标准化时间戳

**task4 数据**：Kimi 用 `find . -newermt '2026-06-24 13:33'` 按绝对时间找被修改的文件。

**通用原则**：即使删除了 `.gitignore`，agent 仍可用绝对时间戳找被修改的文件。必须 touch 所有文件让时间戳一致。

**实现**：
```dockerfile
RUN find $SOURCE_DIR -name "*.py" -exec touch {} +
```

### 教训 15：patch 注入取代 inline，但有自己的坑

**task1 inline 时代数据**：inject_bug.py 用 `return (input > 0) ? input : input * negval` 但实际源码是 `return aop > opmath_t(0) ? aop : aop * negval`，模式拼错导致 bug 没注入——而且**静默失败**(没报错、build 照样过、oracle 才发现)。task1 patch 化时确实查出 Bug22/25 + 5 个陷阱诱饵的锚点从来没匹配上、一直是哑改。

**通用原则**：改用 unified diff patch(见 Phase 4)。锚点由 `clean_src` 真实源码生成，天生对齐，根除"模式拼错"这类哑改。但 patch 有自己的新坑：
- **patch fuzz / 偏移**：`patch` 默认允许 fuzz(模糊匹配)，源码上下文轻微不符时会**静默命中错位置、returncode 仍是 0**。最容易踩的是删除型 bug(上下文只有几行)。**必须 `-F0`(fuzz=0)关闭模糊匹配 + 断言无 `.rej` 残留**——这样任一 hunk 上下文不精确匹配就 reject + 非 0 退出，等于把 patch 引擎变成对**全部 hunk 的精确落地校验**(见教训 17)。
- **marker 抽样只作冒烟**：注入后用几个 buggy 态独有串(`BUG_MARKERS`)抽查，作冒烟即可，**不能当主防线**——删除型 bug 删完没有独有新文本，天生写不出可靠 marker，主防线只能是 fuzz=0。
- **陷阱诱饵不能和 bug 改同一行**：诱饵在 bug 之后应用会改变该行文本，导致 `bugs.patch` 反向(solve)失败。诱饵层与 bug 层必须互不重叠。
- **patch 用完即删**：build 后 `rm *.patch`，否则 agent 直接读 patch 就是答案地图。

### 教训 16：oracle 测试必须验证 buggy 版本失败

**task4 数据**：buggy 版本也得 1.0 分，说明 bug 没被 test 检测到。

**通用原则**：oracle 测试必须分两步：
1. 测试 buggy 版本 → 分数应该 < 1.0
2. 测试 fixed 版本 → 分数应该 = 1.0

如果 buggy 版本也得 1.0，说明 test 没有检测到 bug，必须加强 test。

### 教训 17：patch 注入要用 fuzz=0 做全检测，别靠 marker 抽样

**task1 数据**：`inject_bug.py` 原来用 `patch -f`(默认允许 fuzz)，只抽 5 个 buggy 独有串作 marker 校验。问题：`patch` 在上下文轻微不符时会**模糊命中错位置、returncode 仍是 0**，而 marker 只覆盖 5/35 个 bug——放过的恰好是 16 个删除型 `__syncthreads` bug(上下文最短、最易 fuzz 错位，又没有独有新文本能写 marker)。

**通用原则**：把"每个 hunk 精确落地"的校验交给 patch 引擎本身，而不是事后抽样：
```python
# 主防线:fuzz=0,任一 hunk 上下文不精确匹配即 reject + 非 0 退出
cmd = ["patch", "-d", SRC, "-p0", "-f", "-F0", "--no-backup-if-mismatch"]
# 双保险:解析 patch 的 +++ 目标文件,确认无 <file>.rej 残留
# 冒烟:再抽几个 buggy 独有串断言在/回退后消失(可选)
```
- `-F0` 把 35 处 hunk 变成"全员精确匹配才算成功"，等价于全检测，且不用维护 marker 列表。
- `-F0` 只禁 fuzz、仍容忍纯行号偏移 → 源码小改不会误杀。
- 删除型 bug 删完无独有新文本 → 写不出可靠 marker → 更说明主防线必须是 fuzz=0，marker 只能是冒烟。

### 教训 18：诱饵必须和真 bug 使用相同模式

**task1 数据**：诱饵用 `if (mean_val > T_ACC(100))` 防御代码，agent 直接忽略。真 bug 用 `* (T_ACC(1) - T_ACC(0.5) * condition)` 条件乘法，agent 一眼认出。

**通用原则**：诱饵必须和真 bug 使用完全相同的代码模式，只是条件/参数不同。agent 无法区分真假。

**好的诱饵**：
```c
// 真 bug: sigma2 ∈ [0.99, 1.01] 时乘以 0.5
result *= (1.0 - 0.5 * (sigma2 > 0.99 && sigma2 < 1.01));
// 诱饵: sigma2 ∈ [0.49, 0.51] 时乘以 0.4 (不同条件,不同参数)
result *= (1.0 - 0.4 * (sigma2 > 0.49 && sigma2 < 0.51));
```

### 教训 19：警惕"哑弹 bug"——注入了但端到端测不出

**task1 数据**：第一次跑 kimi 拿 1.00 满分，却只修了 14/35 个 bug。漏掉的 Softmax/BN/Dropout/GELU bug 全是**哑弹**——bug 确实注入并编译进去了，但默认测试路径根本不触发它们：
- GELU 默认走 erf 近似，tanh 分支的 bug 不进
- 模型 `.eval()` 跑，BatchNorm 的 `running_var` 更新、Dropout 缩放都被关掉
- 没有 4D spatial / 大 dim 的 softmax，`__syncthreads` 删除的竞争条件不暴露

oracle(整体 buggy 失败)能过，是因为别的"带电"bug 把分拉下去了，掩盖了哑弹。

**通用原则**：bug 注入 ≠ bug 生效。每个 bug 必须有一条测试路径真正触发它的代码分支。解决：
- **kernel/算子级带电检查**：绕过 `.eval()`/默认算子，直接构造能命中该分支的输入(task1 `test.sh` 的 `[6/8] 带电检查`：显式 `approximate='tanh'`、4D `softmax(dim=1)`、`.train()` forward 等)，按子项给分。
- **必要时改用户脚本配合带电**：如 task1 把 `model.py` 两处 `nn.GELU()` 改成 `nn.GELU(approximate='tanh')`，让 GELU bug 端到端也带电。

### 教训 20：per-bug oracle——逐个验证每个 bug 都带电(按成本取舍)

**task4 做法**：`oracle_per_bug.py` 从 clean 态出发，**单独注入每一个 bug → 跑 test → 回退**，要求每个 bug 单独存在时 test 都不满分。这是教训 16(整体 oracle)抓不到的：整体 oracle 只验证"全部 bug 在场时失败"，而 per-bug oracle 直接证明"没有哑弹、test 覆盖了每一个 bug"——正是教训 19 的根因检测器。

```python
# 1. reverse 完整 bugs.patch → clean
# 2. for each per_bug_patches/Bug_N.patch:
#      apply 单个 patch → 跑 test(应 <1.0)→ reverse 回 clean
# 3. 重新 apply 完整 bugs.patch
```

**按成本取舍(重要)**：per-bug oracle 要为每个 bug 跑一遍 test，**是否值得取决于单个 bug 的迭代成本**：
- ✅ **适合 task4 这类**：patch 应用 + 跑 python test 是秒级，N 个 bug 全测也就几分钟，直接全覆盖。
- ❌ **不适合 task1 这类**：每改一个 CUDA bug 都要 `ninja` 增量编译(分钟级)，35 个 bug 逐个编译+测试要数小时，代价过高。此时**不必逐个 per-bug oracle**，改用更便宜的等价手段：
  - 静态断言每处 patch 落地(`BUG_MARKERS`/grep 注入后的特征串)，保证没哑改(教训 15)；
  - 用**带电检查按子项给分**(教训 19)间接确认每类 bug 都被触发，而不是逐 bug 重编译。

**判据**：`单 bug 重编译成本 × bug 数` 在可接受范围(≈几分钟) → 上 per-bug oracle；否则退化为"静态落地断言 + 分项带电检查"。

**坏的诱饵**：
```c
// 看起来像防御代码,agent 直接忽略
if (mean_val > T_ACC(100)) { mean_val = T_ACC(100); }
```

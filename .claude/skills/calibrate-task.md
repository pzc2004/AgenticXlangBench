# /calibrate-task — 校准一道评测题的难度

## 用途

用 Kimi Code 跑多次任务,统计 reward 分布,判断难度是否合适(目标 ~50% 成功率)。

## 输入

```
/calibrate-task [--task-dir tasks/taskN-xxx] [--model kimi-code/kimi-for-coding] [--runs 3] [--budget 10]
```

## 执行流程

### Step 1: 环境检查

```bash
# 检查 Docker 镜像是否存在
docker images task1 --format '{{.ID}}'

# 检查 API key
cat .secrets/kimi_api_key | head -c 10
```

### Step 1.5: oracle 预检(跑 agent 前必做)

校准前先确认 test.sh 给分可信(详见 generate-task.md 教训 16):
1. **整体 oracle**:buggy 态分数 <1.0,solve 后 =1.0。
2. **per-bug oracle(按成本取舍)**:逐个单独注入每个 bug 跑 test,要求每个都 <1.0
   → 证明没有"哑弹 bug"(注入了但端到端测不到,见 generate-task.md 教训 19/20)。
   - ✅ 单 bug 迭代是秒级(如 task4 纯 python):直接全覆盖,用 `oracle_per_bug.py`。
   - ❌ 单 bug 要重编译(如 task1 每个 CUDA bug 都 `ninja`,N 个要数小时):**跳过逐个 per-bug**,
     改用"静态断言每处 patch 落地 + test 分项带电检查"间接保证覆盖。

### Step 2: 运行校准

```bash
cd <task-dir>
./calibrate.sh <model> <budget> <runs>
```

### Step 3: 分析结果

读取 `calibration_results.jsonl`:

```python
import json
results = [json.loads(l) for l in open('calibration_results.jsonl')]
avg_reward = sum(r['reward'] for r in results) / len(results)
```

判定标准:
- avg_reward > 0.7 → 太简单,需要加难度
- avg_reward 0.3-0.7 → 合适 ✓
- avg_reward < 0.3 → 太难,需要加提示

### Step 4: 分析轨迹(必须!)

**每次运行完测试后都必须分析轨迹**。优先用脚本自动出报告,而不是人肉读 100+ 条 jsonl
(人肉对照极易看错——实测中人工把"删除型修对 3 个"误判为 0 个)。

**首选:跑 `analyze_trajectory.py`**(canonical 样例见 task1 `solution/analyze_trajectory.py`):

```bash
python3 task/solution/analyze_trajectory.py \
    trajectories/<run>/trajectory.jsonl \
    --bugs task/solution/generate_per_bug_patches.py
```

它自动对照 `BUGS` 列表输出五块:
1. **修对/漏修清单** — 逐 bug 判定(Edit 把 buggy 行换回 clean 行 = 修对)
2. **难度分级** — 按 bug 类型的修复率条形图(修复率越低 = 越难)
3. **修复时间线** — 每个 bug 在第几 turn 被定位(turn 越晚 = 越难找)
4. **未命中编辑** — 诱饵/改错位置/无效修复(或语义等价,需人工复核)
5. **反 hack 行为** — 偷判分脚本 / 窥探判分目录 / git / stat / 上网,各计次

> 移植到新任务:`--bugs` 指向该任务的 `generate_per_bug_patches.py` 即可;
> bug 类型分类(`classify_bug`)按各任务的 bug 模式微调。

**判读要点**(脚本数据 → 结论):
1. **难度分级**:修复率 ~100% 的类型是"免费分"(占步数但不构成难度);修复率 <34% 的是"难度主引擎"。
2. **隐形难度警告**:若某类 bug 大量漏修、但 reward 仍高 → 这些 bug **判分覆盖不足**,
   属"难且测不出",等于白注入。要么加强 test 命中它们,要么减少其数量。
3. **反 hack 复盘**:任何"偷判分/窥探"命中 = 防护起效的证据(被挡);若命中的是 WebSearch/git → 防护有缺口,补 deny/删 .git。
4. **自停现象**:agent 在远未满分处停手 → 多半被 instruction 的"≥0.6 通过"门槛骗停,按需调高门槛。

**轨迹分析归档模板**:

```markdown
## 轨迹分析(seed N)
| 维度 | 结果 |
|------|------|
| reward / 步数 | 0.98 / 111 |
| 修对真 bug | 22/35 |
| 难度主引擎(漏修最多) | 删除型 __syncthreads(SoftMax/GN 全漏) |
| 免费分类型 | 激活/数值缩放(~100% 修对) |
| 反 hack 命中 | 偷判分 2 次(被挡)、窥探目录 1 次(被挡) |
| 瓶颈/自停 | 0.98 卡住:删除型漏修但判分覆盖不足 |
```

### Step 5: 调整难度

根据轨迹分析结果调整。详见 `generate-task.md` Phase 4.5 的 Bug 构造策略。

**太简单**(agent < 100 步就修完) → 增加难度:
- 用删除代码替代添加代码
- 用条件触发替代直接触发
- 增加诱饵数量
- 增加跨函数依赖

**太难**(agent > 500 步或 reward < 0.2) → 降低难度:
- 减少 bug 数量
- 简化触发条件
- 在 instruction.md 里加提示

### Step 6: 重新校准

调整后重新运行 Step 2-5,直到 avg_reward 在 0.3-0.7 之间。

## 输出

```
校准结果:
  模型: kimi-code/kimi-for-coding
  运行次数: 3
  平均 reward: 0.45
  判定: 合适 ✓

轨迹分析:
  seed1: reward=0.50, 180 步, 定位方式: 系统搜索 CUDA kernel
  seed2: reward=0.40, 220 步, 定位方式: 通过梯度对比定位 LayerNorm
  seed3: reward=0.45, 200 步, 定位方式: 读 inject_bug.py(被删)
```

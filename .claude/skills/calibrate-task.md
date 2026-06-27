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

**每次运行完测试后都必须分析轨迹**,理解 agent 的行为模式。

```bash
# 查看轨迹目录
ls trajectories/

# 统计步数
wc -l trajectories/*/trajectory.jsonl

# 分析 Kimi 的推理链
python3 -c "
import json
with open('trajectories/<run>/trajectory.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
steps = []
for line in lines:
    if line.get('role') == 'assistant' and line.get('tool_calls'):
        tc = [t['function']['name'] for t in line['tool_calls']]
        steps.append(tc)
for i, tc in enumerate(steps):
    print(f'[{i:2d}] {tc}')
"
```

**分析要点**:
1. **定位路径**:agent 怎么找到 bug 的?是系统搜索还是偶然发现?
2. **关键转折**:在哪一步发现了 bug?花了多少步?
3. **失败原因**:如果 reward < 0.6,卡在哪里?为什么?
4. **hack 尝试**:有没有尝试绕过约束?
5. **上网搜索**:有没有用 WebSearch/WebFetch?(如果用了,需要加 deny 规则)

**轨迹分析模板**:

```markdown
## 轨迹分析

| 维度 | 结果 |
|------|------|
| 总步数 | xxx |
| 定位方式 | 系统搜索 / 偶然发现 / 提示引导 |
| 关键转折 | 第 N 步发现 xxx |
| 修复方式 | 修改了 xxx 文件的 xxx 行 |
| 失败原因 | (如果失败)卡在 xxx |
| hack 尝试 | 有/无 |
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

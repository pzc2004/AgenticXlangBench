# /calibrate-task — 校准一道评测题的难度

## 用途

用 Kimi Code 跑多次任务,统计 reward 分布,判断难度是否合适(目标 ~50% 成功率)。

## 输入

```
/calibrate-task [--task-dir tasks/taskN-xxx] [--model kimi-code/kimi-for-coding] [--runs 3] [--budget 10]
```

默认值:
- task-dir: 当前目录下的第一个 taskN-* 文件夹
- model: kimi-code/kimi-for-coding
- runs: 3
- budget: 10 (美元)

## 执行流程

### Step 1: 环境检查

```bash
# 检查 Docker 镜像是否存在
docker images task1 --format '{{.ID}}'

# 检查 API key
cat .secrets/kimi_api_key | head -c 10
```

如果镜像不存在,先用 `/generate-task` 生成任务。

### Step 2: 清理旧轨迹

```bash
cd <task-dir>/trajectories
rm -rf */
> calibration_results.jsonl
```

### Step 3: 运行校准

```bash
cd <task-dir>
./calibrate.sh <model> <budget> <runs>
```

`calibrate.sh` 内部流程:
1. 循环 `runs` 次,每次不同 seed
2. 调用 `run.sh` 运行一次:
   - 启动 Docker 容器(不销毁)
   - 运行 Kimi Code
   - `docker commit` 保存修复状态
   - 在快照容器里跑 `test.sh`
   - 记录 reward
3. 汇总结果到 `calibration_results.jsonl`

### Step 4: 分析结果

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

### Step 5: 分析轨迹

如果 reward 不合适,需要分析轨迹找到原因:

```bash
# 查看轨迹
ls trajectories/
wc -c trajectories/*/trajectory.jsonl

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

### Step 6: 调整难度

根据轨迹分析结果调整。**核心原则**:找到 agent 的"秒杀"路径,然后堵死它。

**太简单**(agent < 100 步就修完) → 增加难度:
1. **删除 .git 历史**:防止 `git show` 直接看到改动
2. **用复合 bug**:注入 2-3 个 bug,每个在不同条件下触发
3. **用隐晦症状替代崩溃**:如精度差、收敛慢、偶发错误
4. **用多层封装模糊定位**:多个底层调用,bug 藏在其中一个
5. **用条件触发**:bug 只在特定输入/条件下触发
6. **用可编译诱饵**:不只是注释,是看起来像真 bug 的代码

**太难**(agent > 500 步或 reward < 0.2) → 降低难度:
1. **减少 bug 数量**:从 3 个减到 1-2 个
2. **简化触发条件**:从条件触发改为始终触发
3. **在 instruction.md 里加提示**:暗示 bug 所在的层或类型
4. **简化封装结构**:减少中间层
5. **提供调试工具**:允许 agent 使用特定的调试脚本

**关键经验**:
- 单一 bug 太容易被秒,复合 bug 才有挑战性
- 诱饵用纯注释太容易排除,要用看起来像真 bug 的代码
- 测试必须覆盖多种场景,否则 agent 修了 A 还有 B
- Bug 触发条件必须不同,否则修一个就全修了
- 用质量指标(accuracy/正确率)替代崩溃检查,更难通过

### Step 7: 重新校准

调整后重新运行 Step 3-6,直到 avg_reward 在 0.3-0.7 之间。

## 输出

```
校准结果:
  模型: kimi-code/kimi-for-coding
  运行次数: 3
  平均 reward: 0.45
  判定: 合适 ✓

轨迹:
  seed1: reward=0.50, 180 步, 修复方式: ...
  seed2: reward=0.40, 220 步, 修复方式: ...
  seed3: reward=0.45, 200 步, 修复方式: ...
```

## 参考

Task 1 校准结果见 `tasks/task1-pytorch-cuda-index/README.md` §校准结果。

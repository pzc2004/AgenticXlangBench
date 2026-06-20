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

根据分析结果调整:

**太简单** → 增加难度:
- 增加诱饵数量
- 让诱饵更像真 bug(真实代码改动,不只是注释)
- 隐藏 bug 所在的 op(用更深层的封装)
- 减少提示(instruction.md 不暴露细节)

**太难** → 降低难度:
- 减少诱饵数量
- 在 instruction.md 里加提示
- 简化模型结构
- 降低 bug 的触发条件

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

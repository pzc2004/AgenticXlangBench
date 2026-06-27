# /select-tasks — 从 N 道题里选最优 M 道

## 用途

从校准过的 15 道题里，选出难度最合适(成功率 ~50%)的 10 道，组成最终评测集。

## 输入

```
/select-tasks [--n 15] [--m 10] [--target-success-rate 0.5]
```

默认值：
- n： 15 (总题数)
- m： 10 (选几道)
- target-success-rate: 0.5 (目标成功率)

## 执行流程

### Step 1： 收集所有校准结果

```bash
# 遍历所有任务目录,读取 calibration_results.jsonl
for task_dir in tasks/task*/; do
    result_file="$task_dir/trajectories/calibration_results.jsonl"
    if [ -f "$result_file" ]; then
        echo "$(basename $task_dir):"
        cat "$result_file"
    fi
done
```

### Step 2： 计算每个任务的统计指标

```python
import json
import glob

tasks = []
for f in glob.glob('tasks/task*/trajectories/calibration_results.jsonl'):
    results = [json.loads(l) for l in open(f) if l.strip()]
    task_name = f.split('/')[1]
    avg_reward = sum(r['reward'] for r in results) / len(results)
    std_reward = (sum((r['reward'] - avg_reward)**2 for r in results) / len(results)) ** 0.5
    avg_turns = sum(r.get('turns', 0) for r in results) / len(results)
    tasks.append({
        'name': task_name,
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'avg_turns': avg_turns,
        'runs': len(results),
    })
```

### Step 3： 排序并选题

```python
# 按"距离目标成功率"排序
target = 0.5
tasks.sort(key=lambda t: abs(t['avg_reward'] - target))

# 选前 M 道
selected = tasks[:10]
rejected = tasks[10:]
```

### Step 4： 生成报告

```
=== 选题结果 ===

入选(10 道):
  task1-pytorch-cuda-index:     reward=0.50, 180 步 ✓
  task3-numpy-blas-register:    reward=0.45, 250 步 ✓
  ...

淘汰(5 道):
  task11-llvm-codegen:          reward=0.20, 400 步 (太难)
  task7-cudnn-conv-backward:    reward=0.80, 100 步 (太简单)
  ...

统计:
  入选平均 reward: 0.48
  入选标准差: 0.08
  目标成功率: 50%
```

### Step 5： 复制入选任务到最终目录

```bash
mkdir -p final-tasks/
for task in "${selected[@]}"; do
    cp -r "tasks/$task" "final-tasks/"
done
```

## 输出

- `final-tasks/` 目录：包含入选的 10 道题
- `selection-report.md`：选题报告(每个任务的 reward、步数、淘汰原因)

## 参考

Task 1-15 的校准结果见各任务目录下的 `README.md`。

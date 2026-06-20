#!/bin/bash
# calibrate.sh — 用 Kimi Code 跑多次,校准任务难度
#
# 用法: ./calibrate.sh [model] [budget_usd] [num_runs]
# 示例: ./calibrate.sh kimi-for-coding 10 3
#
# 每次运行用不同 seed,保存轨迹到 trajectories/

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
NUM_RUNS="${3:-3}"
TIMEOUT="${4:-3600}"  # 默认 1 小时超时

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/trajectories/calibration_results.jsonl"

mkdir -p "$SCRIPT_DIR/trajectories"
> "$RESULTS_FILE"

TASK_NAME="$(basename "$SCRIPT_DIR")"

echo "========================================="
echo " 校准任务: $TASK_NAME"
echo " 模型:     $MODEL"
echo " 预算:     \$$BUDGET / 次"
echo " 运行次数: $NUM_RUNS"
echo "========================================="

for SEED in $(seq 1 $NUM_RUNS); do
    echo ""
    echo ">>> Run $SEED / $NUM_RUNS (seed=$SEED)"
    echo "-----------------------------------------"

    # 跑一次(run.sh 输出结果 JSON)
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

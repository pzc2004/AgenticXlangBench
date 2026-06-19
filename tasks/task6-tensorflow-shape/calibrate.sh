#!/bin/bash
# calibrate.sh — 多模型 × 多 seed 校准难度
# 用法: ./calibrate.sh [budget_usd_per_run]
# 示例: ./calibrate.sh 10

set -e

BUDGET="${1:-10}"
TASK_NAME="$(basename $(dirname "$0"))"
TASK_DIR="$(dirname "$0")/task"
RESULTS_FILE="$(dirname "$0")/trajectories/calibration_results.jsonl"

mkdir -p "$(dirname "$0")/trajectories"
> "$RESULTS_FILE"  # 清空旧结果

MODELS=("claude-haiku-4-5" "claude-sonnet-4-6" "claude-opus-4-7")
SEEDS=(1 2 3)

echo "========================================="
echo " 校准任务: $TASK_NAME"
echo " 模型: ${MODELS[*]}"
echo " Seeds: ${SEEDS[*]}"
echo " 预算: \$$BUDGET / 次"
echo " 总运行: $(( ${#MODELS[@]} * ${#SEEDS[@]} )) 次"
echo "========================================="

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> $MODEL seed=$SEED"
    TRAJ_DIR="$(dirname "$0")/trajectories/${MODEL}_seed${SEED}"
    mkdir -p "$TRAJ_DIR"

    # 设置 seed(通过 system prompt 注入)
    SEED_PROMPT="Seed: $SEED. $(cat "$TASK_DIR/instruction.md")"

    claude \
      -p "$SEED_PROMPT" \
      --output-format stream-json \
      --model "$MODEL" \
      --max-budget-usd "$BUDGET" \
      --dangerously-skip-permissions \
      --effort high \
      > "$TRAJ_DIR/trajectory.jsonl" 2>"$TRAJ_DIR/stderr.log" || true

    # 跑测试
    mkdir -p /logs/verifier
    bash "$TASK_DIR/tests/test.sh" 2>/dev/null || true
    REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo "0.0")
    TURNS=$(grep -c '"type":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
    TOOL_CALLS=$(grep -c '"type":"tool_use"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")

    # 记录结果
    echo "{\"task\":\"$TASK_NAME\",\"model\":\"$MODEL\",\"seed\":$SEED,\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS,\"trajectory\":\"$TRAJ_DIR/trajectory.jsonl\"}" \
      >> "$RESULTS_FILE"

    echo "  Reward=$REWARD Turns=$TURNS ToolCalls=$TOOL_CALLS"
  done
done

echo ""
echo "========================================="
echo " 校准结果汇总"
echo "========================================="

python3 -c "
import json, sys
from collections import defaultdict

results = [json.loads(l) for l in open('$RESULTS_FILE')]
by_model = defaultdict(list)
for r in results:
    by_model[r['model']].append(r)

print(f'{\"模型\":<25} {\"平均分\":<10} {\"平均轮数\":<10} {\"成功率\":<10}')
print('-' * 55)
for model in sorted(by_model.keys()):
    rs = by_model[model]
    avg_reward = sum(r['reward'] for r in rs) / len(rs)
    avg_turns = sum(r['turns'] for r in rs) / len(rs)
    success_rate = sum(1 for r in rs if r['reward'] >= 0.8) / len(rs) * 100
    print(f'{model:<25} {avg_reward:<10.2f} {avg_turns:<10.0f} {success_rate:<10.0f}%')

print()
print('判定:')
avg_all = sum(r['reward'] for r in results) / len(results)
if avg_all > 0.7:
    print(f'  平均分 {avg_all:.2f} > 0.7 → 太简单,需要加难度')
elif avg_all < 0.3:
    print(f'  平均分 {avg_all:.2f} < 0.3 → 太难,需要加提示或降低难度')
else:
    print(f'  平均分 {avg_all:.2f} 在 0.3-0.7 → 难度合适 ✓')
"

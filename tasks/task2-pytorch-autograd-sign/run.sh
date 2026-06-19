#!/bin/bash
# run.sh — 用 Claude Code 跑单次任务 + 保存轨迹 + 拿 reward
# 用法: ./run.sh [model] [budget_usd]
# 示例: ./run.sh claude-sonnet-4-6 10

set -e

MODEL="${1:-claude-sonnet-4-6}"
BUDGET="${2:-10}"
TASK_DIR="$(dirname "$0")/task"
TRAJ_DIR="$(dirname "$0")/trajectories/$(date +%Y%m%d_%H%M%S)_${MODEL}"
mkdir -p "$TRAJ_DIR"

echo "=== 任务: $(basename $(dirname "$0")) ==="
echo "模型: $MODEL"
echo "预算: \$$BUDGET"
echo "轨迹: $TRAJ_DIR/trajectory.jsonl"

# 用 Claude Code 跑任务
claude \
  -p "$(cat "$TASK_DIR/instruction.md")" \
  --output-format stream-json \
  --model "$MODEL" \
  --max-budget-usd "$BUDGET" \
  --dangerously-skip-permissions \
  --effort high \
  > "$TRAJ_DIR/trajectory.jsonl" 2>"$TRAJ_DIR/stderr.log"

# 跑测试拿 reward
echo "=== 跑测试 ==="
mkdir -p /logs/verifier
bash "$TASK_DIR/tests/test.sh"
REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo "0.0")

# 提取统计
TURNS=$(grep -c '"type":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
TOOL_CALLS=$(grep -c '"type":"tool_use"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")

# 记录结果
echo "{\"task\":\"$(basename $(dirname "$0"))\",\"model\":\"$MODEL\",\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS,\"trajectory\":\"$TRAJ_DIR/trajectory.jsonl\"}" \
  | tee "$TRAJ_DIR/result.jsonl"

echo ""
echo "=== 结果 ==="
echo "Reward:    $REWARD"
echo "Turns:     $TURNS"
echo "Tool calls: $TOOL_CALLS"
echo "轨迹文件:  $TRAJ_DIR/trajectory.jsonl"

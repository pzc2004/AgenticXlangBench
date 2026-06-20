#!/bin/bash
# run.sh — 用 Kimi Code 跑一次任务 + 保存轨迹 + 记录分数
#
# 用法: ./run.sh [model] [budget_usd] [seed]
# 示例: ./run.sh kimi-for-coding 10 42
#
# 工作流程:
#   1. 启动 Docker 容器(GPU)
#   2. 容器内运行 Kimi Code,读取 instruction.md,修复 bug
#   3. Kimi 调用 tests/test.sh 记录分数
#   4. 保存轨迹到 trajectories/
#   5. 容器退出后,再启一次容器跑 test.sh 确认最终分数

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
SEED="${3:-42}"
TIMEOUT="${4:-3600}"  # 默认 1 小时超时

TASK_DIR="$(cd "$(dirname "$0")/task" && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL}_seed${SEED}"
TRAJ_DIR="$(cd "$(dirname "$0")/trajectories" && pwd)/$RUN_ID"
mkdir -p "$TRAJ_DIR"

TASK_NAME="$(basename "$(cd "$(dirname "$0")" && pwd)")"

echo "========================================="
echo " 任务:   $TASK_NAME"
echo " 模型:   $MODEL"
echo " 预算:   \$$BUDGET"
echo " Seed:   $SEED"
echo " 超时:   ${TIMEOUT}s"
echo " 轨迹:   $TRAJ_DIR/"
echo "========================================="

# 构造任务 prompt
# 注入 seed + 告诉模型完成后调用 test.sh
TASK_PROMPT="Seed: $SEED.

$(cat "$TASK_DIR/instruction.md")

## 完成后的验证步骤

修复 bug 后,运行以下命令验证:

\`\`\`bash
bash tests/test.sh
\`\`\`

这会输出分数(0-1)。确保在修复后运行这一步。"

# 启动容器,挂载任务文件和轨迹目录
# 用系统 timeout 命令限制总时间(Docker 本身不支持 --timeout)
echo ""
echo ">>> [1/3] 启动 Kimi Code (预算 \$$BUDGET, 种子 $SEED, 超时 ${TIMEOUT}s)..."
# 运行 Kimi Code,过滤掉 Docker 容器的 CUDA 初始化噪音(只保留 JSON 行)
timeout "$TIMEOUT" \
docker run --rm --gpus all \
  -v "$TASK_DIR:/workspace/task:ro" \
  -v "$TRAJ_DIR:/workspace/trajectories" \
  task1 \
  kimi -p "$TASK_PROMPT" \
    --model "$MODEL" \
    --output-format stream-json \
    2>"$TRAJ_DIR/stderr.log" \
    | grep '^{.*}$' > "$TRAJ_DIR/trajectory.jsonl" || true

echo ">>> [2/3] Kimi Code 结束,运行最终测试..."
# 再启容器跑测试(上一个容器已退出)
docker run --rm --gpus all \
  -v "$TASK_DIR:/workspace/task:ro" \
  task1 \
  bash /workspace/task/tests/test.sh 2>/dev/null || true

# 提取测试结果(从容器内复制出来)
docker run --rm --gpus all \
  -v "$TASK_DIR:/workspace/task:ro" \
  task1 \
  bash -c "
    bash /workspace/task/tests/test.sh 2>/dev/null || true
    cat /logs/verifier/reward.txt 2>/dev/null || echo '0.0'
  " > "$TRAJ_DIR/reward.txt" 2>/dev/null || true

REWARD=$(cat "$TRAJ_DIR/reward.txt" 2>/dev/null | tail -1 | tr -d '[:space:]')
REWARD="${REWARD:-0.0}"

# 提取轨迹统计
TURNS=$(grep -c '"type":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
TOOL_CALLS=$(grep -c '"type":"tool_use"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")

# 记录结果
echo "{\"task\":\"$TASK_NAME\",\"model\":\"$MODEL\",\"seed\":$SEED,\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS,\"trajectory\":\"$TRAJ_DIR/trajectory.jsonl\"}" \
  | tee "$TRAJ_DIR/result.jsonl"

echo ""
echo "========================================="
echo " Reward:    $REWARD"
echo " Turns:     $TURNS"
echo " Tool calls: $TOOL_CALLS"
echo " 轨迹:      $TRAJ_DIR/trajectory.jsonl"
echo "========================================="

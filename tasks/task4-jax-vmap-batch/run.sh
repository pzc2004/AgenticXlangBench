#!/bin/bash
# run.sh — 用 Kimi Code 跑单次任务 + 保存轨迹 + 拿 reward
# 用法: ./run.sh [model] [budget_usd] [seed] [timeout]

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
SEED="${3:-42}"
TIMEOUT="${4:-10800}"

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$SCRIPT_DIR/task"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_SAFE}_seed${SEED}"
TRAJ_DIR="$SCRIPT_DIR/trajectories/$RUN_ID"
mkdir -p "$TRAJ_DIR"
chmod 777 "$TRAJ_DIR"  # 容器内 agent(uid 1500) 需要写 /trajectories

TASK_NAME="$(basename "$SCRIPT_DIR")"

SECRETS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/.secrets"
ANTHROPIC_KEY=$(cat "$SECRETS_DIR/claude_api_key" 2>/dev/null | tr -d '[:space:]')
KIMI_KEY=$(cat "$SECRETS_DIR/kimi_api_key" 2>/dev/null | tr -d '[:space:]')

KIMI_CONFIG="$TRAJ_DIR/kimi_config.toml"
cat > "$KIMI_CONFIG" << KIMEOF
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

[[permission.rules]]
decision = "deny"
pattern = "WebSearch"

[[permission.rules]]
decision = "deny"
pattern = "WebFetch"
KIMEOF

echo "========================================="
echo " 任务:   $TASK_NAME"
echo " 模型:   $MODEL"
echo " 预算:   \$$BUDGET"
echo " Seed:   $SEED"
echo " 超时:   ${TIMEOUT}s"
echo " 轨迹:   $TRAJ_DIR/"
echo "========================================="

TASK_PROMPT="Seed: $SEED.

$(cat "$TASK_DIR/instruction.md")

## 完成后的验证步骤

修复 bug 后,运行以下命令查看分数:

\`\`\`bash
grade
\`\`\`

这会输出 0-1 的总分。**目标是 1.0 满分:只要分数没到 1.0,就说明还有 JAX batching rule bug 没修干净,请继续逐操作对比定位、修复,直到分数无法再提高,不要在中途分数停手。**"

CONTAINER_NAME="${TASK_NAME}_$(date +%s)"
SNAPSHOT_IMAGE="${TASK_NAME}_snapshot_$(date +%s)"

# 隐藏 /task/solution，防止 agent 直接看到 bugs.patch / solve.sh 等答案
EMPTY_SOLUTION_DIR="/tmp/${TASK_NAME}_empty_solution_$$"
mkdir -p "$EMPTY_SOLUTION_DIR"

echo ""
echo ">>> [1/3] 启动 Kimi Code (超时 ${TIMEOUT}s)..."
timeout "$TIMEOUT" \
docker run --name $CONTAINER_NAME --gpus all \
  --user 1500 -e HOME=/home/agent \
  --add-host="github.com:127.0.0.1" \
  --add-host="raw.githubusercontent.com:127.0.0.1" \
  --add-host="codeload.github.com:127.0.0.1" \
  --add-host="objects.githubusercontent.com:127.0.0.1" \
  --add-host="pypi.org:127.0.0.1" \
  --add-host="files.pythonhosted.org:127.0.0.1" \
  -v "$TASK_DIR/workspace:/workspace:ro" \
  -v "$TASK_DIR/instruction.md:/task/instruction.md:ro" \
  -v "$EMPTY_SOLUTION_DIR:/task/solution:ro" \
  -v "$TRAJ_DIR:/trajectories" \
  -v "$KIMI_CONFIG:/home/agent/.kimi-code/config.toml:ro" \
  -e "ANTHROPIC_API_KEY=$ANTHROPIC_KEY" \
  task4 \
  kimi -p "$TASK_PROMPT" \
    --model "$MODEL" \
    --output-format stream-json \
    2>"$TRAJ_DIR/stderr.log" \
    | grep '^{.*}$' > "$TRAJ_DIR/trajectory.jsonl" || true

echo ">>> [2/3] Kimi Code 结束,保存修复状态..."
docker commit "$CONTAINER_NAME" "$SNAPSHOT_IMAGE" > /dev/null 2>&1
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1
rm -rf "$EMPTY_SOLUTION_DIR"

echo ">>> 运行测试（受保护的最终判分，root 跑镜像内 /opt/judge/test.sh）..."
docker run --rm --gpus all --user 0 \
  -v "$TASK_DIR/workspace:/workspace:ro" \
  "$SNAPSHOT_IMAGE" \
  bash -c "
    bash /opt/judge/test.sh 2>/dev/null || true
    cat /logs/verifier/reward.txt 2>/dev/null || echo '0.0'
  " > "$TRAJ_DIR/reward.txt" 2>/dev/null || true

docker rmi "$SNAPSHOT_IMAGE" > /dev/null 2>&1 || true

REWARD=$(tail -1 "$TRAJ_DIR/reward.txt" 2>/dev/null | tr -d '[:space:]')
REWARD="${REWARD:-0.0}"

TURNS=$(grep -c '"role":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
TOOL_CALLS=$(grep -c '"tool_calls"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")

rm -f "$KIMI_CONFIG"

echo "{\"task\":\"$TASK_NAME\",\"model\":\"$MODEL\",\"seed\":$SEED,\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS,\"trajectory\":\"$TRAJ_DIR/trajectory.jsonl\"}" \
  | tee "$TRAJ_DIR/result.jsonl"

echo ""
echo "========================================="
echo " Reward:     $REWARD"
echo " Turns:      $TURNS"
echo " Tool calls: $TOOL_CALLS"
echo " 轨迹:       $TRAJ_DIR/trajectory.jsonl"
echo "========================================="

#!/bin/bash
# run.sh — 用 Kimi Code 跑一次任务 + 保存轨迹 + 记录分数
#
# 用法: ./run.sh [model] [budget_usd] [seed]
# 示例: ./run.sh kimi-code/kimi-for-coding 10 42
#
# API key 通过 .secrets/ 目录读取,启动容器时注入(不写入镜像)

set -e

MODEL="${1:-kimi-code/kimi-for-coding}"
BUDGET="${2:-10}"
SEED="${3:-42}"
TIMEOUT="${4:-3600}"

MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$SCRIPT_DIR/task"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${MODEL_SAFE}_seed${SEED}"
TRAJ_DIR="$SCRIPT_DIR/trajectories/$RUN_ID"
mkdir -p "$TRAJ_DIR"

TASK_NAME="$(basename "$SCRIPT_DIR")"

# 读取 API keys
SECRETS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/.secrets"
ANTHROPIC_KEY=$(cat "$SECRETS_DIR/claude_api_key" 2>/dev/null | tr -d '[:space:]')
KIMI_KEY=$(cat "$SECRETS_DIR/kimi_api_key" 2>/dev/null | tr -d '[:space:]')

# 生成 Kimi config.toml(临时,含真实 API key)
KIMI_CONFIG="$TRAJ_DIR/kimi_config.toml"
cat > "$KIMI_CONFIG" << EOF
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
EOF

echo "========================================="
echo " 任务:   $TASK_NAME"
echo " 模型:   $MODEL"
echo " 预算:   \$$BUDGET"
echo " Seed:   $SEED"
echo " 超时:   ${TIMEOUT}s"
echo " 轨迹:   $TRAJ_DIR/"
echo "========================================="

# 构造任务 prompt
TASK_PROMPT="Seed: $SEED.

$(cat "$TASK_DIR/instruction.md")

## 完成后的验证步骤

修复 bug 后,运行以下命令验证:

\`\`\`bash
bash /task/tests/test.sh
\`\`\`

这会输出分数(0-1)。确保在修复后运行这一步。"

# Docker 运行参数(通用)
# task/workspace → /workspace (train.py)
# task/          → /task (instruction.md, test.sh, solution/)
# trajectories   → /trajectories (轨迹输出,不进 /workspace 避免只读冲突)
DOCKER_ARGS="--rm --gpus all
  -v $TASK_DIR/workspace:/workspace:ro
  -v $TASK_DIR:/task:ro
  -v $TRAJ_DIR:/trajectories
  -v $KIMI_CONFIG:/root/.kimi-code/config.toml:ro
  -e ANTHROPIC_API_KEY=$ANTHROPIC_KEY"

# [1/3] 启动 Kimi Code
echo ""
echo ">>> [1/3] 启动 Kimi Code (超时 ${TIMEOUT}s)..."
timeout "$TIMEOUT" \
docker run $DOCKER_ARGS \
  task1 \
  kimi -p "$TASK_PROMPT" \
    --model "$MODEL" \
    --output-format stream-json \
    2>"$TRAJ_DIR/stderr.log" \
    | grep '^{.*}$' > "$TRAJ_DIR/trajectory.jsonl" || true

# [2/3] 最终测试
echo ">>> [2/3] Kimi Code 结束,运行最终测试..."
docker run --rm --gpus all \
  -v "$TASK_DIR:/workspace/task:ro" \
  task1 \
  bash -c "
    bash /workspace/task/tests/test.sh 2>/dev/null || true
    cat /logs/verifier/reward.txt 2>/dev/null || echo '0.0'
  " > "$TRAJ_DIR/reward.txt" 2>/dev/null || true

# [3/3] 汇总
REWARD=$(tail -1 "$TRAJ_DIR/reward.txt" 2>/dev/null | tr -d '[:space:]')
REWARD="${REWARD:-0.0}"

TURNS=$(grep -c '"type":"assistant"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")
TOOL_CALLS=$(grep -c '"type":"tool_use"' "$TRAJ_DIR/trajectory.jsonl" 2>/dev/null || echo "0")

# 清理临时 kimi config
rm -f "$KIMI_CONFIG"

# 记录结果
echo "{\"task\":\"$TASK_NAME\",\"model\":\"$MODEL\",\"seed\":$SEED,\"reward\":$REWARD,\"turns\":$TURNS,\"tool_calls\":$TOOL_CALLS,\"trajectory\":\"$TRAJ_DIR/trajectory.jsonl\"}" \
  | tee "$TRAJ_DIR/result.jsonl"

echo ""
echo "========================================="
echo " Reward:     $REWARD"
echo " Turns:      $TURNS"
echo " Tool calls: $TOOL_CALLS"
echo " 轨迹:       $TRAJ_DIR/trajectory.jsonl"
echo "========================================="

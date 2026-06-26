#!/bin/bash
# Oracle 测试：验证 bug 存在 + 修复后通过
set -e
WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

echo "========================================="
echo " Oracle 测试"
echo "========================================="

# 确保处于 buggy 状态
python3 /task/solution/inject_bug.py

# === 步骤 1：测试 buggy 版本（应该失败）===
echo ""
echo ">>> [1/2] 测试 buggy 版本（应该失败）..."
bash /task/tests/test.sh 2>&1 | tail -5
BUGGY_SCORE=$(cat "$REWARD_FILE" 2>/dev/null || echo "0.0")
echo "  Buggy 分数: $BUGGY_SCORE"

if python3 -c "exit(0 if float('$BUGGY_SCORE') >= 0.99 else 1)" 2>/dev/null; then
    echo "  ❌ Bug 没有被检测到！测试仍然通过。"
    echo "  ❌ Oracle 失败：bug 无效"
    echo "0" > "$REWARD_FILE"
    exit 1
else
    echo "  ✅ Bug 被检测到（分数 < 1.0）"
fi

# === 步骤 2：修复 bug ===
echo ""
echo ">>> [2/2] 修复 bug 并重新测试..."
bash /task/solution/solve.sh 2>&1 | tail -5

# 重新测试
bash /task/tests/test.sh 2>&1 | tail -5
FIXED_SCORE=$(cat "$REWARD_FILE" 2>/dev/null || echo "0.0")
echo "  Fixed 分数: $FIXED_SCORE"

if python3 -c "exit(0 if float('$FIXED_SCORE') >= 0.99 else 1)" 2>/dev/null; then
    echo "  ✅ 修复后测试通过"
    echo "1.0" > "$REWARD_FILE"
else
    echo "  ❌ 修复后测试失败"
    echo "$FIXED_SCORE" > "$REWARD_FILE"
    exit 1
fi

echo ""
echo "========================================="
echo " Oracle 测试通过"
echo "  Buggy: $BUGGY_SCORE < 1.0"
echo "  Fixed: $FIXED_SCORE = 1.0"
echo "========================================="

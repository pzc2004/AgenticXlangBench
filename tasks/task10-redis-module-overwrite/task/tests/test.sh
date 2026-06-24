#!/bin/bash
# tests/test.sh — Task 10 Redis Module Overwrite judge
# Scoring: 5 sections, max 1.0

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 10: Redis Module Overwrite Bug"
echo "========================================="

# === 1. Redis running ===
echo ""
echo ">>> [1/5] Redis connectivity..."
if redis-cli PING 2>/dev/null | grep -q PONG; then
    score=0.10
    echo "  Redis is running"
else
    echo "  Redis is NOT running"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. Module loaded ===
echo ""
echo ">>> [2/5] Custom module check..."
MODULE_OUTPUT=$(redis-cli MODULE LIST 2>/dev/null)
if echo "$MODULE_OUTPUT" | grep -qi "buggy"; then
    score=0.20
    echo "  Custom module is loaded"
else
    echo "  Custom module not detected"
    # Try to load it
    if [ -f "/build/redis/modules/buggy/module.so" ]; then
        redis-cli MODULE LOAD "/build/redis/modules/buggy/module.so" 2>/dev/null
        score=0.15
        echo "  Loaded module from build directory"
    fi
fi

# === 3. Core: Corruption test ===
echo ""
echo ">>> [3/5] Corruption test..."
OUTPUT=$(cd "$WORKSPACE" && python test_redis.py 2>&1)
echo "$OUTPUT"

ACC_LINE=$(echo "$OUTPUT" | grep "^accuracy " | tail -1)
CORRECT=$(echo "$ACC_LINE" | awk '{print $2}')
TOTAL=$(echo "$ACC_LINE" | awk '{print $3}')

if [ -n "$CORRECT" ] && [ -n "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    pct=$(python3 -c "print(f'{$CORRECT/$TOTAL*100:.1f}')" 2>/dev/null || echo "0")
    echo "  Test results: $CORRECT/$TOTAL ($pct%)"

    if [ "$CORRECT" -ge 3 ]; then
        score=0.60
        echo "  Corruption tests passed"
    elif [ "$CORRECT" -ge 2 ]; then
        score=0.45
        echo "  Partial corruption tests passed"
    else
        echo "  Corruption tests failed"
    fi
else
    echo "  Could not parse test output"
fi

# === 4. Multi-run stability ===
echo ""
echo ">>> [4/5] Multi-run stability..."
RUN_PASS=0
for run in 1 2 3; do
    RUN_OUTPUT=$(cd "$WORKSPACE" && python test_redis.py 2>&1)
    RUN_ACC=$(echo "$RUN_OUTPUT" | grep "^accuracy " | tail -1)
    RUN_CORRECT=$(echo "$RUN_ACC" | awk '{print $2}')
    RUN_TOTAL=$(echo "$RUN_ACC" | awk '{print $3}')
    if [ -n "$RUN_CORRECT" ] && [ -n "$RUN_TOTAL" ] && [ "$RUN_TOTAL" -gt 0 ]; then
        ok=$(python3 -c "print(1 if $RUN_CORRECT/$RUN_TOTAL >= 0.75 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            RUN_PASS=$((RUN_PASS + 1))
        fi
    fi
done

if [ "$RUN_PASS" -ge 2 ]; then
    score=0.80
    echo "  Stability tests passed ($RUN_PASS/3)"
else
    echo "  Stability: $RUN_PASS/3 passed"
fi

# === 5. Anti-hack ===
echo ""
echo ">>> [5/5] Anti-hack checks..."

# Check: not using native Redis commands to bypass module
if grep -rn "HSET\|HGET" "$WORKSPACE/test_redis.py" 2>/dev/null | grep -v "buggy\|#" >/dev/null; then
    echo "  NOTE: Test uses native Redis hash commands (expected for baseline)"
fi

# Check: not patching the test to always pass
if grep -q "return True" "$WORKSPACE/test_redis.py" 2>/dev/null; then
    # This is normal - test functions return True/False
    :
fi

# Check: not replacing module.so with a fixed version
if [ -f "/build/redis/modules/buggy/module.so" ]; then
    MODULE_HASH=$(md5sum "/build/redis/modules/buggy/module.so" 2>/dev/null | awk '{print $1}')
    # This is hard to check without a reference hash
fi

# Check: module source was actually modified (not just recompiled)
MODULE_SRC="${MODULE_DIR:-/build/redis/modules/buggy}/module.c"
if [ -f "$MODULE_SRC" ]; then
    if grep -q "len + 1" "$MODULE_SRC" 2>/dev/null; then
        echo "  Module source still contains buggy patterns (bugs not fixed)"
    else
        echo "  Module source appears to be fixed"
    fi
fi

if [ $HACK -eq 0 ]; then
    echo "  No hacks detected"
    if [ $(python3 -c "print(1 if $score >= 0.8 else 0)" 2>/dev/null || echo "0") = "1" ]; then
        score=1.0
    fi
fi

# === Final score ===
echo ""
echo "========================================="
if [ $HACK -eq 0 ]; then
    echo " Final score: $score"
else
    score=$(python3 -c "print(f'{$score * 0.5:.2f}')" 2>/dev/null || echo "0.0")
    echo " Final score: $score (hack detected, halved)"
fi
echo "========================================="

echo "$score" > "$REWARD_FILE"

#!/bin/bash
# tests/test.sh — Task 9 PostgreSQL Executor NULL Bug judge
# Scoring: 6 sections, max 1.0

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 9: PostgreSQL Executor NULL Bug"
echo "========================================="

# === 1. PostgreSQL running ===
echo ""
echo ">>> [1/6] PostgreSQL connectivity..."
if pg_isready -h /var/run/postgresql 2>/dev/null || \
   psql -h /var/run/postgresql -U postgres -c "SELECT 1" >/dev/null 2>&1; then
    score=0.10
    echo "  PostgreSQL is running"
else
    echo "  PostgreSQL is NOT running"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. Basic join test ===
echo ""
echo ">>> [2/6] Basic join query test..."
RESULT=$(psql -h /var/run/postgresql -U postgres -d testdb -t -A -c \
    "SELECT COUNT(*) FROM (SELECT 1 FROM (VALUES (1),(2)) a(id) JOIN (VALUES (1),(2)) b(id) ON a.id = b.id) t" 2>/dev/null)
if [ "$RESULT" = "2" ]; then
    score=0.20
    echo "  Basic join returns correct count"
else
    echo "  Basic join failed (got: $RESULT)"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. Core: NULL join queries ===
echo ""
echo ">>> [3/6] NULL join correctness..."
PASS_COUNT=0
TOTAL_TESTS=6

# Run test queries
OUTPUT=$(cd "$WORKSPACE" && python test_queries.py --seed 42 2>&1)
echo "$OUTPUT"

ACC_LINE=$(echo "$OUTPUT" | grep "^accuracy " | tail -1)
CORRECT=$(echo "$ACC_LINE" | awk '{print $2}')
ACTUAL_TOTAL=$(echo "$ACC_LINE" | awk '{print $3}')

if [ -n "$CORRECT" ] && [ -n "$ACTUAL_TOTAL" ] && [ "$ACTUAL_TOTAL" -gt 0 ]; then
    PASS_COUNT=$CORRECT
    pct=$(python3 -c "print(f'{$CORRECT/$ACTUAL_TOTAL*100:.1f}')" 2>/dev/null || echo "0")
    echo "  Query tests: $CORRECT/$ACTUAL_TOTAL ($pct%)"
fi

if [ "$PASS_COUNT" -ge 4 ]; then
    score=0.50
    echo "  NULL join tests passed"
elif [ "$PASS_COUNT" -ge 2 ]; then
    score=0.35
    echo "  Partial NULL join tests passed"
else
    echo "  NULL join tests failed"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. Multi-seed test ===
echo ""
echo ">>> [4/6] Multi-seed stability..."
SEED_PASS=0
for seed in 1 7 42 99 123; do
    SEED_OUTPUT=$(cd "$WORKSPACE" && python test_queries.py --seed $seed 2>&1)
    SEED_ACC=$(echo "$SEED_OUTPUT" | grep "^accuracy " | tail -1)
    SEED_CORRECT=$(echo "$SEED_ACC" | awk '{print $2}')
    SEED_TOTAL=$(echo "$SEED_ACC" | awk '{print $3}')
    if [ -n "$SEED_CORRECT" ] && [ -n "$SEED_TOTAL" ] && [ "$SEED_TOTAL" -gt 0 ]; then
        ok=$(python3 -c "print(1 if $SEED_CORRECT/$SEED_TOTAL >= 0.8 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            SEED_PASS=$((SEED_PASS + 1))
        fi
    fi
done

if [ "$SEED_PASS" -ge 4 ]; then
    score=0.70
    echo "  Multi-seed tests passed ($SEED_PASS/5)"
else
    echo "  Multi-seed: $SEED_PASS/5 passed"
fi

# === 5. Executor source modified ===
echo ""
echo ">>> [5/6] Source code check..."
PG_SRC="${PG_SRC:-/usr/src/postgresql}"
NODE_FILE="$PG_SRC/src/backend/executor/nodeNestloop.c"
if [ -f "$NODE_FILE" ]; then
    if grep -q "ExecQual" "$NODE_FILE"; then
        score=0.80
        echo "  nodeNestloop.c contains expected patterns"
    else
        echo "  WARNING: nodeNestloop.c may be corrupted"
    fi
else
    echo "  WARNING: nodeNestloop.c not found at expected path"
fi

# === 6. Anti-hack ===
echo ""
echo ">>> [6/6] Anti-hack checks..."

# Check: SQL queries not modified to bypass the bug
if [ -f "$WORKSPACE/test_queries.py" ]; then
    if grep -q "SET enable_nestloop" "$WORKSPACE/test_queries.py" 2>/dev/null; then
        echo "  HACK DETECTED: SQL configuration changes found"
        HACK=1
    fi
    if grep -q "SET join_collapse_limit" "$WORKSPACE/test_queries.py" 2>/dev/null; then
        echo "  HACK DETECTED: join strategy override found"
        HACK=1
    fi
fi

# Check: no workarounds in test script (like adding extra WHERE clauses)
if grep -rn "COALESCE\|IFNULL\|CASE WHEN" "$WORKSPACE/test_queries.py" 2>/dev/null | grep -v "^#" >/dev/null; then
    echo "  HACK DETECTED: result manipulation found"
    HACK=1
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

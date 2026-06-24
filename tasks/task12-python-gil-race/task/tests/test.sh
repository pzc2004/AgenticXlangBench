#!/bin/bash
# tests/test.sh — Task 12 Python GIL Race Condition judge
# Scoring: 6 sections, max 1.0

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 12: Python C Extension GIL Race"
echo "========================================="

# === 1. Module importable ===
echo ""
echo ">>> [1/6] Module import check..."
if python3 -c "import compute; print('compute module loaded')" 2>/dev/null; then
    score=0.10
    echo "  compute module is importable"
else
    echo "  compute module NOT importable"
    # Try to build it
    if [ -f "$WORKSPACE/setup.py" ]; then
        echo "  Attempting to build..."
        cd "$WORKSPACE" && pip install -e . 2>&1 | tail -3
        if python3 -c "import compute" 2>/dev/null; then
            score=0.10
            echo "  Module built successfully"
        else
            echo "$score" > "$REWARD_FILE"
            exit 0
        fi
    else
        echo "$score" > "$REWARD_FILE"
        exit 0
    fi
fi

# === 2. Single-threaded correctness ===
echo ""
echo ">>> [2/6] Single-threaded test..."
ST_OUTPUT=$(cd "$WORKSPACE" && python3 test_gil.py --threads 1 --iterations 50 2>&1)
ST_ACC=$(echo "$ST_OUTPUT" | grep "^accuracy " | tail -1)
ST_CORRECT=$(echo "$ST_ACC" | awk '{print $2}')
ST_TOTAL=$(echo "$ST_ACC" | awk '{print $3}')

if [ -n "$ST_CORRECT" ] && [ -n "$ST_TOTAL" ] && [ "$ST_TOTAL" -gt 0 ]; then
    st_pct=$(python3 -c "print(f'{$ST_CORRECT/$ST_TOTAL*100:.1f}')" 2>/dev/null || echo "0")
    echo "  Single-threaded: $ST_CORRECT/$ST_TOTAL ($st_pct%)"
    if [ "$ST_CORRECT" = "$ST_TOTAL" ]; then
        score=0.25
        echo "  Single-threaded passes perfectly"
    fi
else
    echo "  Could not parse single-threaded output"
fi

# === 3. Core: Multi-threaded test ===
echo ""
echo ">>> [3/6] Multi-threaded test..."
MT_OUTPUT=$(cd "$WORKSPACE" && python3 test_gil.py --threads 4 --iterations 100 2>&1)
echo "$MT_OUTPUT"
MT_ACC=$(echo "$MT_OUTPUT" | grep "^accuracy " | tail -1)
MT_CORRECT=$(echo "$MT_ACC" | awk '{print $2}')
MT_TOTAL=$(echo "$MT_ACC" | awk '{print $3}')

if [ -n "$MT_CORRECT" ] && [ -n "$MT_TOTAL" ] && [ "$MT_TOTAL" -gt 0 ]; then
    mt_pct=$(python3 -c "print(f'{$MT_CORRECT/$MT_TOTAL*100:.1f}')" 2>/dev/null || echo "0")
    echo "  Multi-threaded: $MT_CORRECT/$MT_TOTAL ($mt_pct%)"

    mt_ratio=$(python3 -c "print($MT_CORRECT/$MT_TOTAL)" 2>/dev/null || echo "0")
    if python3 -c "exit(0 if $mt_ratio >= 0.99 else 1)" 2>/dev/null; then
        score=0.60
        echo "  Multi-threaded test passes (race condition fixed)"
    elif python3 -c "exit(0 if $mt_ratio >= 0.90 else 1)" 2>/dev/null; then
        score=0.45
        echo "  Multi-threaded mostly passes"
    else
        echo "  Multi-threaded has significant failures"
    fi
else
    echo "  Could not parse multi-threaded output"
fi

# === 4. Counter integrity ===
echo ""
echo ">>> [4/6] Counter integrity test..."
CT_OUTPUT=$(cd "$WORKSPACE" && python3 test_gil.py --threads 8 --iterations 50 2>&1)
CT_ACC=$(echo "$CT_OUTPUT" | grep "^accuracy " | tail -1)
CT_CORRECT=$(echo "$CT_ACC" | awk '{print $2}')
CT_TOTAL=$(echo "$CT_ACC" | awk '{print $3}')

if [ -n "$CT_CORRECT" ] && [ -n "$CT_TOTAL" ] && [ "$CT_TOTAL" -gt 0 ]; then
    ct_ratio=$(python3 -c "print($CT_CORRECT/$CT_TOTAL)" 2>/dev/null || echo "0")
    if python3 -c "exit(0 if $ct_ratio >= 0.99 else 1)" 2>/dev/null; then
        score=0.75
        echo "  Counter integrity passes"
    else
        echo "  Counter integrity issues ($CT_CORRECT/$CT_TOTAL)"
    fi
fi

# === 5. High-thread stress test ===
echo ""
echo ">>> [5/6] Stress test (16 threads)..."
STRESS_OUTPUT=$(cd "$WORKSPACE" && python3 test_gil.py --threads 16 --iterations 200 2>&1)
STRESS_ACC=$(echo "$STRESS_OUTPUT" | grep "^accuracy " | tail -1)
STRESS_CORRECT=$(echo "$STRESS_ACC" | awk '{print $2}')
STRESS_TOTAL=$(echo "$STRESS_ACC" | awk '{print $3}')

if [ -n "$STRESS_CORRECT" ] && [ -n "$STRESS_TOTAL" ] && [ "$STRESS_TOTAL" -gt 0 ]; then
    stress_ratio=$(python3 -c "print($STRESS_CORRECT/$STRESS_TOTAL)" 2>/dev/null || echo "0")
    if python3 -c "exit(0 if $stress_ratio >= 0.99 else 1)" 2>/dev/null; then
        score=0.85
        echo "  Stress test passes"
    else
        echo "  Stress test: $STRESS_CORRECT/$STRESS_TOTAL"
    fi
fi

# === 6. Anti-hack ===
echo ""
echo ">>> [6/6] Anti-hack checks..."

# Check: not using threading.Lock in Python to mask the C-level bug
if grep -rn "Lock\|lock\|mutex\|RLock" "$WORKSPACE/test_gil.py" 2>/dev/null | grep -v "^#" | grep -v "import"; then
    echo "  HACK DETECTED: Python-level locking to mask C bug"
    HACK=1
fi

# Check: not using multiprocessing instead of threading
if grep -q "multiprocessing\|Pool\|Process" "$WORKSPACE/test_gil.py" 2>/dev/null; then
    echo "  HACK DETECTED: Using multiprocessing instead of threading"
    HACK=1
fi

# Check: not setting threads=1 to bypass
if grep -q "threads.*=.*1\b" "$WORKSPACE/test_gil.py" 2>/dev/null; then
    # This is the single-thread test, which is expected
    :
fi

# Check: compute.c actually has the fix (not just test workaround)
if [ -f "$WORKSPACE/compute.c" ]; then
    if grep -A3 "Py_BEGIN_ALLOW_THREADS" "$WORKSPACE/compute.c" | grep -q "global_counter\|global_call_count"; then
        echo "  NOTE: compute.c still has shared state in NO-GIL zone"
    fi
fi

if [ $HACK -eq 0 ]; then
    echo "  No hacks detected"
    if [ $(python3 -c "print(1 if $score >= 0.85 else 0)" 2>/dev/null || echo "0") = "1" ]; then
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

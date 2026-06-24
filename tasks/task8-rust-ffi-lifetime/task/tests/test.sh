#!/bin/bash
# tests/test.sh — Task 8 Evaluation Script
#
# Scoring:
#   0.10  Extension importable
#   0.30  Short stress test (1000 iterations x 5 runs, no crash)
#   0.30  Long stress test (10000 iterations x 3 runs, no crash)
#   0.15  Data correctness (no corrupted results)
#   0.05  Performance (Rust faster than Python)
#   0.10  Anti-hack: no signal handlers, no Box::leak, no Python reimpl

set -e

WORKSPACE_DIR="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
REWARD_DIR=$(dirname "$REWARD_FILE")

mkdir -p "$REWARD_DIR"
echo "0.0" > "$REWARD_FILE"

TOTAL=0.0
REPORT=""

# Helper: add to score
add_score() {
    local points="$1"
    local desc="$2"
    TOTAL=$(echo "$TOTAL + $points" | bc)
    REPORT="${REPORT}\n  +${points}  ${desc}"
    echo "[PASS] ${desc} (+${points})"
}

# Helper: fail a check
fail_check() {
    local desc="$1"
    REPORT="${REPORT}\n  +0.00  ${desc} (FAILED)"
    echo "[FAIL] ${desc}"
}

echo "========================================"
echo "Task 8: Rust FFI Lifetime — Evaluation"
echo "========================================"

# -------------------------------------------------------
# Check 1: Extension importable (0.10)
# -------------------------------------------------------
echo ""
echo "--- Check 1: Extension importable ---"
if python3 -c "import rustops; print('rustops version:', rustops.__version__)" 2>/dev/null; then
    add_score "0.10" "Extension importable"
else
    fail_check "Extension importable"
fi

# -------------------------------------------------------
# Check 2: Short stress test (0.30)
# 1000 iterations x 5 runs, no crash
# -------------------------------------------------------
echo ""
echo "--- Check 2: Short stress test (1000 x 5) ---"
SHORT_PASS=0
for run in 1 2 3 4 5; do
    echo -n "  Run $run/5: "
    if timeout 60 python3 "$WORKSPACE_DIR/test_stress.py" --iterations 1000 --seed $((42 + run)) >/dev/null 2>&1; then
        echo "OK"
        SHORT_PASS=$((SHORT_PASS + 1))
    else
        echo "FAIL (exit=$?)"
    fi
done

if [ "$SHORT_PASS" -eq 5 ]; then
    add_score "0.30" "Short stress test: 5/5 runs passed"
elif [ "$SHORT_PASS" -ge 3 ]; then
    SCORE=$(echo "0.30 * $SHORT_PASS / 5" | bc -l | xargs printf "%.2f")
    add_score "$SCORE" "Short stress test: $SHORT_PASS/5 runs passed (partial)"
else
    fail_check "Short stress test: only $SHORT_PASS/5 passed"
fi

# -------------------------------------------------------
# Check 3: Long stress test (0.30)
# 10000 iterations x 3 runs, no crash
# -------------------------------------------------------
echo ""
echo "--- Check 3: Long stress test (10000 x 3) ---"
LONG_PASS=0
for run in 1 2 3; do
    echo -n "  Run $run/3: "
    if timeout 120 python3 "$WORKSPACE_DIR/test_stress.py" --iterations 10000 --seed $((100 + run)) >/dev/null 2>&1; then
        echo "OK"
        LONG_PASS=$((LONG_PASS + 1))
    else
        echo "FAIL (exit=$?)"
    fi
done

if [ "$LONG_PASS" -eq 3 ]; then
    add_score "0.30" "Long stress test: 3/3 runs passed"
elif [ "$LONG_PASS" -ge 2 ]; then
    SCORE=$(echo "0.30 * $LONG_PASS / 3" | bc -l | xargs printf "%.2f")
    add_score "$SCORE" "Long stress test: $LONG_PASS/3 runs passed (partial)"
else
    fail_check "Long stress test: only $LONG_PASS/3 passed"
fi

# -------------------------------------------------------
# Check 4: Data correctness (0.15)
# Run with verification enabled, check for corrupted results
# -------------------------------------------------------
echo ""
echo "--- Check 4: Data correctness ---"
CORRECTNESS_PASS=0
for run in 1 2 3; do
    echo -n "  Run $run/3: "
    OUTPUT=$(timeout 60 python3 "$WORKSPACE_DIR/test_stress.py" --iterations 2000 --seed $((200 + run)) 2>&1)
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ] && echo "$OUTPUT" | grep -q "ALL PASS"; then
        echo "OK"
        CORRECTNESS_PASS=$((CORRECTNESS_PASS + 1))
    else
        echo "FAIL"
        echo "    Output: $(echo "$OUTPUT" | tail -5)"
    fi
done

if [ "$CORRECTNESS_PASS" -eq 3 ]; then
    add_score "0.15" "Data correctness: 3/3 runs passed"
elif [ "$CORRECTNESS_PASS" -ge 2 ]; then
    SCORE=$(echo "0.15 * $CORRECTNESS_PASS / 3" | bc -l | xargs printf "%.2f")
    add_score "$SCORE" "Data correctness: $CORRECTNESS_PASS/3 runs passed (partial)"
else
    fail_check "Data correctness: only $CORRECTNESS_PASS/3 passed"
fi

# -------------------------------------------------------
# Check 5: Performance (0.05)
# Rust should be faster than pure Python equivalent
# -------------------------------------------------------
echo ""
echo "--- Check 5: Performance ---"
# Time Rust implementation
RUST_TIME=$(timeout 60 python3 "$WORKSPACE_DIR/test_stress.py" --iterations 5000 --seed 42 --profile 2>/dev/null | grep "total_time" | awk '{print $2}')
# Time pure Python equivalent (using test_stress.py's Python fallback if available)
# We'll do a simple comparison: just check that Rust completes in reasonable time
if [ -n "$RUST_TIME" ]; then
    # Rust should complete 5000 iterations in under 10 seconds
    if python3 -c "exit(0 if float('$RUST_TIME') < 10.0 else 1)" 2>/dev/null; then
        add_score "0.05" "Performance: Rust completes 5000 iters in ${RUST_TIME}s (< 10s)"
    else
        fail_check "Performance: Rust too slow (${RUST_TIME}s for 5000 iters)"
    fi
else
    fail_check "Performance: could not measure Rust timing"
fi

# -------------------------------------------------------
# Check 6: Anti-hack checks (0.10)
# -------------------------------------------------------
echo ""
echo "--- Check 6: Anti-hack checks ---"
ANTI_HACK=0.0

# 6a: No signal handlers in Rust code (no std::panic::set_hook, no signal_hook, no libc::signal)
if grep -rn "signal\|sigaction\|SIGSEGV\|SIGBUS\|set_hook\|catch_unwind" "$WORKSPACE_DIR/src/" 2>/dev/null | grep -v "//.*signal" | grep -v "BUG_LOCATION" | grep -q "."; then
    fail_check "Anti-hack: signal handler detected in Rust code"
else
    ANTI_HACK=$(echo "$ANTI_HACK + 0.033" | bc)
    echo "[PASS] Anti-hack: no signal handlers (+0.033)"
fi

# 6b: No Box::leak in Rust code (would mask use-after-free)
if grep -rn "Box::leak\|ManuallyDrop\|mem::forget\|std::mem::forget" "$WORKSPACE_DIR/src/" 2>/dev/null | grep -v "BUG_LOCATION" | grep -q "."; then
    fail_check "Anti-hack: Box::leak/ManuallyDrop/mem::forget detected"
else
    ANTI_HACK=$(echo "$ANTI_HACK + 0.033" | bc)
    echo "[PASS] Anti-hack: no Box::leak/mem::forget (+0.033)"
fi

# 6c: No Python reimplementation (check that .rs files still have actual Rust code)
RS_LINES=$(wc -l < "$WORKSPACE_DIR/src/ffi.rs" 2>/dev/null || echo "0")
if [ "$RS_LINES" -gt 50 ]; then
    ANTI_HACK=$(echo "$ANTI_HACK + 0.034" | bc)
    echo "[PASS] Anti-hack: Rust code intact (${RS_LINES} lines) (+0.034)"
else
    fail_check "Anti-hack: Rust code too short (${RS_LINES} lines), possible reimpl"
fi

TOTAL=$(echo "$TOTAL + $ANTI_HACK" | bc)

# -------------------------------------------------------
# Final Score
# -------------------------------------------------------
echo ""
echo "========================================"
echo -e "Score Breakdown:${REPORT}"
echo "========================================"
echo "TOTAL SCORE: $TOTAL / 1.00"
echo "$TOTAL" > "$REWARD_FILE"

if python3 -c "exit(0 if float('$TOTAL') >= 0.6 else 1)" 2>/dev/null; then
    echo "RESULT: PASS (score >= 0.6)"
else
    echo "RESULT: FAIL (score < 0.6)"
fi

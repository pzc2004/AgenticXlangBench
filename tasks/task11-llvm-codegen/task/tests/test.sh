#!/bin/bash
# tests/test.sh — Task 11 LLVM Codegen Bug judge
# Scoring: 6 sections, max 1.0

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 11: LLVM Instruction Selection Bug"
echo "========================================="

# === 1. Compiler available ===
echo ""
echo ">>> [1/6] Compiler check..."
if clang --version 2>/dev/null | head -1; then
    COMPILER="clang"
elif gcc --version 2>/dev/null | head -1; then
    COMPILER="gcc"
else
    echo "  No C compiler found"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi
score=0.10
echo "  Using: $COMPILER"

# === 2. -O0 correctness ===
echo ""
echo ">>> [2/6] -O0 baseline test..."
$COMPILER -O0 -o /tmp/test_O0 "$WORKSPACE/test_codegen.c" 2>/dev/null
if [ -f /tmp/test_O0 ]; then
    O0_OUTPUT=$(/tmp/test_O0 2>&1)
    O0_ACC=$(echo "$O0_OUTPUT" | grep "^accuracy " | awk '{print $2}')
    O0_TOTAL=$(echo "$O0_OUTPUT" | grep "^accuracy " | awk '{print $3}')
    if [ -n "$O0_ACC" ] && [ -n "$O0_TOTAL" ] && [ "$O0_TOTAL" -gt 0 ]; then
        if [ "$O0_ACC" = "$O0_TOTAL" ]; then
            score=0.20
            echo "  -O0 passes all tests ($O0_ACC/$O0_TOTAL)"
        else
            echo "  -O0 has failures ($O0_ACC/$O0_TOTAL)"
        fi
    fi
else
    echo "  Failed to compile with -O0"
fi

# === 3. Core: -O2 test ===
echo ""
echo ">>> [3/6] -O2 codegen test..."
$COMPILER -O2 -o /tmp/test_O2 "$WORKSPACE/test_codegen.c" 2>/dev/null
if [ -f /tmp/test_O2 ]; then
    O2_OUTPUT=$(/tmp/test_O2 2>&1)
    echo "$O2_OUTPUT"
    O2_ACC=$(echo "$O2_OUTPUT" | grep "^accuracy " | awk '{print $2}')
    O2_TOTAL=$(echo "$O2_OUTPUT" | grep "^accuracy " | awk '{print $3}')
    if [ -n "$O2_ACC" ] && [ -n "$O2_TOTAL" ] && [ "$O2_TOTAL" -gt 0 ]; then
        pct=$(python3 -c "print(f'{$O2_ACC/$O2_TOTAL*100:.1f}')" 2>/dev/null || echo "0")
        echo "  -O2 accuracy: $O2_ACC/$O2_TOTAL ($pct%)"

        if [ "$O2_ACC" = "$O2_TOTAL" ]; then
            score=0.60
            echo "  -O2 passes all tests (bug may be fixed)"
        elif [ "$O2_ACC" -ge "$((O2_TOTAL * 4 / 5))" ]; then
            score=0.40
            echo "  -O2 mostly passes"
        else
            echo "  -O2 has significant failures"
        fi
    fi
else
    echo "  Failed to compile with -O2"
fi

# === 4. O0 vs O2 comparison ===
echo ""
echo ">>> [4/6] -O0 vs -O2 comparison..."
if [ -n "$O0_ACC" ] && [ -n "$O2_ACC" ] && [ "$O0_ACC" = "$O0_TOTAL" ] && [ "$O2_ACC" = "$O2_TOTAL" ]; then
    echo "  Both -O0 and -O2 pass: codegen bug appears to be fixed"
    score=0.80
elif [ "$O0_ACC" = "$O0_TOTAL" ] && [ "$O2_ACC" != "$O2_TOTAL" ]; then
    echo "  -O0 passes but -O2 fails: codegen bug confirmed"
    # This is the expected buggy state - score based on how much was fixed
fi

# === 5. Multi-compilation test ===
echo ""
echo ">>> [5/6] Multi-optimization-level test..."
OPT_PASS=0
for opt in O1 O2 O3; do
    $COMPILER -$opt -o "/tmp/test_$opt" "$WORKSPACE/test_codegen.c" 2>/dev/null
    if [ -f "/tmp/test_$opt" ]; then
        OPT_OUTPUT=$("/tmp/test_$opt" 2>&1)
        OPT_ACC=$(echo "$OPT_OUTPUT" | grep "^accuracy " | awk '{print $2}')
        OPT_TOTAL=$(echo "$OPT_OUTPUT" | grep "^accuracy " | awk '{print $3}')
        if [ -n "$OPT_ACC" ] && [ "$OPT_ACC" = "$OPT_TOTAL" ]; then
            OPT_PASS=$((OPT_PASS + 1))
        fi
    fi
done

if [ "$OPT_PASS" -ge 2 ]; then
    echo "  Passes at $OPT_PASS/3 optimization levels"
    if [ "$score" -lt 0.80 ]; then
        score=0.80
    fi
else
    echo "  Passes at $OPT_PASS/3 optimization levels"
fi

# === 6. Anti-hack ===
echo ""
echo ">>> [6/6] Anti-hack checks..."

# Check: not using -O0 as workaround
if grep -q "\-O0" "$WORKSPACE/test_asm.sh" 2>/dev/null; then
    # This is normal for comparison, check if -O0 is the ONLY level used
    :
fi

# Check: not modifying test source to avoid triggering bug
if [ -f "$WORKSPACE/test_codegen.c" ]; then
    if grep -q "volatile" "$WORKSPACE/test_codegen.c" 2>/dev/null; then
        echo "  HACK DETECTED: volatile keyword used to prevent optimization"
        HACK=1
    fi
    if grep -q "asm\|__asm__" "$WORKSPACE/test_codegen.c" 2>/dev/null; then
        echo "  HACK DETECTED: inline assembly workaround"
        HACK=1
    fi
fi

# Check: not using a different compiler
if grep -q "gcc\|g++" "$WORKSPACE/test_asm.sh" 2>/dev/null; then
    if ! grep -q "clang" "$WORKSPACE/test_asm.sh" 2>/dev/null; then
        echo "  NOTE: Using gcc instead of clang (acceptable but less precise)"
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

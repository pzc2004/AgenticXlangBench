#!/bin/bash
# test.sh -- Evaluation script for Task 6 (TensorFlow Custom Op Shape Inference)
#
# Scoring breakdown:
#   0.10  Extension builds and imports
#   0.30  Short chain passes  (10 ops  x 5 seed/config combos)
#   0.30  Long chain passes   (60 ops  x 3 seed/config combos)
#   0.15  Shape correctness   (metadata == actual at every op)
#   0.05  Performance         (60-op chain completes within 60 seconds)
#   0.10  Anti-hack checks
#
# Total >= 0.60 to pass.

set -e

LOGDIR="/logs/verifier"
mkdir -p "$LOGDIR"
REWARD_FILE="$LOGDIR/reward.txt"

# Auto-detect workspace: prefer /build/ (modified), then /workspace (read-only)
if [ -f "/build/tensor_ops.c" ]; then
    WS="/build"
elif [ -f "/workspace/tensor_ops.c" ]; then
    WS="/workspace"
else
    WS="$(cd "$(dirname "$0")/../workspace" && pwd)"
fi
TEST_SCRIPT="$WS/test_chain.py"
SRC_FILE="$WS/tensor_ops.c"
SETUP_FILE="$WS/setup.py"
MODEL_FILE="$WS/model.py"

TOTAL=0.0

# Helper: add to total with 2-decimal precision
add_score() {
    TOTAL=$(python3 -c "print(round($TOTAL + $1, 2))")
}

fail_section() {
    echo "  [FAIL] $1"
}

# ====================================================================
# Section 1: Extension builds and imports  (0.10)
# ====================================================================
echo "=== Section 1: Build and Import ==="
BUILD_OK=false

if [ ! -f "$SRC_FILE" ]; then
    fail_section "tensor_ops.c not found"
else
    # Build
    cd "$WS"
    BUILD_LOG="$LOGDIR/build.log"
    if pip install -e . > "$BUILD_LOG" 2>&1; then
        # Import test
        if python3 -c "import tensor_ops; print('import OK')" 2>/dev/null; then
            echo "  [PASS] Extension builds and imports"
            add_score 0.10
            BUILD_OK=true
        else
            fail_section "Extension imports failed"
        fi
    else
        fail_section "Build failed (see $BUILD_LOG)"
    fi
fi

if [ "$BUILD_OK" = false ]; then
    echo ""
    echo "=== Cannot proceed without build. Final score: $TOTAL ==="
    echo "$TOTAL" > "$REWARD_FILE"
    exit 0
fi

# ====================================================================
# Section 2: Short chain passes  (0.30)
# ====================================================================
echo ""
echo "=== Section 2: Short Chain (10 ops) ==="
SHORT_PASS=0
SHORT_TOTAL=5

for SEED in 42 123 456 789 1024; do
    if python3 "$TEST_SCRIPT" --num_ops 10 --seed "$SEED" > /dev/null 2>&1; then
        echo "  [PASS] seed=$SEED"
        SHORT_PASS=$((SHORT_PASS + 1))
    else
        echo "  [FAIL] seed=$SEED"
    fi
done

SHORT_SCORE=$(python3 -c "print(round(0.30 * $SHORT_PASS / $SHORT_TOTAL, 2))")
echo "  Short chain: $SHORT_PASS / $SHORT_TOTAL -> +$SHORT_SCORE"
add_score "$SHORT_SCORE"

# ====================================================================
# Section 3: Long chain passes  (0.30)
# ====================================================================
echo ""
echo "=== Section 3: Long Chain (60 ops) ==="
LONG_PASS=0
LONG_TOTAL=3

for SEED in 42 123 456; do
    if python3 "$TEST_SCRIPT" --num_ops 60 --seed "$SEED" > /dev/null 2>&1; then
        echo "  [PASS] seed=$SEED"
        LONG_PASS=$((LONG_PASS + 1))
    else
        echo "  [FAIL] seed=$SEED"
    fi
done

LONG_SCORE=$(python3 -c "print(round(0.30 * $LONG_PASS / $LONG_TOTAL, 2))")
echo "  Long chain: $LONG_PASS / $LONG_TOTAL -> +$LONG_SCORE"
add_score "$LONG_SCORE"

# ====================================================================
# Section 4: Shape correctness  (0.15)
# ====================================================================
echo ""
echo "=== Section 4: Shape Correctness ==="

SHAPE_CHECK=$(python3 -c "
import numpy as np
import tensor_ops

rng = np.random.RandomState(99)
ok = True
errors = []

# Test conv2d shape inference matches forward output shape
for _ in range(10):
    c_in = rng.randint(1, 8)
    c_out = rng.randint(1, 8)
    h = rng.randint(8, 33)
    w = rng.randint(8, 33)
    kh, kw = 3, 3
    stride, pad = 1, 1

    inp = rng.randn(1, c_in, h, w).astype(np.float32) * 0.1
    wt = rng.randn(c_out, c_in, kh, kw).astype(np.float32) * 0.1

    out = tensor_ops.conv2d_forward(inp, wt, stride, pad)
    shape = tensor_ops.conv2d_shape(inp.shape, wt.shape, stride, pad)

    if tuple(out.shape) != shape:
        errors.append(f'conv2d: actual={out.shape} meta={shape}')
        ok = False

# Test relu
for _ in range(5):
    dims = tuple(rng.randint(1, 16, size=rng.randint(2, 5)))
    inp = rng.randn(*dims).astype(np.float32)
    out = tensor_ops.relu_forward(inp)
    shape = tensor_ops.relu_shape(dims)
    if tuple(out.shape) != shape:
        errors.append(f'relu: actual={out.shape} meta={shape}')
        ok = False

# Test pool
for _ in range(5):
    c = rng.randint(1, 8)
    h = rng.randint(4, 17)
    w = rng.randint(4, 17)
    inp = rng.randn(1, c, h, w).astype(np.float32)
    out = tensor_ops.pool_forward(inp, 2, 2)
    shape = tensor_ops.pool_shape(inp.shape, 2, 2)
    if tuple(out.shape) != shape:
        errors.append(f'pool: actual={out.shape} meta={shape}')
        ok = False

if ok:
    print('PASS')
else:
    for e in errors:
        print(f'ERR: {e}')
" 2>&1)

if echo "$SHAPE_CHECK" | grep -q "^PASS"; then
    echo "  [PASS] All shape inference outputs match forward output shapes"
    add_score 0.15
else
    echo "  [FAIL] Shape mismatches detected:"
    echo "$SHAPE_CHECK" | grep "^ERR" | head -5
fi

# ====================================================================
# Section 5: Performance  (0.05)
# ====================================================================
echo ""
echo "=== Section 5: Performance ==="

PERF_START=$(python3 -c "import time; print(time.time())")
python3 "$TEST_SCRIPT" --num_ops 60 --seed 42 > /dev/null 2>&1 || true
PERF_END=$(python3 -c "import time; print(time.time())")
PERF_TIME=$(python3 -c "print(round($PERF_END - $PERF_START, 2))")

echo "  60-op chain completed in ${PERF_TIME}s"
if python3 -c "exit(0 if $PERF_TIME < 60 else 1)"; then
    echo "  [PASS] Under 60-second limit"
    add_score 0.05
else
    fail_section "Over 60-second limit (${PERF_TIME}s)"
fi

# ====================================================================
# Section 6: Anti-hack checks  (0.10)
# ====================================================================
echo ""
echo "=== Section 6: Anti-hack Checks ==="
ANTI_HACK=0.0

# Check 1: test_chain.py not modified
if [ -f "$TEST_SCRIPT" ]; then
    # Verify it still has the strict shape check
    if grep -q "meta_shape != actual_shape" "$TEST_SCRIPT" 2>/dev/null; then
        echo "  [PASS] test_chain.py shape check intact"
        ANTI_HACK=$(python3 -c "print(round($ANTI_HACK + 0.025, 2))")
    else
        echo "  [FAIL] test_chain.py shape check appears removed"
    fi
fi

# Check 2: setup.py not modified
if [ -f "$SETUP_FILE" ]; then
    if grep -q "tensor_ops" "$SETUP_FILE" 2>/dev/null; then
        echo "  [PASS] setup.py references tensor_ops"
        ANTI_HACK=$(python3 -c "print(round($ANTI_HACK + 0.025, 2))")
    else
        echo "  [FAIL] setup.py appears modified"
    fi
fi

# Check 3: model.py not modified
if [ -f "$MODEL_FILE" ]; then
    if grep -q "class ConvBlock" "$MODEL_FILE" 2>/dev/null; then
        echo "  [PASS] model.py ConvBlock class intact"
        ANTI_HACK=$(python3 -c "print(round($ANTI_HACK + 0.025, 2))")
    else
        echo "  [FAIL] model.py appears modified"
    fi
fi

# Check 4: No hardcoded shapes in tensor_ops.c
if [ -f "$SRC_FILE" ]; then
    if grep -cE '"\(1,\s*16,\s*32,\s*32\)"|hardcode|HACK' "$SRC_FILE" | grep -q '^0$'; then
        echo "  [PASS] No hardcoded shapes in tensor_ops.c"
        ANTI_HACK=$(python3 -c "print(round($ANTI_HACK + 0.025, 2))")
    else
        echo "  [FAIL] Suspicious hardcoded values in tensor_ops.c"
    fi
fi

add_score "$ANTI_HACK"

# ====================================================================
# Final score
# ====================================================================
echo ""
echo "========================================="
echo " Final Score: $TOTAL"
echo "========================================="

echo "$TOTAL" > "$REWARD_FILE"

if python3 -c "exit(0 if $TOTAL >= 0.6 else 1)"; then
    echo "PASS"
else
    echo "FAIL (need >= 0.60)"
fi

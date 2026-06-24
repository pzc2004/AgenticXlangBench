#!/bin/bash
# solve.sh -- Oracle fix for the 3 shape-inference bugs in tensor_ops.c.
#
# This script reverses the damage done by inject_bug.py by restoring the
# correct values at each BUG_LOCATION marker.
#
# /workspace is mounted read-only from host; copy to /build/ for modification.
set -e

# Copy workspace files to build directory
BUILD_DIR="/build"
WORKSPACE="/workspace"
echo "=== Copying source to $BUILD_DIR ==="
mkdir -p "$BUILD_DIR"
cp "$WORKSPACE"/*.c "$WORKSPACE"/*.h "$WORKSPACE"/*.py "$WORKSPACE"/*.sh "$BUILD_DIR"/ 2>/dev/null || true
# Also copy build_op.sh if present
[ -f "$WORKSPACE/build_op.sh" ] && cp "$WORKSPACE/build_op.sh" "$BUILD_DIR"/

TARGET="$BUILD_DIR/tensor_ops.c"

if [ ! -f "$TARGET" ]; then
    echo "[ERR] $TARGET not found"
    exit 1
fi

echo "=== Fixing tensor_ops.c ==="

# ---- Bug 1: conv2d_shape  (W_out - 1) -> W_out ----
sed -i 's/return Py_BuildValue("(iiii)", N, C_out, H_out, (W_out - 1));/return Py_BuildValue("(iiii)", N, C_out, H_out, W_out);/' "$TARGET"
echo "[OK] Bug 1 fixed: conv2d_shape returns W_out (not W_out-1)"

# ---- Bug 2: conv2d_forward  ci * H  ->  ci * H * W ----
# More precise fix: replace the buggy comment + code block
sed -i '/BUG_LOCATION_2: input offset -- INJECTED BUG/,+4{
    s/ci \* H$/ci * H * W/
}' "$TARGET"
echo "[OK] Bug 2 fixed: conv2d_forward uses ci*H*W (not ci*H)"

# ---- Bug 3: pool_shape  (H_out - 1) -> H_out ----
sed -i 's/return Py_BuildValue("(iiii)", N, C, (H_out - 1), W_out);/return Py_BuildValue("(iiii)", N, C, H_out, W_out);/' "$TARGET"
echo "[OK] Bug 3 fixed: pool_shape returns H_out (not H_out-1)"

echo ""
echo "All 3 bugs fixed.  Rebuild with: cd $BUILD_DIR && pip install -e ."

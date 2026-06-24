#!/bin/bash
set -e

JAX_DIR=$(python -c "
import jax._src.lax.slicing as m
import os
print(os.path.dirname(os.path.dirname(os.path.dirname(m.__file__))))
")

SRC_DIR="$JAX_DIR/_src"

echo ">>> 修复所有 bug..."

python3 /task/solution/inject_bug.py --reverse

echo ">>> 清除字节码缓存..."
find "$SRC_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo ">>> 验证修复..."
cd /workspace && python test_vmap.py --seed 42 --check 2>&1 | tail -5

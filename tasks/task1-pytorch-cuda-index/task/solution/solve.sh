#!/bin/bash
PYTORCH_SRC="/build/pytorch"

echo ">>> 修复所有 bug（回退 bugs.patch，诱饵保留）..."

PYTORCH_DIR="$PYTORCH_SRC" python3 /task/solution/inject_bug.py --reverse

echo ">>> 增量编译..."
cd "$PYTORCH_SRC/build"
ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
echo ">>> 验证修复完成"

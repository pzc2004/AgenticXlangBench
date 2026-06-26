#!/bin/bash
set -e

echo ">>> 修复所有 bug..."
python3 /task/solution/inject_bug.py --reverse

echo ">>> 验证修复..."
bash /task/tests/test.sh 2>&1 | tail -5

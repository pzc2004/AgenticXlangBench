#!/bin/bash
# Oracle:修复 3 个复合 bug (SQLite query optimizer)
set -e

SQLITE_SRC="/build/sqlite"
TARGET="$SQLITE_SRC/sqlite3.c"

if [ ! -f "$TARGET" ]; then
    echo "❌ 找不到 $TARGET"
    exit 1
fi

echo ">>> 修复 3 个复合 bug ($TARGET)..."

# Bug 1: 索引范围扫描代价翻倍
# 修复: * 2 → (删除翻倍因子)
python3 -c "
import re, sys
f = sys.argv[1]
with open(f) as fh: content = fh.read()

# Bug 1: 代价翻倍修复
patterns1 = [
    (r'nRow \* pProbe->aiRowEst\[0\] \* 2', 'nRow * pProbe->aiRowEst[0]'),
    (r'nRow \* 2 \*', 'nRow *'),
    (r'rCost\.nRow = nRow \* 2', 'rCost.nRow = nRow'),
]
for pat, repl in patterns1:
    content, n = re.subn(pat, repl, content, count=1)
    if n > 0:
        print('Fixed Bug 1: 代价翻倍')
        break

# Bug 2: WHERE 条件过滤率被忽略
# 修复: 恢复 nOBSat 的正确赋值
content, n = re.subn(
    r'nOBSat = 0  /\* BUG: ignore OBSat \*/',
    'nOBSat = 1',
    content,
    count=1
)
if n > 0:
    print('Fixed Bug 2: WHERE 条件过滤率')

# Bug 3: 行数估算除数错误
# 修复: / 4 → / 2 或 >> 2 → >> 1
patterns3 = [
    (r'/ 4  /\* BUG: wrong divisor \*/', '/ 2'),
    (r'>> 2  /\* BUG: wrong shift \*/', '>> 1'),
    (r'>>= 2  /\* BUG: wrong shift amount \*/', '>>= 1'),
]
for pat, repl in patterns3:
    content, n = re.subn(pat, repl, content, count=1)
    if n > 0:
        print('Fixed Bug 3: 行数估算除数')
        break

with open(f, 'w') as fh: fh.write(content)
" "$TARGET"

# 验证修复
if grep -q "\* 2" "$TARGET" 2>/dev/null && grep -q "BUG: wrong" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

if grep -q "nOBSat = 0  /\* BUG" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

if grep -q "BUG: wrong divisor" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 3 修复失败"
    exit 1
fi

echo "✅ 所有 bug 已修复"

# 重新编译
echo ">>> 重新编译 SQLite..."
cd "$SQLITE_SRC"
make clean && make -j$(nproc) 2>&1 | tail -3

echo ">>> 验证修复..."
bash /task/tests/test.sh 2>&1 | tail -20

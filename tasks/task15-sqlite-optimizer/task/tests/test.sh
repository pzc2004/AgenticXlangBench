#!/bin/bash
# tests/test.sh — Task 15 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  SQLite 可用
#   0.30  JOIN + WHERE 查询结果正确(Bug 1)
#   0.20  多种查询组合正确(Bug 2)
#   0.15  聚合查询正确(Bug 3)
#   0.10  性能:自编译 SQLite 查询速度正常
#   0.15  Anti-hack: 无 INDEXED BY + 无 SQL 修改 + 无后处理
#
# 如果检测到 hack,总分减半。
# 不暴露任何 bug 细节。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 15 评测"
echo "========================================="

# === 1. 基础:SQLite 可用(0.10) ===
echo ""
echo ">>> [1/6] SQLite 检查..."
SQLITE_BIN="/build/sqlite/sqlite3"
if [ ! -x "$SQLITE_BIN" ]; then
    SQLITE_BIN="/build/sqlite/build/sqlite3"
fi
if [ ! -x "$SQLITE_BIN" ]; then
    SQLITE_BIN="sqlite3"
fi

if $SQLITE_BIN --version 2>/dev/null; then
    score=0.10
    echo "  ✅ SQLite 可用 ($SQLITE_BIN)"
else
    echo "  ❌ SQLite 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 运行测试 ===
echo ""
echo ">>> [2/6] 运行查询测试..."
result=$(python "$WORKSPACE/test_query.py" 2>&1)
echo "$result"

# === 3. JOIN + WHERE 查询结果检查(0.30) ===
echo ""
echo ">>> [3/6] JOIN + WHERE 查询正确性测试(Bug 1)..."
join_passed=$(echo "$result" | grep "RESULT_JOIN_PASSED" | awk '{print $2}')
join_total=$(echo "$result" | grep "RESULT_JOIN_TOTAL" | awk '{print $2}')

if [ -n "$join_passed" ] && [ -n "$join_total" ]; then
    if [ "$join_passed" = "$join_total" ]; then
        score=0.40
        echo "  ✅ JOIN 查询 $join_passed/$join_total 通过"
    elif python -c "exit(0 if $join_passed/$join_total >= 0.7 else 1)" 2>/dev/null; then
        score=0.30
        echo "  ⚠️ JOIN 查询 $join_passed/$join_total 通过"
    else
        echo "  ❌ JOIN 查询 $join_passed/$join_total 通过"
        echo "$score" > "$REWARD_FILE"
        exit 0
    fi
else
    echo "  ❌ 无法解析 JOIN 查询结果"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 多种查询组合检查(0.20) ===
echo ""
echo ">>> [4/6] 多种查询组合测试(Bug 2)..."
total_passed=$(echo "$result" | grep "RESULT_TOTAL_PASSED" | awk '{print $2}')
total_tests=$(echo "$result" | grep "RESULT_TOTAL" | awk '{print $2}')

if [ -n "$total_passed" ] && [ -n "$total_tests" ]; then
    if [ "$total_passed" = "$total_tests" ]; then
        score=0.60
        echo "  ✅ 总计 $total_passed/$total_tests 通过"
    elif python -c "exit(0 if $total_passed/$total_tests >= 0.7 else 1)" 2>/dev/null; then
        score=0.50
        echo "  ⚠️ 总计 $total_passed/$total_tests 通过"
    else
        echo "  ❌ 总计 $total_passed/$total_tests 通过"
    fi
else
    echo "  ⚠️ 无法解析总计结果"
fi

# === 5. 聚合查询检查(0.15) ===
echo ""
echo ">>> [5/6] 聚合查询正确性测试(Bug 3)..."
agg_passed=$(echo "$result" | grep "RESULT_AGG_PASSED" | awk '{print $2}')
agg_total=$(echo "$result" | grep "RESULT_AGG_TOTAL" | awk '{print $2}')

if [ -n "$agg_passed" ] && [ -n "$agg_total" ]; then
    if [ "$agg_passed" = "$agg_total" ]; then
        score=0.75
        echo "  ✅ 聚合查询 $agg_passed/$agg_total 通过"
    else
        echo "  ⚠️ 聚合查询 $agg_passed/$agg_total 通过"
    fi
else
    echo "  ⚠️ 无法解析聚合查询结果"
fi

# === 6. 性能 + Anti-hack(0.25) ===
echo ""
echo ">>> [6/6] 性能 + Anti-hack 检查..."

# 性能测试: 自编译 SQLite 查询速度
PERF_TIME=$(python -c "
import sqlite3, time, os, random

db = '/tmp/perf_test.db'
if os.path.exists(db):
    os.remove(db)

conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute('CREATE TABLE t (id INTEGER PRIMARY KEY, x INTEGER, y INTEGER)')
cur.execute('CREATE INDEX idx_x ON t(x)')
cur.execute('CREATE INDEX idx_y ON t(y)')
random.seed(42)
for i in range(10000):
    cur.execute('INSERT INTO t VALUES (?, ?, ?)', (i, random.randint(1, 1000), random.randint(1, 1000)))
conn.commit()

t0 = time.time()
for _ in range(1000):
    cur.execute('SELECT * FROM t WHERE x > 500 AND y < 800').fetchall()
t1 = time.time()
conn.close()
os.remove(db)
print(f'{(t1-t0)/1000:.6f}')
" 2>/dev/null)

if [ -n "$PERF_TIME" ]; then
    echo "  查询时间: ${PERF_TIME}s"
else
    echo "  ⚠️ 性能测试无法运行"
fi

# Anti-hack: 检查是否有 INDEXED BY
if grep -rn "INDEXED BY\|FORCE INDEX\|USE INDEX" "$WORKSPACE/test_query.py" 2>/dev/null; then
    echo "  ❌ 发现 INDEXED BY hint"
    HACK=1
else
    echo "  ✅ 未发现 INDEXED BY hint"
fi

# Anti-hack: 检查 SQL 是否被修改
if grep -q "JOIN t2 ON t1.id = t2.t1_id" "$WORKSPACE/test_query.py" 2>/dev/null; then
    echo "  ✅ SQL 查询未被修改"
else
    echo "  ❌ SQL 查询被修改"
    HACK=1
fi

# Anti-hack: 检查是否有后处理补行
if grep -rn "result.*extend\|result.*append\|missing.*row\|补行\|补全" "$WORKSPACE/test_query.py" 2>/dev/null; then
    echo "  ❌ 发现后处理补行"
    HACK=1
else
    echo "  ✅ 未发现后处理补行"
fi

# Anti-hack: 检查关键文件未被修改
if grep -q "setup_database" "$WORKSPACE/test_query.py" 2>/dev/null; then
    echo "  ✅ test_query.py 未被修改"
else
    echo "  ❌ test_query.py 被修改"
    HACK=1
fi

if [ $HACK -eq 0 ]; then
    score=1.0
fi

# === 最终评分 ===
echo ""
echo "========================================="
if [ $HACK -eq 0 ]; then
    echo " 最终分数: $score"
else
    score=$(python -c "print(f'{$score * 0.5:.2f}')")
    echo " 最终分数: $score (检测到 hack,减半)"
fi
echo "========================================="

echo "$score" > "$REWARD_FILE"

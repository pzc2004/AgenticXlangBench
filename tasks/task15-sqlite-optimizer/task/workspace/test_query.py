#!/usr/bin/env python3
"""
SQLite 查询优化器测试脚本
用法: python test_query.py

测试多索引 + JOIN + WHERE 条件组合的查询正确性,
重点关注索引选择和结果完整性。
"""

import sqlite3
import os
import sys
import subprocess
import tempfile


def get_sqlite_path():
    """获取自编译 SQLite 的路径"""
    # 优先使用自编译的 sqlite3
    candidates = [
        "/build/sqlite/sqlite3",
        "/build/sqlite/build/sqlite3",
        "/usr/local/bin/sqlite3",
    ]
    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return "sqlite3"  # fallback to system


def setup_database(db_path):
    """创建测试数据库,包含多索引的表"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 创建表 t1: 多索引
    cur.execute("""
        CREATE TABLE t1 (
            id INTEGER PRIMARY KEY,
            x INTEGER,
            y INTEGER,
            z INTEGER,
            value TEXT
        )
    """)

    # 创建多个索引
    cur.execute("CREATE INDEX idx_t1_x ON t1(x)")
    cur.execute("CREATE INDEX idx_t1_y ON t1(y)")
    cur.execute("CREATE INDEX idx_t1_xy ON t1(x, y)")
    cur.execute("CREATE INDEX idx_t1_z ON t1(z)")

    # 创建表 t2: 用于 JOIN
    cur.execute("""
        CREATE TABLE t2 (
            id INTEGER PRIMARY KEY,
            t1_id INTEGER,
            a INTEGER,
            b INTEGER,
            value TEXT,
            FOREIGN KEY (t1_id) REFERENCES t1(id)
        )
    """)

    cur.execute("CREATE INDEX idx_t2_t1_id ON t2(t1_id)")
    cur.execute("CREATE INDEX idx_t2_a ON t2(a)")
    cur.execute("CREATE INDEX idx_t2_b ON t2(b)")

    # 插入测试数据
    # t1: 1000 行,x 在 1-100, y 在 1-100, z 在 1-50
    import random
    random.seed(42)

    for i in range(1, 1001):
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        z = random.randint(1, 50)
        value = f"t1_value_{i}"
        cur.execute("INSERT INTO t1 VALUES (?, ?, ?, ?, ?)", (i, x, y, z, value))

    # t2: 2000 行,每个 t1 有 0-3 个关联行
    t2_id = 1
    for t1_id in range(1, 1001):
        n_related = random.randint(0, 3)
        for _ in range(n_related):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            value = f"t2_value_{t2_id}"
            cur.execute("INSERT INTO t2 VALUES (?, ?, ?, ?, ?)", (t2_id, t1_id, a, b, value))
            t2_id += 1

    conn.commit()
    conn.close()


def run_query_direct(db_path, query):
    """直接用 sqlite3 模块运行查询"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows


def run_query_cli(sqlite_path, db_path, query):
    """用 SQLite CLI 运行查询"""
    try:
        result = subprocess.run(
            [sqlite_path, db_path, query],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip().split('\n') if result.stdout.strip() else []
    except Exception as e:
        return [f"ERROR: {e}"]


def get_explain_plan(db_path, query):
    """获取查询执行计划"""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"EXPLAIN QUERY PLAN {query}")
    plan = cur.fetchall()
    conn.close()
    return plan


def test_join_with_where(db_path):
    """测试 JOIN + WHERE 条件查询"""
    # 这个查询应该返回特定数量的行
    # Bug 会导致返回行数偏少
    queries = [
        {
            'name': 't1 JOIN t2 with x > 50 AND y < 80',
            'query': """
                SELECT t1.id, t1.x, t1.y, t2.a, t2.b
                FROM t1
                JOIN t2 ON t1.id = t2.t1_id
                WHERE t1.x > 50 AND t1.y < 80
            """,
            'expected_min': 50,  # 至少应该有这么多行
        },
        {
            'name': 't1 JOIN t2 with x BETWEEN 20 AND 80',
            'query': """
                SELECT t1.id, t1.x, t2.a
                FROM t1
                JOIN t2 ON t1.id = t2.t1_id
                WHERE t1.x BETWEEN 20 AND 80
            """,
            'expected_min': 100,
        },
        {
            'name': 't1 JOIN t2 with z > 25 AND a > 50',
            'query': """
                SELECT t1.id, t1.z, t2.a
                FROM t1
                JOIN t2 ON t1.id = t2.t1_id
                WHERE t1.z > 25 AND t2.a > 50
            """,
            'expected_min': 30,
        },
    ]

    results = []
    for q in queries:
        rows = run_query_direct(db_path, q['query'])
        count = len(rows)
        plan = get_explain_plan(db_path, q['query'])

        is_correct = count >= q['expected_min']
        results.append({
            'name': q['name'],
            'count': count,
            'expected_min': q['expected_min'],
            'is_correct': is_correct,
            'plan': plan,
        })

    return results


def test_index_selection(db_path):
    """测试索引选择是否正确"""
    # 用 EXPLAIN QUERY PLAN 检查是否使用了正确的索引
    queries = [
        {
            'name': 'x range scan',
            'query': "SELECT * FROM t1 WHERE x > 50 AND x < 80",
            'should_use_index': True,
        },
        {
            'name': 'y range scan',
            'query': "SELECT * FROM t1 WHERE y > 30 AND y < 70",
            'should_use_index': True,
        },
        {
            'name': 'compound index scan',
            'query': "SELECT * FROM t1 WHERE x > 20 AND y < 80",
            'should_use_index': True,
        },
    ]

    results = []
    for q in queries:
        plan = get_explain_plan(db_path, q['query'])
        plan_str = str(plan).lower()
        uses_index = 'scan' in plan_str or 'index' in plan_str

        rows = run_query_direct(db_path, q['query'])
        count = len(rows)

        results.append({
            'name': q['name'],
            'plan': plan,
            'row_count': count,
            'uses_index': uses_index,
        })

    return results


def test_aggregate_queries(db_path):
    """测试聚合查询的正确性"""
    queries = [
        {
            'name': 'COUNT with JOIN and WHERE',
            'query': """
                SELECT COUNT(*)
                FROM t1
                JOIN t2 ON t1.id = t2.t1_id
                WHERE t1.x > 50
            """,
            'check': lambda rows: rows[0][0] > 0 if rows else False,
        },
        {
            'name': 'SUM with JOIN and WHERE',
            'query': """
                SELECT SUM(t1.x)
                FROM t1
                JOIN t2 ON t1.id = t2.t1_id
                WHERE t1.y < 50
            """,
            'check': lambda rows: rows[0][0] is not None and rows[0][0] > 0 if rows else False,
        },
    ]

    results = []
    for q in queries:
        rows = run_query_direct(db_path, q['query'])
        is_correct = q['check'](rows)
        results.append({
            'name': q['name'],
            'result': rows[0][0] if rows else None,
            'is_correct': is_correct,
        })

    return results


def main():
    print("=" * 60)
    print("SQLite 查询优化器测试")
    print("=" * 60)

    # 检查 SQLite
    sqlite_path = get_sqlite_path()
    print(f"SQLite 路径: {sqlite_path}")

    # 创建测试数据库
    db_path = "/tmp/test_optimizer.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("创建测试数据库...")
    setup_database(db_path)

    # 验证数据
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM t1")
    t1_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM t2")
    t2_count = cur.fetchone()[0]
    conn.close()
    print(f"  t1: {t1_count} 行")
    print(f"  t2: {t2_count} 行")
    print()

    # 测试 1: JOIN + WHERE 查询
    print(">>> 测试 1: JOIN + WHERE 查询正确性")
    join_results = test_join_with_where(db_path)
    join_passed = 0
    for r in join_results:
        status = "✅" if r['is_correct'] else "❌"
        print(f"  {status} {r['name']}: {r['count']} 行 (期望 >= {r['expected_min']})")
        if r['is_correct']:
            join_passed += 1
    print()

    # 测试 2: 索引选择
    print(">>> 测试 2: 索引选择检查")
    index_results = test_index_selection(db_path)
    for r in index_results:
        print(f"  {r['name']}: {r['row_count']} 行, plan={r['plan']}")
    print()

    # 测试 3: 聚合查询
    print(">>> 测试 3: 聚合查询正确性")
    agg_results = test_aggregate_queries(db_path)
    agg_passed = 0
    for r in agg_results:
        status = "✅" if r['is_correct'] else "❌"
        print(f"  {status} {r['name']}: {r['result']}")
        if r['is_correct']:
            agg_passed += 1
    print()

    # 汇总
    total_join = len(join_results)
    total_agg = len(agg_results)
    total_passed = join_passed + agg_passed
    total_tests = total_join + total_agg

    print("=" * 60)
    print(f"JOIN 查询: {join_passed}/{total_join} 通过")
    print(f"聚合查询: {agg_passed}/{total_agg} 通过")
    print(f"总计: {total_passed}/{total_tests} 通过")
    print(f"整体: {'✅ 全部通过' if total_passed == total_tests else '❌ 存在失败'}")

    # 输出结构化结果供 test.sh 解析
    print(f"\nRESULT_JOIN_PASSED {join_passed}")
    print(f"RESULT_JOIN_TOTAL {total_join}")
    print(f"RESULT_AGG_PASSED {agg_passed}")
    print(f"RESULT_AGG_TOTAL {total_agg}")
    print(f"RESULT_TOTAL_PASSED {total_passed}")
    print(f"RESULT_TOTAL {total_tests}")

    # 清理
    os.remove(db_path)


if __name__ == "__main__":
    main()

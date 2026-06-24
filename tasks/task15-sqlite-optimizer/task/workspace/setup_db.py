#!/usr/bin/env python3
"""
SQLite 测试数据库初始化脚本
用法: python setup_db.py [db_path]

创建包含多索引的测试表,用于验证查询优化器行为。
"""

import sqlite3
import random
import sys


def setup_database(db_path="/tmp/test_optimizer.db"):
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
    random.seed(42)

    # t1: 1000 行
    for i in range(1, 1001):
        x = random.randint(1, 100)
        y = random.randint(1, 100)
        z = random.randint(1, 50)
        value = f"t1_value_{i}"
        cur.execute("INSERT INTO t1 VALUES (?, ?, ?, ?, ?)", (i, x, y, z, value))

    # t2: ~2000 行
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

    # 验证
    cur.execute("SELECT COUNT(*) FROM t1")
    t1_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM t2")
    t2_count = cur.fetchone()[0]

    conn.close()

    print(f"数据库创建完成: {db_path}")
    print(f"  t1: {t1_count} 行")
    print(f"  t2: {t2_count} 行")

    return db_path


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_optimizer.db"
    setup_database(db_path)

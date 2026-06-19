# Task 15: SQLite 查询优化器 Bug → 特定 Index + Join 组合返回错误行

## 概述

在 SQLite 的查询优化器中注入一个索引选择错误。
Bug 在 C 层,症状在 SQL 层表现为**特定 index + join + filter 组合返回错误行**,
**延迟跨越查询编译的多个阶段** —— parser → planner → optimizer → executor。

## Bug 设计

- **位置**:`src/where.c`(SQLite 查询优化器,C)
- **类型**:优化器在特定条件下选了错误的索引(选了 idx_a 应该选 idx_b)
- **效果**:用错误索引扫描,漏掉部分匹配行
- **触发条件**:仅当表有多个索引 + 特定 join 顺序 + 特定 WHERE 条件时

## 延迟显现机制

```
SQL: SELECT * FROM t1 JOIN t2 ON t1.id = t2.id WHERE t1.x > 5 AND t1.y < 10
    ↓ Parser:正确解析
    ↓ Planner:生成查询计划
    ↓ Optimizer:选索引 — bug 导致选了错误的索引
    ↓ Executor:用错误索引扫描,漏掉部分行
Python 通过 sqlite3 查询:拿到错误结果(少了行)
```

**关键**:不报错,只是结果少了几行。Agent 需要**知道正确结果应该有多少行**。

## 为什么难

1. **症状不报错**:结果"看起来对"(只是少了行)
2. **需要理解 SQLite 内部**:查询优化器、索引选择、执行计划
3. **触发条件复杂**:多索引 + join + filter 组合
4. **需要 EXPLAIN 验证**:要看执行计划才能发现选错了索引
5. **跨语言**:SQL → C 实现

## 预估轨迹

- **步数**:200+ 步
- **时间**:2-3 小时
- **关键转折点**:
  - 步骤 30:发现查询结果行数不对
  - 步骤 80:检查 SQL(看起来对)
  - 步骤 120:用 EXPLAIN 查看执行计划
  - 步骤 160:发现选错了索引
  - 步骤 200:找到优化器 bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 用 FORCE INDEX 指定索引 | 检查 SQL 是否加了 hint |
| 删除多余索引绕过 | diff 检查 schema |
| 改 SQL 绕过 | diff 检查 SQL 查询 |
| 后处理结果补行 | grep 结果处理代码 |
| 只改 SQL 不改 C | diff 只允许 .c/.h |

## Oracle

```bash
#!/bin/bash
cd /path/to/sqlite
git checkout HEAD -- src/where.c
make clean && make -j$(nproc)
```

## 资源需求

- **GPU**:不需要
- **编译**:1-2 分钟(SQLite 编译很快)
- **验证**:1 分钟
- **磁盘**:~50MB

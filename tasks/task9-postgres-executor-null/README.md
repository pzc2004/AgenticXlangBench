# Task 9: PostgreSQL Executor NULL 处理 Bug → SQL 返回行数不对

## 概述

在 PostgreSQL 的 nested loop join executor 中注入一个 NULL 值处理错误。
Bug 在 C 层,症状在 SQL 查询层表现为**返回行数不对**,
**且仅在有 NULL 值 + 特定 join + filter 组合时触发**。

## Bug 设计

- **位置**:`src/backend/executor/nodeNestloop.c`(C)
- **类型**:nested loop join 在有 NULL 值 + 特定 filter 条件下,跳过不该跳过的行
- **效果**:查询结果少了几行
- **触发条件**:JOIN 列有 NULL + WHERE 条件涉及 NULL 比较 + 特定 join 类型

## 延迟显现机制

```
SQL:SELECT * FROM A JOIN B ON A.id = B.id WHERE A.x > 5
    ↓ Parser → Planner → Executor
Executor 的 nested loop join:遇到 NULL 时错误跳过
    ↓ 返回的结果少了几行(但不报错)
Python 通过 psycopg2 查询:拿到错误结果
    ↓ 后续用这个结果做业务逻辑
业务逻辑异常(如统计数字不对)
```

**关键**:不报错,只是结果少了几行。Agent 需要**知道正确结果应该有多少行**才能发现。

## 为什么难

1. **跨 3 层**:SQL → PostgreSQL C executor → 结果
2. **症状不报错**:结果"看起来对"(只是少了行)
3. **需要理解 PostgreSQL 内部**:executor、join 算法、NULL 语义
4. **触发条件复杂**:NULL + join + filter 组合
5. **需要精确验证**:要对比"正确结果"和"错误结果"

## 预估轨迹

- **步数**:200+ 步
- **时间**:2-4 小时
- **关键转折点**:
  - 步骤 30:发现查询结果行数不对
  - 步骤 80:检查 SQL(看起来对)
  - 步骤 120:对比不同 join 类型,发现 nested loop 有问题
  - 步骤 160:发现 NULL 值相关
  - 步骤 200:读 C executor 代码,找到 NULL 处理 bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 改 SQL 绕过 | diff 检查 SQL 查询 |
| 用 hash join 替代 nested loop | 检查 join hint / 配置 |
| 过滤掉 NULL 数据 | diff 检查数据预处理 |
| 后处理结果补行 | grep 结果处理代码 |
| 只改 SQL 不改 C | diff 只允许 .c/.h 文件 |

## Oracle

```bash
#!/bin/bash
cd /path/to/postgresql
git checkout HEAD -- src/backend/executor/nodeNestloop.c
make -j$(nproc) && make install
```

## 资源需求

- **GPU**:不需要
- **编译**:5-15 分钟(PostgreSQL 编译)
- **验证**:1-2 分钟
- **磁盘**:~1GB

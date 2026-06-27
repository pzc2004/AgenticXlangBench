# 任务：修复 SQLite 查询结果缺失问题

## 背景

我们使用了一个从源码编译的 SQLite(源码在 `/build/sqlite/`)。
在执行特定的多表 JOIN 查询时，返回的结果行数少于预期。

## Bug 现象

运行 SQL 查询测试脚本：

```bash
cd /workspace
python test_query.py
```

预期：特定 JOIN + WHERE 条件的查询应返回 100 行结果。
实际：仅返回 ~70-80 行，缺失了部分应匹配的行。

注意：单表查询 **结果正常**。问题仅出现在多索引 + JOIN + WHERE 的组合查询。

## 已知信息

- SQLite 是从源码编译的，源码在 `/build/sqlite/`
- **特定 JOIN 查询返回的行数少于预期**
- 你需要找到并修复 SQLite 源码中导致查询结果缺失的 bug
- Bug 位于 `src/where.c` 中的查询优化器索引选择逻辑

## 你的任务

1. **理解 bug**：分析为什么特定查询返回的行数不正确
2. **定位 bug**：在 `/build/sqlite/` 的 C 源码中找到有问题的代码
3. **修复 bug**：修改 C 源码(只允许修改 `.c` / `.h` 文件)
4. **重新编译**：修复后需要重新编译 SQLite
   ```bash
   cd /build/sqlite
   make clean && make -j$(nproc)
   ```
5. **验证修复**：运行测试确认修复成功
   ```bash
   bash /task/tests/test.sh
   ```
6. **检查分数**：测试会输出 0-1 的分数。**分数 >= 0.6 才算通过。如果分数 < 0.6，说明修复不正确，请继续排查。**

## 约束条件

- **只允许修改 C 文件**(`.c` / `.h`)，不允许修改 Python/SQL 文件
- **不允许**：
  - 在 SQL 中使用 INDEXED BY 或 FORCE INDEX 指定索引
  - 删除多余索引来绕过问题
  - 修改 SQL 查询语句
  - 在 Python 层后处理补行
  - 修改 test_query.py

## 文件说明

- `/workspace/test_query.py` — SQL 查询测试脚本
- `/workspace/setup_db.py` — 数据库初始化脚本
- `/task/tests/test.sh` — 测试脚本(跑完后输出 0-1 分数)
- `/build/sqlite/` — SQLite 源码

## 运行环境

- SQLite(从源码编译)
- 编译工具：gcc， make

## 验收标准

运行 `/task/tests/test.sh`，分数 >= 0.6 视为通过。
**如果分数 < 0.6，说明修复不正确，请继续排查 SQLite 源码。**

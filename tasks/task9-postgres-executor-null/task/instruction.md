# Task: Fix PostgreSQL Query Result Inconsistency

## Background

We are running a PostgreSQL 15 server compiled from source. When executing join queries involving tables with NULL values in the join key columns, the query results are inconsistent — some expected rows are missing or extra rows appear unexpectedly.

## Bug Symptoms

Run the test script:

```bash
cd /workspace
python test_queries.py
```

Expected: All 6 join query tests pass (correct row counts).
Actual: Several tests fail — the row counts don't match expectations.

The issue appears to be related to:
- **NULL values** in join key columns
- **Nested loop join** execution strategy
- Queries with **specific join + filter combinations**

## Known Information

- PostgreSQL is compiled from source; source code is in `/usr/src/postgresql/`
- The relevant executor code is in `/usr/src/postgresql/src/backend/executor/`
- The bug is in the **nested loop join executor** (C code), not in SQL syntax
- Single-threaded; the issue is a logic error, not a race condition

## Your Task

1. **Understand the bug**: Analyze why join queries return wrong results
2. **Locate the bug**: Find the problematic code in the PostgreSQL executor source
3. **Fix the bug**: Modify the C source code (only `.c` / `.h` files allowed)
4. **Recompile**: After fixing, recompile PostgreSQL:
   ```bash
   cd /usr/src/postgresql
   make -j$(nproc) -C src/backend/executor
   make -j$(nproc)
   make install
   ```
5. **Restart PostgreSQL**: Restart the server for changes to take effect:
   ```bash
   pg_ctl -D /var/lib/postgresql/data restart -l /var/lib/postgresql/logfile
   # or: service postgresql restart
   ```
6. **Verify the fix**: Run the test script again:
   ```bash
   bash /task/tests/test.sh
   ```
7. **Check the score**: The test outputs a 0–1 score. **Score >= 0.6 counts as passing.**

## Constraints

- **Only modify `.c` / `.h` files** in the PostgreSQL executor source
- **NOT allowed**:
  - Modifying the test script or test data
  - Changing PostgreSQL configuration (e.g., `enable_nestloop = off`)
  - Using SQL workarounds (e.g., `COALESCE`, `IFNULL`)
  - Replacing the join with a subquery or CTE
  - Using a different PostgreSQL version

## File Layout

- `/workspace/test_queries.py` — Join query test script
- `/task/tests/test.sh` — Automated judge (outputs 0–1 score)
- `/usr/src/postgresql/src/backend/executor/` — PostgreSQL executor source

## Environment

- PostgreSQL 15 (compiled from source)
- Python 3.x with psycopg2
- Ubuntu 22.04

## Acceptance Criteria

Run `/task/tests/test.sh`. Score >= 0.6 counts as passing.

# Task: Fix Redis Module Data Corruption Bug

## Background

We have a Redis server with a custom C module (`buggy`) loaded. The module provides commands `buggy.set`, `buggy.setnx`, `buggy.append`, and `buggy.get` for storing and retrieving string values using direct memory access (DMA).

After running certain command sequences, adjacent keys in Redis get corrupted — their values become garbled or change unexpectedly.

## Bug Symptoms

Run the test script:

```bash
cd /workspace
python test_redis.py
```

Expected: All 4 tests pass (module loaded, basic ops work, no corruption with safe values, no corruption with aligned values).
Actual: Data corruption occurs with certain value lengths, causing adjacent keys to be damaged.

The corruption appears to be related to:
- **Specific value lengths** (particularly 16-byte aligned lengths)
- **Memory layout** — corruption of adjacent keys
- The custom module's **direct memory access (DMA)** operations

## Known Information

- Redis is compiled from source; source code is in `/build/redis/`
- The custom module source is in `/build/redis/modules/buggy/module.c`
- The module uses `RedisModule_StringDMA` for direct memory access
- The bug is in the module's C code, not in Redis itself
- The module is loaded via `MODULE LOAD /build/redis/modules/buggy/module.so`

## Your Task

1. **Understand the bug**: Analyze why certain command sequences corrupt data
2. **Locate the bug**: Find the problematic code in the module source
3. **Fix the bug**: Modify the C source code (only `.c` / `.h` files allowed)
4. **Rebuild the module**: After fixing, recompile:
   ```bash
   cd /build/redis/modules/buggy
   make clean && make -j$(nproc)
   ```
5. **Reload the module**: Unload and reload in Redis:
   ```bash
   redis-cli MODULE UNLOAD buggy
   redis-cli MODULE LOAD /build/redis/modules/buggy/module.so
   ```
6. **Verify the fix**: Run the test script again:
   ```bash
   bash /task/tests/test.sh
   ```
7. **Check the score**: The test outputs a 0–1 score. **Score >= 0.6 counts as passing.**

## Constraints

- **Only modify `.c` / `.h` files** in the module source directory
- **NOT allowed**:
  - Modifying the test script or test data
  - Using native Redis commands to bypass the module
  - Replacing the module .so with a pre-built version
  - Modifying Redis server configuration

## File Layout

- `/workspace/test_redis.py` — Redis command test script
- `/workspace/module.c` — Module source (reference copy)
- `/workspace/Makefile` — Build script (reference)
- `/task/tests/test.sh` — Automated judge (outputs 0–1 score)
- `/build/redis/modules/buggy/module.c` — Module source (editable)
- `/build/redis/` — Redis source tree

## Environment

- Redis 7.x (compiled from source)
- Custom module with DMA-based string storage
- GCC with Redis module API headers

## Acceptance Criteria

Run `/task/tests/test.sh`. Score >= 0.6 counts as passing.

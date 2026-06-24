# Task: Fix Python C Extension Multi-threaded Race Condition

## Background

We have a Python C extension module (`compute`) that provides array computation functions. The module works perfectly in single-threaded usage, but produces incorrect results when called from multiple threads concurrently.

## Bug Symptoms

Run the test script:

```bash
cd /workspace
pip install -e .
python test_gil.py
```

Expected: All 4 tests pass (single-threaded, multi-threaded consistency, counter integrity, stress test).
Actual: Single-threaded test passes, but multi-threaded tests fail intermittently.

The issue is related to:
- **GIL (Global Interpreter Lock)** handling in the C extension
- **Shared state** (`global_counter`, `global_accumulator`) being modified without proper synchronization
- **Probabilistic** failures — more threads and iterations increase failure likelihood

## Known Information

- The C extension source is at `/workspace/compute.c`
- The module uses `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` for GIL management
- Single-threaded: always correct
- Multi-threaded: probabilistic incorrect results
- The bug is in the GIL release/acquire placement, not in the computation logic

## Your Task

1. **Understand the bug**: Run the test with different thread counts to see the pattern
2. **Locate the bug**: Find where the GIL is released too early in `compute.c`
3. **Fix the bug**: Modify the C source code (only `.c` / `.h` files allowed)
4. **Rebuild the extension**: After fixing, reinstall:
   ```bash
   cd /workspace
   pip install -e .
   ```
5. **Verify the fix**: Run the test script again:
   ```bash
   bash /task/tests/test.sh
   ```
6. **Check the score**: The test outputs a 0–1 score. **Score >= 0.6 counts as passing.**

## Constraints

- **Only modify `.c` / `.h` files** in the extension source
- **NOT allowed**:
  - Using `threading.Lock` or `multiprocessing.Lock` in Python to mask the C bug
  - Using `multiprocessing` instead of `threading`
  - Modifying the test script
  - Setting thread count to 1

## File Layout

- `/workspace/compute.c` — C extension source (editable)
- `/workspace/setup.py` — Build script
- `/workspace/test_gil.py` — Multi-threaded test script
- `/task/tests/test.sh` — Automated judge (outputs 0–1 score)

## Environment

- Python 3.x with development headers
- GCC/Clang with C compilation support
- Multi-core system (for meaningful race condition testing)

## Acceptance Criteria

Run `/task/tests/test.sh`. Score >= 0.6 counts as passing.

# Task: Fix Random Segfaults in Rust FFI Extension

## Background

We have a Rust-based Python extension library (`rustops`) that provides high-performance data processing via PyO3. The source code is in `/workspace/`.

## Bug Symptoms

Run the stress test:
```bash
cd /workspace
pip install -e .
python test_stress.py --iterations 10000
```

Expected: All iterations complete successfully.
Actual: Random segfaults occur (approximately every 1,000-5,000 iterations).

Notes:
- Does not crash every time — probabilistic trigger
- No Python traceback on crash — only segfault
- Sometimes does not crash but returns corrupted data

## Known Information

- Rust source is in `/workspace/src/ffi.rs`
- The bugs are in `unsafe` blocks
- Probabilistic trigger (depends on when the memory allocator reuses freed addresses)

## Your Task

1. Reproduce the segfault (may require multiple runs)
2. Locate the bugs in `/workspace/src/ffi.rs`
3. Fix the bugs (only `.rs` files may be modified)
4. Recompile: `cd /workspace && pip install -e .`
5. Verify: `bash /task/tests/test.sh`

## Constraints

- Only `.rs` files may be modified
- Do NOT modify `test_stress.py` or `Cargo.toml`
- Do NOT use try/except to catch segfaults
- Do NOT replace Rust implementation with Python
- Do NOT use `Box::leak` to mask memory issues

## File Reference

- `/workspace/src/ffi.rs` — FFI function implementations (contains unsafe code)
- `/workspace/src/lib.rs` — PyO3 module definition
- `/workspace/test_stress.py` — Stress test script
- `/task/tests/test.sh` — Evaluation script

## Acceptance Criteria

Run `/task/tests/test.sh`. A score >= 0.6 is required to pass.

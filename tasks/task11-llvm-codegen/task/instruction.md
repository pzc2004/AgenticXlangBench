# Task: Fix Compiler Optimization Bug (-O2 Produces Wrong Results)

## Background

We are using a version of LLVM/Clang compiled from source. A specific C test program produces correct results when compiled with `-O0` but incorrect results when compiled with `-O2`. This indicates a bug in the compiler's optimization passes.

## Bug Symptoms

Run the test program at different optimization levels:

```bash
cd /workspace

# Compile and run with -O0 (correct)
clang -O0 -o test_O0 test_codegen.c && ./test_O0

# Compile and run with -O2 (buggy)
clang -O2 -o test_O2 test_codegen.c && ./test_O2
```

Expected: Both `-O0` and `-O2` produce identical correct results.
Actual: `-O0` passes all tests, but `-O2` fails on sign-extension/truncation patterns.

The bug is related to:
- **Sign extension** (`MOVSX`) vs **zero extension** (`MOVZX`) instruction selection
- Specific **IR patterns**: `sext` + `trunc` combinations
- Only triggers under **`-O2` optimization** (not `-O0`)

## Known Information

- LLVM is compiled from source; source code is in `/build/llvm-project/`
- The relevant code is in `/build/llvm-project/llvm/lib/Target/X86/`
- The bug is in **instruction selection** (`X86ISelDAGToDAG.cpp`), not in the test program
- The `llc` tool may be available at `/build/llvm-project/build/bin/llc`

## Your Task

1. **Understand the bug**: Compare `-O0` and `-O2` assembly output
2. **Locate the bug**: Find the wrong instruction selection in LLVM source
3. **Fix the bug**: Modify LLVM C++ source (only `.cpp` / `.h` / `.td` files)
4. **Rebuild LLVM**: After fixing, rebuild the `llc` target:
   ```bash
   cd /build/llvm-project
   cmake --build build --target llc -j$(nproc)
   ```
5. **Verify the fix**: Recompile the test with the fixed compiler and run:
   ```bash
   bash /task/tests/test.sh
   ```
6. **Check the score**: The test outputs a 0–1 score. **Score >= 0.6 counts as passing.**

## Constraints

- **Only modify `.cpp` / `.h` / `.td` files** in the LLVM x86 backend
- **NOT allowed**:
  - Modifying the test C program to avoid the bug pattern
  - Using `volatile` or `asm` to prevent optimization
  - Using `-O0` as a workaround (performance must match `-O2`)
  - Using a different compiler (gcc)

## File Layout

- `/workspace/test_codegen.c` — C test program with sign-extension patterns
- `/workspace/test_asm.sh` — Assembly comparison script
- `/task/tests/test.sh` — Automated judge (outputs 0–1 score)
- `/build/llvm-project/llvm/lib/Target/X86/` — LLVM x86 backend source

## Environment

- LLVM 18+ (compiled from source)
- Clang (from the LLVM build)
- GCC available as fallback
- x86_64 architecture

## Acceptance Criteria

Run `/task/tests/test.sh`. Score >= 0.6 counts as passing.

#!/bin/bash
# Oracle: Fix 3 compound bugs in LLVM x86 instruction selection
set -e

LLVM_DIR="${LLVM_DIR:-/build/llvm-project}"
TARGET="$LLVM_DIR/llvm/lib/Target/X86/X86ISelDAGToDAG.cpp"

echo ">>> Fixing 3 compound bugs in X86ISelDAGToDAG.cpp..."

# Bug 1: MOVZX32rr8 → MOVSX32rr8 (restore sign-extend for i8→i32)
sed -i 's/X86::MOVZX32rr8/X86::MOVSX32rr8/' "$TARGET"

# Bug 2: MOVZX32rr16 → MOVSX32rr16 (restore sign-extend for i16→i32)
sed -i 's/X86::MOVZX32rr16/X86::MOVSX32rr16/' "$TARGET"

# Bug 3: SETL → SETG (restore correct signed comparison)
# First try the X86:: prefix versions
sed -i 's/X86::SETL\b/X86::SETG/' "$TARGET"
sed -i 's/X86::COND_LE\b/X86::COND_GE/' "$TARGET"

# Verify fixes
echo ">>> Verifying fixes..."
errors=0

if grep -q 'MOVZX32rr8' "$TARGET" 2>/dev/null; then
    echo "  Bug 1: NOT FIXED (MOVZX32rr8 still present)"
    errors=$((errors + 1))
else
    echo "  Bug 1: FIXED"
fi

if grep -q 'MOVZX32rr16' "$TARGET" 2>/dev/null; then
    echo "  Bug 2: NOT FIXED (MOVZX32rr16 still present)"
    errors=$((errors + 1))
else
    echo "  Bug 2: FIXED"
fi

if [ $errors -gt 0 ]; then
    echo "WARNING: $errors bugs not fully fixed"
fi

# Rebuild LLVM (incremental)
echo ">>> Rebuilding LLVM..."
cd "$LLVM_DIR"
if [ -d "build" ]; then
    cmake --build build --target llc -j$(nproc) 2>&1 | tail -5
else
    mkdir -p build && cd build
    cmake -G Ninja -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_BUILD_TYPE=Release ../llvm 2>&1 | tail -5
    ninja llc 2>&1 | tail -5
fi

# Verify (copy test file to /tmp since /workspace is read-only)
echo ">>> Verifying fix..."
cp /workspace/test_codegen.c /tmp/test_codegen.c
cd /tmp
clang -O2 -o test_O2_fixed test_codegen.c 2>/dev/null || gcc -O2 -o test_O2_fixed test_codegen.c
OUTPUT=$(./test_O2_fixed 2>&1)
echo "$OUTPUT"

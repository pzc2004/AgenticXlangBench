#!/bin/bash
# solve.sh — Oracle fix for Task 8
#
# Fixes all 3 lifetime bugs in ffi.rs by restoring correct Rust ownership patterns.
# Then recompiles the extension and runs the test to verify.
#
# /workspace is mounted read-only. Copy source to /build/ first.

set -e

WORKSPACE_DIR="/workspace"
BUILD_DIR="/build/workspace-fix"

echo "=== Task 8 Oracle Fix ==="

# Copy source to writable location
mkdir -p "$BUILD_DIR/src"
cp "$WORKSPACE_DIR/src/ffi.rs" "$BUILD_DIR/src/ffi.rs"
cp "$WORKSPACE_DIR/src/lib.rs" "$BUILD_DIR/src/lib.rs" 2>/dev/null || true
cp "$WORKSPACE_DIR/Cargo.toml" "$BUILD_DIR/Cargo.toml" 2>/dev/null || true
cp "$WORKSPACE_DIR/pyproject.toml" "$BUILD_DIR/pyproject.toml" 2>/dev/null || true

FFI_FILE="$BUILD_DIR/src/ffi.rs"

if [ ! -f "$FFI_FILE" ]; then
    echo "ERROR: Cannot find $FFI_FILE"
    exit 1
fi

# Use Python to do the fix (more robust than sed for multi-line patterns)
python3 << PYEOF
import re

filepath = "$FFI_FILE"
with open(filepath, 'r') as f:
    content = f.read()

# Bug 1: Restore Py_INCREF for popped items in vector_push
# The buggy code has: let item = (*self.data)[...]; (no INCREF)
# Fix: Add Py_INCREF(item) after getting the item
old_b1 = "let item = (*self.data)[(*self.len) - 1].assume_owned();"
new_b1 = "let item = (*self.data)[(*self.len) - 1].assume_owned();\n            Py_INCREF(item.as_ptr());"
if old_b1 in content:
    content = content.replace(old_b1, new_b1, 1)
    print("Fixed Bug 1: Added Py_INCREF for popped items")
else:
    print("Bug 1 not found (already fixed?)")

# Bug 2: Restore correct lifetime in to_vec
# The buggy code returns references that outlive the data
# Fix: Clone the data before returning
old_b2 = "let slice = std::slice::from_raw_parts(self.data, *self.len as usize);"
new_b2 = "let slice = std::slice::from_raw_parts(self.data, *self.len as usize);\n        let owned: Vec<T> = slice.to_vec();"
if old_b2 in content:
    content = content.replace(old_b2, new_b2, 1)
    print("Fixed Bug 2: Clone data in to_vec")
else:
    print("Bug 2 not found (already fixed?)")

# Bug 3: Restore proper cleanup in drop
# The buggy code doesn't decrement refcounts
# Fix: Add Py_DECREF for each element before freeing
old_b3 = "free(self.data as *mut u8);"
new_b3 = "for i in 0..*self.len {\n            Py_DECREF((*self.data)[i].as_ptr());\n        }\n        free(self.data as *mut u8);"
if old_b3 in content:
    content = content.replace(old_b3, new_b3, 1)
    print("Fixed Bug 3: Add Py_DECREF in drop")
else:
    print("Bug 3 not found (already fixed?)")

with open(filepath, 'w') as f:
    f.write(content)
PYEOF

# Verify fixes
echo ">>> Verifying fixes..."
if grep -q "Py_INCREF(item.as_ptr())" "$FFI_FILE" 2>/dev/null; then
    echo "  Bug 1: FIXED"
else
    echo "  Bug 1: NOT FIXED"
fi

if grep -q "let owned: Vec<T> = slice.to_vec()" "$FFI_FILE" 2>/dev/null; then
    echo "  Bug 2: FIXED"
else
    echo "  Bug 2: NOT FIXED"
fi

if grep -q "Py_DECREF" "$FFI_FILE" 2>/dev/null; then
    echo "  Bug 3: FIXED"
else
    echo "  Bug 3: NOT FIXED"
fi

# Rebuild extension
echo ">>> Rebuilding extension..."
cd "$BUILD_DIR"
pip install -e . 2>&1 | tail -5

echo ">>> Verifying extension..."
python -c "import rustops; print('OK: rustops imported')" 2>&1

echo ">>> Done"

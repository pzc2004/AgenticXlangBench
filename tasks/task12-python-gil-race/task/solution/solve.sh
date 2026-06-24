#!/bin/bash
# Oracle: Fix 3 GIL race condition bugs in compute.c
set -e

# /workspace is mounted read-only, so copy source to /build/ first
BUILD_DIR="/build/compute-fix"
mkdir -p "$BUILD_DIR"
cp /workspace/compute.c "$BUILD_DIR/compute.c"
cp /workspace/setup.py "$BUILD_DIR/setup.py"

TARGET="$BUILD_DIR/compute.c"

echo ">>> Fixing 3 GIL race condition bugs in compute.c..."

# Fix Bug 1 & 3: Move shared state updates to proper GIL zones
# Pattern: Find the buggy compute_sum function and fix it

python3 << 'PYEOF'
import re

target = "/build/compute-fix/compute.c"
with open(target, "r") as f:
    content = f.read()

# Fix compute_sum:
# Before (buggy):
#   Py_BEGIN_ALLOW_THREADS
#   global_counter += n;
#   global_call_count++;
#   ... computation ...
#   global_accumulator += sum;
#   Py_END_ALLOW_THREADS

# After (fixed):
#   global_counter += n;
#   global_call_count++;
#   Py_BEGIN_ALLOW_THREADS
#   ... computation ...
#   Py_END_ALLOW_THREADS
#   global_accumulator += sum;  (after re-acquiring GIL)

# For compute_sum function
old_sum_pattern = '''    Py_BEGIN_ALLOW_THREADS  /* BUG: GIL released before updating shared state */

    /* These updates are now unprotected! */
    global_counter += n;     /* RACE: another thread may be updating too */
    global_call_count++;     /* RACE: increment is not atomic */

    /* Compute sum (this part is safe to run without GIL) */
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        sum += data[i];
    }

    /*
     * BUG 3: Shared state update after GIL re-acquisition is in wrong order
     * Correct: acquire GIL, then update accumulator
     * Buggy:   update accumulator while still in NO-GIL zone
     */
    global_accumulator += sum;  /* BUG: update without GIL */

    Py_END_ALLOW_THREADS'''

new_sum_code = '''    /* Update shared state with GIL held (before releasing) */
    global_counter += n;
    global_call_count++;

    Py_BEGIN_ALLOW_THREADS

    /* Compute sum (this part is safe to run without GIL) */
    double sum = 0.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        sum += data[i];
    }

    Py_END_ALLOW_THREADS

    /* Update accumulator with GIL held (after re-acquiring) */
    global_accumulator += sum;'''

if old_sum_pattern in content:
    content = content.replace(old_sum_pattern, new_sum_code)
    print("Fixed compute_sum function")
else:
    print("WARNING: compute_sum pattern not found (may already be fixed)")

# For compute_product function
old_prod_pattern = '''    /* BUG 1: Same premature GIL release */
    Py_BEGIN_ALLOW_THREADS  /* BUG: released before shared state update */

    global_counter += n;     /* RACE */
    global_call_count++;     /* RACE */

    /* Compute product */
    double product = 1.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        product *= data[i];
    }

    /* BUG: update shared state without GIL */
    global_accumulator += log(fabs(product) + 1e-10);  /* BUG: no GIL */

    Py_END_ALLOW_THREADS'''

new_prod_code = '''    /* Update shared state with GIL held */
    global_counter += n;
    global_call_count++;

    Py_BEGIN_ALLOW_THREADS

    /* Compute product (safe without GIL) */
    double product = 1.0;
    for (Py_ssize_t i = 0; i < n; i++) {
        product *= data[i];
    }

    Py_END_ALLOW_THREADS

    /* Update accumulator with GIL held */
    global_accumulator += log(fabs(product) + 1e-10);'''

if old_prod_pattern in content:
    content = content.replace(old_prod_pattern, new_prod_code)
    print("Fixed compute_product function")
else:
    print("WARNING: compute_product pattern not found (may already be fixed)")

with open(target, "w") as f:
    f.write(content)

print("All fixes applied")
PYEOF

# Verify fixes
echo ">>> Verifying fixes..."
errors=0

# Check that shared state updates come BEFORE Py_BEGIN_ALLOW_THREADS
if grep -A5 "Py_BEGIN_ALLOW_THREADS" "$TARGET" | grep -q "global_counter"; then
    echo "  Bug 1: NOT FIXED (counter update still in NO-GIL zone)"
    errors=$((errors + 1))
else
    echo "  Bug 1: FIXED"
fi

if grep -B2 "Py_END_ALLOW_THREADS" "$TARGET" | grep -q "global_accumulator"; then
    echo "  Bug 3: NOT FIXED (accumulator still updated without GIL)"
    errors=$((errors + 1))
else
    echo "  Bug 3: FIXED"
fi

if [ $errors -gt 0 ]; then
    echo "WARNING: $errors bugs not fully fixed"
fi

# Rebuild extension from /build/ and install globally
echo ">>> Rebuilding extension..."
cd "$BUILD_DIR"
pip install . 2>&1 | tail -10

# Verify the module is importable
python3 -c "import compute; print('compute module installed successfully')" || {
    echo "ERROR: compute module not importable after install"
    exit 1
}

# Run verification test
echo ">>> Verifying fix..."
cp /workspace/test_gil.py /tmp/test_gil.py
cd /tmp
python3 test_gil.py --threads 4 --iterations 100

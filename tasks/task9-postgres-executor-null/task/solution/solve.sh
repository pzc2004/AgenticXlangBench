#!/bin/bash
# Oracle: Fix 3 compound bugs in PostgreSQL executor
#
# IMPORTANT: /workspace is mounted read-only from the host at runtime.
# PostgreSQL source code lives in /usr/src/postgresql (writable inside the container).

set -e

PG_SRC="${PG_SRC:-/usr/src/postgresql}"
TARGET="$PG_SRC/src/backend/executor/nodeNestloop.c"

echo ">>> Fixing 3 compound bugs in nodeNestloop.c..."

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Cannot find $TARGET"
    exit 1
fi

# Bug 1: Reverse join qualification (undo !ExecQual → ExecQual)
# The buggy code has: qual_ok = !ExecQual(node->js.ps.qual, econtext);
# Fix: remove the ! to restore: qual_ok = ExecQual(node->js.ps.qual, econtext);
sed -i 's/qual_ok = !ExecQual/qual_ok = ExecQual/' "$TARGET"

# Bug 2: Restore end-of-inner-scan logic (false → true)
# The buggy code has: node->nl_NeedNewOuter = false;
# Fix: restore to: node->nl_NeedNewOuter = true;
sed -i 's/node->nl_NeedNewOuter = false/node->nl_NeedNewOuter = true/' "$TARGET"

# Bug 3: Restore need-new-outer check (remove the negation reversal)
# The buggy code has: if (node->nl_NeedNewOuter)  /* BUG: reversed condition */
# Fix: restore to: if (!node->nl_NeedNewOuter)
sed -i 's/if (node->nl_NeedNewOuter)  \/\* BUG/if (!node->nl_NeedNewOuter)/' "$TARGET"
sed -i 's/if (node->nl_NeedNewOuter)/if (!node->nl_NeedNewOuter)/' "$TARGET"

# Verify fixes
echo ">>> Verifying fixes..."

errors=0

# Check Bug 1 fix: should NOT have !ExecQual
if grep -q '!ExecQual(node->js.ps.qual' "$TARGET" 2>/dev/null; then
    echo "  Bug 1: NOT FIXED (!ExecQual still present)"
    errors=$((errors + 1))
else
    echo "  Bug 1: FIXED"
fi

# Check Bug 3 fix: should have !node->nl_NeedNewOuter
if grep -q 'if (!node->nl_NeedNewOuter)' "$TARGET" 2>/dev/null; then
    echo "  Bug 3: FIXED"
else
    echo "  Bug 3: NOT FIXED"
    errors=$((errors + 1))
fi

if [ $errors -gt 0 ]; then
    echo "WARNING: $errors bugs not fully fixed"
fi

# Recompile PostgreSQL
echo ">>> Recompiling PostgreSQL..."
cd "$PG_SRC"
make -j$(nproc) -C src/backend/executor 2>&1 | tail -3
make -j$(nproc) 2>&1 | tail -3
make install 2>&1 | tail -3

# Restart PostgreSQL (start if not running)
echo ">>> Restarting PostgreSQL..."
pg_ctl -D /var/lib/postgresql/data restart -l /var/lib/postgresql/logfile 2>/dev/null || \
pg_ctlcluster 15 main restart 2>/dev/null || \
service postgresql restart 2>/dev/null || true

sleep 2

echo ">>> Oracle fix applied and PostgreSQL recompiled."

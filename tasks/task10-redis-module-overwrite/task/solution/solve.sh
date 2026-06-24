#!/bin/bash
# Oracle: Fix 3 compound bugs in Redis module
#
# IMPORTANT: /workspace is mounted read-only from the host at runtime.
# Module source code lives in /build/redis/modules/buggy/ (writable inside the container).
#
# Fixes:
#   Bug 1: Off-by-one write in BUGGY.SET   (memcpy len+1 -> len)
#   Bug 2: Incorrect size in BUGGY.SETNX   (StringTruncate len+1 -> len)
#   Bug 3: Off-by-one in BUGGY.APPEND      (memcpy append_len+1 -> append_len)
#   Compilation fix: Replace RedisModule_GetCurrentExpire (server-internal, not in
#   public Module API) with RedisModule_StringDMA(key, &cur_len, REDISMODULE_READ).

set -e

MODULE_DIR="${MODULE_DIR:-/build/redis/modules/buggy}"
TARGET="$MODULE_DIR/module.c"

echo ">>> Fixing 3 compound bugs in module.c..."

if [ ! -f "$TARGET" ]; then
    echo "ERROR: Cannot find $TARGET"
    exit 1
fi

# Bug 1: Fix off-by-one write (len + 1 -> len)
# Use 'g' flag to replace ALL occurrences (there are two: BuggySet and BuggySetNX)
sed -i 's/memcpy(dma_buf, value, len + 1)/memcpy(dma_buf, value, len)/g' "$TARGET"

# Bug 2: Fix incorrect size calculation (len + 1 -> len)
sed -i 's/RedisModule_StringTruncate(key, len + 1)/RedisModule_StringTruncate(key, len)/' "$TARGET"

# Bug 3: Fix off-by-one in append (append_len + 1 -> append_len)
sed -i 's/memcpy(dma_buf + old_len, append_val, append_len + 1)/memcpy(dma_buf + old_len, append_val, append_len)/' "$TARGET"

# Fix compilation error: Replace RedisModule_GetCurrentExpire (server-internal function,
# not part of the public Redis Module API) with RedisModule_StringDMA in READ mode
# to correctly get the current key's string length.
#
# Before (won't compile - RedisModule_GetCurrentExpire is not in redismodule.h):
#   if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
#       size_t cur_len;
#       RedisModule_StringPtrLen(
#           RedisModule_CreateStringFromString(ctx, RedisModule_GetCurrentExpire(key) ?
#               NULL : key, ""), &cur_len);
#       old_len = cur_len;
#   }
#
# After (uses public Module API):
#   if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
#       size_t cur_len;
#       char *cur_buf = RedisModule_StringDMA(key, &cur_len, REDISMODULE_READ);
#       if (cur_buf) old_len = cur_len;
#   }
python3 -c "
import re
with open('$TARGET', 'r') as f:
    content = f.read()

# Replace the broken BuggyAppend block that uses RedisModule_GetCurrentExpire
old_block = '''    if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
        size_t cur_len;
        RedisModule_StringPtrLen(
            RedisModule_CreateStringFromString(ctx, RedisModule_GetCurrentExpire(key) ?
                NULL : key, \"\"), &cur_len);
        old_len = cur_len;
    }'''

new_block = '''    if (RedisModule_KeyType(key) != REDISMODULE_KEYTYPE_EMPTY) {
        size_t cur_len;
        char *cur_buf = RedisModule_StringDMA(key, &cur_len, REDISMODULE_READ);
        if (cur_buf) old_len = cur_len;
    }'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open('$TARGET', 'w') as f:
        f.write(content)
    print('  Compilation fix: applied')
else:
    print('  Compilation fix: pattern not found (may already be fixed)')
"

# Verify fixes
echo ">>> Verifying fixes..."
errors=0

if grep -q 'memcpy(dma_buf, value, len + 1)' "$TARGET" 2>/dev/null; then
    echo "  Bug 1: NOT FIXED"
    errors=$((errors + 1))
else
    echo "  Bug 1: FIXED"
fi

if grep -q 'StringTruncate(key, len + 1)' "$TARGET" 2>/dev/null; then
    echo "  Bug 2: NOT FIXED"
    errors=$((errors + 1))
else
    echo "  Bug 2: FIXED"
fi

if grep -q 'append_len + 1)' "$TARGET" 2>/dev/null; then
    echo "  Bug 3: NOT FIXED"
    errors=$((errors + 1))
else
    echo "  Bug 3: FIXED"
fi

if grep -q 'RedisModule_GetCurrentExpire' "$TARGET" 2>/dev/null; then
    echo "  Compilation fix: NOT FIXED (RedisModule_GetCurrentExpire still present)"
    errors=$((errors + 1))
else
    echo "  Compilation fix: FIXED"
fi

if [ $errors -gt 0 ]; then
    echo "WARNING: $errors issues not fully fixed"
fi

# Rebuild module
echo ">>> Rebuilding Redis module..."
cd "$MODULE_DIR"
make clean && make -j$(nproc)

# Reload module in Redis
echo ">>> Reloading module..."
redis-cli MODULE UNLOAD buggy 2>/dev/null || true
redis-cli MODULE LOAD "$MODULE_DIR/module.so"

echo ">>> Oracle fix applied and module rebuilt."

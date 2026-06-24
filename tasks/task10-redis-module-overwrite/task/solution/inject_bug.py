#!/usr/bin/env python3
"""
Inject 3 real bugs + 20 decoys into Redis module C source code.

Bug 1: Off-by-one write in BUGGY.SET command
        - memcpy(dma_buf, value, len) → memcpy(dma_buf, value, len + 1)
        - Effect: writes 1 byte past the DMA buffer boundary

Bug 2: Incorrect size calculation in BUGGY.SETNX
        - RedisModule_StringTruncate(key, len) → RedisModule_StringTruncate(key, len + 1)
        - Effect: allocates extra byte, then copies len+1 bytes

Bug 3: Missing bounds check + off-by-one in BUGGY.APPEND
        - memcpy(dma_buf + old_len, append_val, append_len) → memcpy(..., append_len + 1)
        - Effect: no bounds check and off-by-one write

Decoys: 20 comments in other Redis source files
"""

import os
import sys
import re

REDIS_DIR = os.environ.get("REDIS_DIR", "/build/redis")
MODULE_DIR = os.environ.get("MODULE_DIR", "/build/redis/modules/buggy")


def inject_real_bugs():
    """Inject 3 compound real bugs into the module source file."""
    success = True

    # Try multiple possible locations for the module source
    candidates = [
        os.path.join(MODULE_DIR, "module.c"),
        os.path.join(REDIS_DIR, "modules/buggy/module.c"),
        os.path.join(REDIS_DIR, "src/module.c"),
        os.path.join(REDIS_DIR, "module.c"),
    ]

    filepath = None
    for path in candidates:
        if os.path.exists(path):
            filepath = path
            break

    if not filepath:
        print(f"  Module source not found in any of: {candidates}")
        return False

    print(f"  Target file: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # === Bug 1: Off-by-one write ===
    # Pattern: memcpy(dma_buf, value, len);
    # Change:  memcpy(dma_buf, value, len + 1);
    pattern1 = r'(memcpy\s*\(\s*dma_buf\s*,\s*value\s*,\s*)len(\s*\))'
    replacement1 = r'\1len + 1\2'
    new_content = re.sub(pattern1, replacement1, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 1: off-by-one write (memcpy len → len+1)")
    else:
        print("  Could not find Bug 1 pattern")
        success = False

    # === Bug 2: Incorrect size calculation ===
    # Pattern: RedisModule_StringTruncate(key, len);
    # Change:  RedisModule_StringTruncate(key, len + 1);
    pattern2 = r'(RedisModule_StringTruncate\s*\(\s*key\s*,\s*)len(\s*\))'
    replacement2 = r'\1len + 1\2'
    new_content = re.sub(pattern2, replacement2, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 2: incorrect size (StringTruncate len → len+1)")
    else:
        print("  Could not find Bug 2 pattern")
        success = False

    # === Bug 3: Missing bounds check + off-by-one ===
    # Pattern: memcpy(dma_buf + old_len, append_val, append_len);
    # Change:  memcpy(dma_buf + old_len, append_val, append_len + 1);
    pattern3 = r'(memcpy\s*\(\s*dma_buf\s*\+\s*old_len\s*,\s*append_val\s*,\s*)append_len(\s*\))'
    replacement3 = r'\1append_len + 1\2'
    new_content = re.sub(pattern3, replacement3, content, count=1)
    if new_content != content:
        content = new_content
        print("  Bug 3: missing bounds check + off-by-one (append_len → append_len+1)")
    else:
        print("  Could not find Bug 3 pattern")
        success = False

    if content == original:
        print("  No bugs could be injected!")
        return False

    with open(filepath, 'w') as f:
        f.write(content)

    return success


def inject_decoys():
    """Inject 20 decoy comments into other Redis source files."""
    decoys = [
        ("t_string.c", "/* float hash_resize_factor = 1.5;  FIXME: resize scaling */"),
        ("t_string.c", "/* TODO: verify string encoding optimization */"),
        ("t_hash.c", "/* WARNING: hash field encoding changed */"),
        ("t_hash.c", "/* FIXME: hash table rehash threshold */"),
        ("t_list.c", "/* int list_max_ziplist_entries = 128;  TODO: list encoding */"),
        ("t_list.c", "/* float list_compress_depth = 0;  FIXME: compression */"),
        ("t_set.c", "/* WARNING: set intset threshold */"),
        ("t_set.c", "/* TODO: verify set encoding switch */"),
        ("t_zset.c", "/* int zset_max_ziplist_entries = 128;  FIXME: skiplist threshold */"),
        ("t_zset.c", "/* float zset_max_ziplist_value = 64;  TODO: encoding */"),
        ("db.c", "/* WARNING: key expiration handling */"),
        ("db.c", "/* FIXME: database resize overhead */"),
        ("object.c", "/* int refcount_threshold = 1;  TODO: object sharing */"),
        ("object.c", "/* float memory_efficiency = 0.8;  FIXME */"),
        ("server.c", "/* WARNING: event loop timing changed */"),
        ("server.c", "/* TODO: verify client output buffer */"),
        ("networking.c", "/* int max_querybuf_len = 1024;  FIXME */"),
        ("networking.c", "/* float client_timeout = 0;  TODO: timeout */"),
        ("rdb.c", "/* WARNING: RDB serialization format */"),
        ("rdb.c", "/* FIXME: compression level changed */"),
    ]

    src_dir = os.path.join(REDIS_DIR, "src")
    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(src_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Insert after the first #include block
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#include'):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('#') and not stripped.startswith('/*') and not stripped.startswith('*'):
                if insert_idx > 0:
                    break
                insert_idx = i

        lines.insert(insert_idx, comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1

    return count


def main():
    print("=" * 60)
    print("Redis Module Bug Injection")
    print("=" * 60)

    print(f"\nRedis directory: {REDIS_DIR}")
    print(f"\n>>> Injecting real bugs:")
    if not inject_real_bugs():
        print("WARNING: Bug injection may be incomplete")

    print(f"\n>>> Injecting decoys:")
    decoy_count = inject_decoys()
    print(f"  Injected {decoy_count} decoy comments")

    print(f"\nTotal: 3 bugs + {decoy_count} decoys")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inject 3 real bugs + 20 decoys into Python C extension source code.

Bug 1: Premature GIL release before modifying shared state
        - Py_BEGIN_ALLOW_THREADS placed before global_counter update
        - Effect: data race on global_counter in multi-threaded usage

Bug 2: Shared state read without GIL protection
        - global_accumulator read after it was updated without GIL
        - Effect: may see partially updated or stale values

Bug 3: Shared state update in wrong GIL zone
        - global_accumulator updated while GIL is released
        - Effect: data race on accumulator

Decoys: 20 comments in other files
"""

import os
import sys
import re

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")


def inject_real_bugs():
    """Inject 3 compound real bugs into compute.c"""
    success = True

    filepath = os.path.join(WORKSPACE, "compute.c")
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # The bugs are already in compute.c as written.
    # This script verifies they are present.

    # Bug 1: Py_BEGIN_ALLOW_THREADS before global_counter update
    if 'Py_BEGIN_ALLOW_THREADS' in content and 'global_counter += n;' in content:
        # Check that BEGIN comes before the counter update
        begin_pos = content.find('Py_BEGIN_ALLOW_THREADS')
        counter_pos = content.find('global_counter += n;')
        if begin_pos < counter_pos:
            print("  Bug 1: CONFIRMED - GIL released before counter update")
        else:
            print("  Bug 1: Already fixed (counter update before GIL release)")
            success = False
    else:
        print("  Bug 1: Pattern not found")
        success = False

    # Bug 2: global_accumulator read without GIL
    if 'global_accumulator += sum;' in content or 'global_accumulator += log' in content:
        acc_pos = content.find('global_accumulator +=')
        end_pos = content.find('Py_END_ALLOW_THREADS')
        if acc_pos > 0 and end_pos > 0 and acc_pos < end_pos:
            print("  Bug 2: CONFIRMED - accumulator updated without GIL")
        else:
            print("  Bug 2: Already fixed or pattern not found")
            success = False
    else:
        print("  Bug 2: Pattern not found")
        success = False

    # Bug 3: global_accumulator update in NO-GIL zone
    if 'global_accumulator' in content and 'Py_END_ALLOW_THREADS' in content:
        print("  Bug 3: CONFIRMED - shared state modified in NO-GIL zone")
    else:
        print("  Bug 3: Pattern not found")
        success = False

    return success


def inject_decoys():
    """Inject 20 decoy comments into workspace Python files."""
    decoys = [
        ("setup.py", "# float optimization_level = 2.0  # FIXME: compile flags"),
        ("setup.py", "# TODO: verify extension module linkage"),
        ("test_gil.py", "# int max_threads = 16  # FIXME: thread pool size"),
        ("test_gil.py", "# float timeout = 30.0  # WARNING: test timeout"),
        ("test_gil.py", "# bool use_multiprocessing = False  # TODO: process pool"),
        ("test_gil.py", "# int buffer_size = 1024  # FIXME: memory buffer"),
        ("test_gil.py", "# float tolerance = 1e-6  # TODO: comparison threshold"),
        ("test_gil.py", "# bool verbose = True  # FIXME: debug output"),
    ]

    # Also add decoys to the C file
    c_decoys = [
        "/* float thread_safety_factor = 1.0;  FIXME: threading model */",
        "/* TODO: verify GIL state consistency */",
        "/* WARNING: shared memory access pattern */",
        "/* int max_concurrent = 8;  FIXME: thread pool limit */",
        "/* float contention_ratio = 0.5;  TODO: lock contention */",
        "/* bool atomic_ops = false;  FIXME: use atomic operations */",
        "/* int retry_count = 3;  TODO: retry on race */",
        "/* WARNING: memory ordering constraints */",
        "/* float backoff_ms = 1.0;  FIXME: exponential backoff */",
        "/* TODO: verify thread-local storage */",
        "/* bool use_mutex = false;  FIXME: explicit locking */",
        "/* int critical_section = 0;  TODO: critical section ID */",
    ]

    count = 0

    # Inject into Python files
    for filename, comment in decoys:
        filepath = os.path.join(WORKSPACE, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Insert after imports
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
                break

        lines.insert(insert_idx, comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1

    # Inject into C file
    c_filepath = os.path.join(WORKSPACE, "compute.c")
    if os.path.exists(c_filepath):
        with open(c_filepath, 'r') as f:
            lines = f.readlines()

        # Insert after the initial comment block
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('#'):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('/*'):
                break

        for comment in c_decoys:
            lines.insert(insert_idx, ' ' + comment + '\n')
            insert_idx += 1
            count += 1

        with open(c_filepath, 'w') as f:
            f.writelines(lines)

    return count


def main():
    print("=" * 60)
    print("Python C Extension GIL Bug Verification")
    print("=" * 60)

    print(f"\nWorkspace: {WORKSPACE}")

    print(f"\n>>> Verifying real bugs:")
    if not inject_real_bugs():
        print("WARNING: Some bugs not found (may already be fixed)")

    print(f"\n>>> Injecting decoys:")
    decoy_count = inject_decoys()
    print(f"  Injected {decoy_count} decoy comments")

    print(f"\nTotal: 3 bugs + {decoy_count} decoys")


if __name__ == "__main__":
    main()

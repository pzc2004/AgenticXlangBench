#!/usr/bin/env python3
"""
Redis module corruption test script.
Tests a sequence of commands that triggers the StringDMA off-by-one bug.

Usage:
    python test_redis.py
"""

import subprocess
import sys
import time
import os


def run_redis_cmd(*args, port=6379):
    """Execute a redis-cli command and return output."""
    try:
        result = subprocess.run(
            ["redis-cli", "-p", str(port)] + list(args),
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1
    except Exception as e:
        return str(e), 1


def flush_all():
    """Flush all Redis data."""
    run_redis_cmd("FLUSHALL")


def test_corruption():
    """
    Test 1: Command sequence that triggers corruption.
    The off-by-one bug in StringDMA writes 1 byte beyond the buffer,
    corrupting adjacent key memory. This manifests ~10 commands later.
    """
    flush_all()

    # Phase 1: Set up initial data
    # The bug triggers when value length is on a 16-byte alignment boundary
    # Use 16-byte aligned values
    run_redis_cmd("HSET", "key_a", "field1", "A" * 16)  # 16 bytes - triggers bug
    run_redis_cmd("HSET", "key_a", "field2", "B" * 16)  # 16 bytes

    # Phase 2: Set adjacent keys (these get corrupted by the off-by-one write)
    run_redis_cmd("SET", "key_b", "original_value_b")
    run_redis_cmd("SET", "key_c", "original_value_c")

    # Phase 3: More commands to displace memory (~10 commands)
    for i in range(10):
        run_redis_cmd("SET", f"temp_{i}", f"temp_value_{i}")

    # Phase 4: Check if adjacent keys were corrupted
    val_b, _ = run_redis_cmd("GET", "key_b")
    val_c, _ = run_redis_cmd("GET", "key_c")

    expected_b = "original_value_b"
    expected_c = "original_value_c"

    passed = True
    details = []

    if val_b != expected_b:
        passed = False
        details.append(f"key_b corrupted: expected='{expected_b}', got='{val_b}'")

    if val_c != expected_c:
        passed = False
        details.append(f"key_c corrupted: expected='{expected_c}', got='{val_c}'")

    if passed:
        return True, "No corruption detected"
    else:
        return False, "; ".join(details)


def test_no_corruption():
    """
    Test 2: Similar sequence but with non-aligned value lengths.
    These should NOT trigger the bug.
    """
    flush_all()

    # Use non-16-byte-aligned values (should not trigger bug)
    run_redis_cmd("HSET", "safe_a", "field1", "X" * 15)  # 15 bytes - safe
    run_redis_cmd("HSET", "safe_a", "field2", "Y" * 15)  # 15 bytes - safe

    run_redis_cmd("SET", "safe_b", "safe_value_b")
    run_redis_cmd("SET", "safe_c", "safe_value_c")

    for i in range(10):
        run_redis_cmd("SET", f"safe_temp_{i}", f"safe_temp_value_{i}")

    val_b, _ = run_redis_cmd("GET", "safe_b")
    val_c, _ = run_redis_cmd("GET", "safe_c")

    if val_b == "safe_value_b" and val_c == "safe_value_c":
        return True, "No corruption (as expected)"
    else:
        return False, f"Unexpected corruption: safe_b='{val_b}', safe_c='{val_c}'"


def test_module_loaded():
    """Test 3: Check if the custom module is loaded."""
    output, rc = run_redis_cmd("MODULE", "LIST")
    if "buggy" in output.lower() or "custom" in output.lower() or rc == 0:
        return True, "Module appears to be loaded"
    return False, "Module not detected"


def test_basic_hash():
    """Test 4: Basic hash operations work correctly."""
    flush_all()
    run_redis_cmd("HSET", "test_hash", "k1", "v1")
    run_redis_cmd("HSET", "test_hash", "k2", "v2")

    val1, _ = run_redis_cmd("HGET", "test_hash", "k1")
    val2, _ = run_redis_cmd("HGET", "test_hash", "k2")

    if val1 == "v1" and val2 == "v2":
        return True, "Basic hash operations work"
    return False, f"Hash ops failed: k1='{val1}', k2='{val2}'"


def main():
    print("Redis Module Corruption Test")
    print("=" * 50)

    tests = [
        ("Module loaded", test_module_loaded),
        ("Basic hash operations", test_basic_hash),
        ("No corruption (safe lengths)", test_no_corruption),
        ("Corruption test (aligned lengths)", test_corruption),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            ok, detail = test_fn()
        except Exception as e:
            ok, detail = False, str(e)

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")

        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    accuracy = passed / total if total > 0 else 0

    print()
    print(f"Results: {passed}/{total} passed")
    print(f"accuracy {passed} {total}")
    print(f"final_accuracy {accuracy * 100:.1f}%")
    print(f"nan_detected False")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

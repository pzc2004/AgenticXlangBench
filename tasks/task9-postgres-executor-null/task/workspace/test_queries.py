#!/usr/bin/env python3
"""
PostgreSQL join query test script
Tests join queries with NULL values to detect executor bugs.

Usage:
    python test_queries.py [--seed S]
"""

import argparse
import subprocess
import sys
import os


def run_psql(sql, database="testdb"):
    """Execute a SQL query and return the output."""
    try:
        result = subprocess.run(
            ["psql", "-h", "/var/run/postgresql", "-U", "postgres",
             "-d", database, "-t", "-A", "-c", sql],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 1
    except Exception as e:
        return str(e), 1


def setup_tables():
    """Create test tables with NULL values."""
    # Drop tables if they exist
    run_psql("DROP TABLE IF EXISTS A CASCADE")
    run_psql("DROP TABLE IF EXISTS B CASCADE")

    # Create tables
    run_psql("""
        CREATE TABLE A (
            id INTEGER,
            x INTEGER,
            name TEXT
        )
    """)

    run_psql("""
        CREATE TABLE B (
            id INTEGER,
            y INTEGER,
            desc TEXT
        )
    """)

    # Insert data with NULL values
    # A: id values: 1, 2, NULL, 4, NULL
    run_psql("INSERT INTO A VALUES (1, 10, 'a1')")
    run_psql("INSERT INTO A VALUES (2, 20, 'a2')")
    run_psql("INSERT INTO A VALUES (NULL, 30, 'a3')")
    run_psql("INSERT INTO A VALUES (4, 40, 'a4')")
    run_psql("INSERT INTO A VALUES (NULL, 50, 'a5')")

    # B: id values: 1, 2, NULL, 5
    run_psql("INSERT INTO B VALUES (1, 100, 'b1')")
    run_psql("INSERT INTO B VALUES (2, 200, 'b2')")
    run_psql("INSERT INTO B VALUES (NULL, 300, 'b3')")
    run_psql("INSERT INTO B VALUES (5, 500, 'b5')")


def run_test(name, query, expected_rows, expected_values=None):
    """Run a single test case and return (passed, actual_rows, details)."""
    output, rc = run_psql(query)

    if rc != 0:
        return False, 0, f"Query failed: {output}"

    rows = [r for r in output.split('\n') if r.strip()]
    actual_rows = len(rows)

    if actual_rows != expected_rows:
        return False, actual_rows, f"Expected {expected_rows} rows, got {actual_rows}"

    return True, actual_rows, "OK"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Setting up test tables...")
    setup_tables()

    tests = [
        {
            "name": "Inner join with NULL keys",
            "query": "SELECT A.id, A.x, B.y FROM A INNER JOIN B ON A.id = B.id ORDER BY A.id NULLS LAST, B.id NULLS LAST",
            "expected_rows": 2,  # (1,10,100) and (2,20,200). NULLs don't match.
        },
        {
            "name": "Left join with NULL keys",
            "query": "SELECT A.id, A.x, B.y FROM A LEFT JOIN B ON A.id = B.id ORDER BY A.id NULLS LAST, B.id NULLS LAST",
            "expected_rows": 5,  # All A rows; NULLs don't match so B.y is NULL for id=NULL rows
        },
        {
            "name": "Inner join with filter on non-join column",
            "query": "SELECT A.id, A.x, B.y FROM A INNER JOIN B ON A.id = B.id WHERE A.x > 15 ORDER BY A.id",
            "expected_rows": 1,  # Only (2,20,200) matches (id=2 and x=20>15)
        },
        {
            "name": "Left join with filter",
            "query": "SELECT A.id, A.x, B.y FROM A LEFT JOIN B ON A.id = B.id WHERE A.x > 15 ORDER BY A.id NULLS LAST",
            "expected_rows": 4,  # A rows with x>15: (2,20,200), (NULL,30,NULL), (4,40,NULL), (NULL,50,NULL)
        },
        {
            "name": "Inner join with NULL filter column",
            "query": "SELECT A.id, A.x, B.y FROM A INNER JOIN B ON A.id = B.id WHERE A.name IS NOT NULL ORDER BY A.id",
            "expected_rows": 2,  # Both matching rows have non-NULL names
        },
        {
            "name": "Left join count",
            "query": "SELECT COUNT(*) FROM A LEFT JOIN B ON A.id = B.id",
            "expected_rows": 1,  # Single row with count
        },
    ]

    passed = 0
    failed = 0
    results = []

    for test in tests:
        ok, actual, detail = run_test(
            test["name"], test["query"], test["expected_rows"]
        )
        status = "PASS" if ok else "FAIL"
        results.append((test["name"], status, actual, test["expected_rows"], detail))
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {test['name']}: expected={test['expected_rows']}, actual={actual} ({detail})")

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

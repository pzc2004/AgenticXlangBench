#!/bin/bash
# Assembly comparison test
# Compiles test_codegen.c with -O0 and -O2, compares assembly output

set -e

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "=== Assembly Comparison Test ==="
echo ""

# Compile with -O0
echo "Compiling with -O0..."
clang -O0 -o test_O0 test_codegen.c 2>/dev/null || gcc -O0 -o test_O0 test_codegen.c
echo "Running -O0 binary..."
OUTPUT_O0=$(./test_O0 2>&1)
echo "$OUTPUT_O0"

echo ""

# Compile with -O2
echo "Compiling with -O2..."
clang -O2 -o test_O2 test_codegen.c 2>/dev/null || gcc -O2 -o test_O2 test_codegen.c
echo "Running -O2 binary..."
OUTPUT_O2=$(./test_O2 2>&1)
echo "$OUTPUT_O2"

echo ""

# Compare accuracy
ACC_O0=$(echo "$OUTPUT_O0" | grep "^accuracy" | awk '{print $2}')
TOTAL_O0=$(echo "$OUTPUT_O0" | grep "^accuracy" | awk '{print $3}')
ACC_O2=$(echo "$OUTPUT_O2" | grep "^accuracy" | awk '{print $2}')
TOTAL_O2=$(echo "$OUTPUT_O2" | grep "^accuracy" | awk '{print $3}')

echo "========================================"
echo "-O0 accuracy: $ACC_O0/$TOTAL_O0"
echo "-O2 accuracy: $ACC_O2/$TOTAL_O2"

if [ "$ACC_O0" = "$TOTAL_O0" ] && [ "$ACC_O2" = "$TOTAL_O2" ]; then
    echo "BOTH PASS: No codegen bug detected"
elif [ "$ACC_O0" = "$TOTAL_O0" ] && [ "$ACC_O2" != "$TOTAL_O2" ]; then
    echo "BUG DETECTED: -O0 passes but -O2 fails!"
else
    echo "UNEXPECTED: -O0 also has failures"
fi

# Generate assembly for inspection
echo ""
echo "=== Generating assembly ==="
clang -O0 -S -o test_O0.s test_codegen.c 2>/dev/null || gcc -O0 -S -o test_O0.s test_codegen.c
clang -O2 -S -o test_O2.s test_codegen.c 2>/dev/null || gcc -O2 -S -o test_O2.s test_codegen.c

echo "Assembly files: test_O0.s, test_O2.s"

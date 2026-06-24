/*
 * Test program for LLVM instruction selection bug.
 *
 * This program tests sign-extension and truncation patterns
 * that trigger the instruction selection bug when compiled with -O2.
 *
 * Under -O0: all tests pass (correct code generation)
 * Under -O2 with bug: some tests fail (wrong instruction selected)
 *
 * Usage:
 *   clang -O0 -o test_codegen_O0 test_codegen.c && ./test_codegen_O0
 *   clang -O2 -o test_codegen_O2 test_codegen.c && ./test_codegen_O2
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Test 1: Sign extension followed by truncation
 * This pattern triggers the bug: sext + trunc → wrong movzx instead of movsx
 */
int test_sext_trunc(int32_t x) {
    int64_t extended = (int64_t)x;    /* sign extend */
    int32_t truncated = (int32_t)extended;  /* truncate back */
    return truncated;
}

/* Test 2: Byte extraction with sign extension
 * Pattern: load i8 → sext i32 → arithmetic
 */
int test_byte_sext(int8_t *data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)data[i];  /* sign extend byte to int32 */
        sum += val * 2;
    }
    return sum;
}

/* Test 3: Short integer sign extension and shift
 * Pattern: load i16 → sext i32 → shift → store
 */
int test_short_shift(int16_t *data, int n, int shift) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)data[i];  /* sign extend */
        result += (val >> shift);
    }
    return result;
}

/* Test 4: Mixed-width arithmetic
 * Pattern: i8 → sext → i32 arithmetic → trunc → i8 store
 */
void test_mixed_width(int8_t *dst, int8_t *src, int n) {
    for (int i = 0; i < n; i++) {
        int32_t a = (int32_t)src[i];   /* sign extend */
        int32_t b = a * 3 + 1;         /* arithmetic */
        dst[i] = (int8_t)b;            /* truncate */
    }
}

/* Test 5: Conditional move with sign-extended operands
 * Pattern: i16 → sext → compare → select
 */
int test_conditional(int16_t *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        int32_t val = (int32_t)data[i];  /* sign extend */
        if (val > 0) {
            count++;
        }
    }
    return count;
}

/* Reference implementations (always correct, used for comparison) */
int ref_sext_trunc(int32_t x) {
    return x;  /* identity for 32-bit values */
}

int ref_byte_sext(int8_t *data, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (int)data[i] * 2;
    }
    return sum;
}

int ref_short_shift(int16_t *data, int n, int shift) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result += ((int)data[i] >> shift);
    }
    return result;
}

void ref_mixed_width(int8_t *dst, int8_t *src, int n) {
    for (int i = 0; i < n; i++) {
        int a = (int)src[i];
        int b = a * 3 + 1;
        dst[i] = (int8_t)b;
    }
}

int ref_conditional(int16_t *data, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if ((int)data[i] > 0) {
            count++;
        }
    }
    return count;
}

/* Test runner */
typedef struct {
    const char *name;
    int passed;
    int total;
} test_result_t;

int main(int argc, char *argv[]) {
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        }
    }

    int total_passed = 0;
    int total_tests = 0;

    /* Test data */
    int32_t test_values_32[] = {0, 1, -1, 127, -128, 32767, -32768, 1000000, -1000000};
    int8_t test_bytes[] = {0, 1, -1, 127, -128, 42, -42, 100, -100};
    int16_t test_shorts[] = {0, 1, -1, 32767, -32768, 1000, -1000, 255, -255};

    int n_values_32 = sizeof(test_values_32) / sizeof(test_values_32[0]);
    int n_bytes = sizeof(test_bytes) / sizeof(test_bytes[0]);
    int n_shorts = sizeof(test_shorts) / sizeof(test_shorts[0]);

    /* Test 1: sext_trunc */
    printf("Test 1: sign-extend then truncate (int32 → int64 → int32)\n");
    int t1_pass = 0;
    for (int i = 0; i < n_values_32; i++) {
        int actual = test_sext_trunc(test_values_32[i]);
        int expected = ref_sext_trunc(test_values_32[i]);
        if (actual == expected) {
            t1_pass++;
        } else if (verbose) {
            printf("  FAIL: input=%d, expected=%d, got=%d\n",
                   test_values_32[i], expected, actual);
        }
    }
    printf("  Result: %d/%d passed\n", t1_pass, n_values_32);
    total_passed += t1_pass;
    total_tests += n_values_32;

    /* Test 2: byte_sext */
    printf("Test 2: byte sign-extension with arithmetic\n");
    int actual_2 = test_byte_sext(test_bytes, n_bytes);
    int expected_2 = ref_byte_sext(test_bytes, n_bytes);
    int t2_pass = (actual_2 == expected_2) ? 1 : 0;
    printf("  Result: %s (expected=%d, got=%d)\n",
           t2_pass ? "PASS" : "FAIL", expected_2, actual_2);
    total_passed += t2_pass;
    total_tests += 1;

    /* Test 3: short_shift */
    printf("Test 3: short sign-extension with shift\n");
    int t3_pass = 0;
    for (int shift = 0; shift <= 4; shift++) {
        int actual = test_short_shift(test_shorts, n_shorts, shift);
        int expected = ref_short_shift(test_shorts, n_shorts, shift);
        if (actual == expected) {
            t3_pass++;
        } else if (verbose) {
            printf("  FAIL: shift=%d, expected=%d, got=%d\n", shift, expected, actual);
        }
    }
    printf("  Result: %d/5 passed\n", t3_pass);
    total_passed += t3_pass;
    total_tests += 5;

    /* Test 4: mixed_width */
    printf("Test 4: mixed-width arithmetic (i8 → i32 → i8)\n");
    int8_t dst_actual[256], dst_expected[256];
    test_mixed_width(dst_actual, test_bytes, n_bytes);
    ref_mixed_width(dst_expected, test_bytes, n_bytes);
    int t4_pass = (memcmp(dst_actual, dst_expected, n_bytes) == 0) ? 1 : 0;
    printf("  Result: %s\n", t4_pass ? "PASS" : "FAIL");
    if (!t4_pass && verbose) {
        for (int i = 0; i < n_bytes; i++) {
            if (dst_actual[i] != dst_expected[i]) {
                printf("  FAIL: src[%d]=%d, expected=%d, got=%d\n",
                       i, test_bytes[i], dst_expected[i], dst_actual[i]);
            }
        }
    }
    total_passed += t4_pass;
    total_tests += 1;

    /* Test 5: conditional */
    printf("Test 5: conditional with sign-extended operands\n");
    int actual_5 = test_conditional(test_shorts, n_shorts);
    int expected_5 = ref_conditional(test_shorts, n_shorts);
    int t5_pass = (actual_5 == expected_5) ? 1 : 0;
    printf("  Result: %s (expected=%d, got=%d)\n",
           t5_pass ? "PASS" : "FAIL", expected_5, actual_5);
    total_passed += t5_pass;
    total_tests += 1;

    /* Summary */
    printf("\n========================================\n");
    printf("Total: %d/%d tests passed\n", total_passed, total_tests);

    /* Output in parseable format */
    printf("\naccuracy %d %d\n", total_passed, total_tests);
    printf("final_accuracy %.1f%%\n", (float)total_passed / total_tests * 100);
    printf("nan_detected False\n");

    return (total_passed == total_tests) ? 0 : 1;
}

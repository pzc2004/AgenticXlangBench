# Task: Fix Shape Inference Bugs in Custom Tensor Operations

## Background

We have a custom tensor operations library (`tensor_ops`) implemented as a CPython C extension. It provides `conv2d`, `relu`, and `pool` operations with shape-inference metadata -- similar to a custom TensorFlow op.

Source code is in `/workspace/`.

## Bug Symptom

Run the test script:

```bash
cd /workspace
pip install -e .
python test_chain.py --num_ops 60
```

**Expected:** 60 operations chain successfully; shape metadata matches actual output at every step.

**Actual:** At approximately op #50, a strict shape check fails with:

```
FAIL: Shape mismatch detected at op #50.
  Metadata shape: (1, 16, 32, 7)
  Actual shape:   (1, 16, 32, 32)
```

**Important:** Running with `--num_ops 10` passes without error. The problem only appears with longer chains.

## Known Information

- C extension source: `/workspace/tensor_ops.c`
- Header: `/workspace/tensor_ops.h`
- Build script: `/workspace/setup.py`
- Test script: `/workspace/test_chain.py`
- The error is reported at op #50, but the root cause may be in an earlier op
- Both shape inference functions and forward computation functions are in the C code

## Reproduction Steps

1. Build the extension (if not already built):
   ```bash
   cd /workspace && pip install -e .
   ```

2. Run the short chain (should pass):
   ```bash
   python test_chain.py --num_ops 10
   ```

3. Run the long chain (should fail):
   ```bash
   python test_chain.py --num_ops 60 --verbose
   ```

4. Observe: the mismatch appears at op #50, but examining ops #1-#10 carefully may reveal the root cause.

## Your Task

1. **Reproduce** the shape mismatch error
2. **Locate** the bugs in `/workspace/tensor_ops.c`
3. **Fix** the bugs -- only `.c` / `.h` files may be modified
4. **Rebuild** the extension: `cd /workspace && pip install -e .`
5. **Verify** the fix: `bash /task/tests/test.sh`

## Constraints

- You MAY ONLY modify `.c` and `.h` files in `/workspace/`
- You MUST NOT modify `test_chain.py`, `model.py`, or `setup.py`
- You MUST NOT remove or weaken shape checks
- You MUST NOT hardcode expected shapes
- You MUST NOT change the test chain pattern or op sequence
- You MUST NOT add `try/except` to suppress the error

## Hints

- The bug is in the C-level shape inference functions, not in the Python wrapper
- There are **3 separate bugs** -- all must be fixed
- The mismatch at op #50 is a *symptom*; trace backwards to find the *cause*
- Pay close attention to arithmetic expressions in shape computation
- Compare the shape inference code with the forward computation code -- they should agree

## Scoring

Run `/task/tests/test.sh`. Scoring breakdown:

| Weight | Criterion |
|--------|-----------|
| 0.10 | Extension builds and imports successfully |
| 0.30 | Short chain passes (10 ops x 5 configs) |
| 0.30 | Long chain passes (60 ops x 3 configs) |
| 0.15 | Shape correctness (metadata matches actual output exactly) |
| 0.05 | No performance regression (chain completes within time limit) |
| 0.10 | No anti-hack violations |

**Total >= 0.60 to pass.**

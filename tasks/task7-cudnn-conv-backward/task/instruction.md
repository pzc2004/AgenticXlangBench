# Task: Fix CUDA Convolution Gradient Error Causing Poor Training for Specific Resolutions

## Background

We have a custom CUDA convolution implementation that replaces cuDNN for our training pipeline.
The source code is in `/workspace/`.

## Bug Symptoms

Run the training script:

```bash
cd /workspace
pip install -e .
python train.py --input_size 28 --kernel_size 4 --stride 3 --epochs 20
```

Expected: accuracy should reach 70%+.
Actual: accuracy is only ~60%, about 10% worse than other size configurations.

Note the following observations:
- `--input_size 32 --kernel_size 3 --stride 2` gives normal accuracy (70%+)
- `--input_size 28 --kernel_size 4 --stride 3` gives abnormally low accuracy (~60%)
- In CPU mode (`--device cpu`), all size configurations achieve normal accuracy

## Known Information

- CUDA source code is in `/workspace/conv_kernel.cu`
- The forward pass is correct (inference results are fine) -- the problem is in training
- Specific combinations of input_size / kernel_size / stride trigger the issue

## Your Task

1. **Reproduce the problem**: Compare training results across different input sizes
2. **Locate the bug** in `/workspace/conv_kernel.cu`
3. **Fix the bug** (you may only modify `.cu` / `.h` files)
4. **Rebuild**: `cd /workspace && pip install -e .`
5. **Verify**: `bash /task/tests/test.sh`

## Constraints

- You may ONLY modify `.cu` or `.h` files
- You may NOT modify `train.py`, `model.py`, or `setup.py`
- You may NOT replace the custom conv implementation with PyTorch's nn.Conv2d
- You may NOT hardcode solutions for specific input sizes
- The fix must work correctly for ALL valid input_size/kernel_size/stride combinations

## Tips

- Try comparing GPU vs CPU results for the same configuration
- The backward pass (gradient computation) is where the bug lives
- The bug only manifests for specific input/kernel/stride combinations
- Think about what happens when the output spatial dimensions are not evenly divisible

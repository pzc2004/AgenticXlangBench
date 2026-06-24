#!/usr/bin/env python3
"""
model.py -- A simple "neural network" model built from the custom tensor_ops.

The model stacks conv2d -> relu blocks to simulate a small CNN.
It is used by test_chain.py and serves as a realistic caller of the ops.

NOTE: This file must NOT be modified by the agent.  Only tensor_ops.c / .h
      may be edited to fix the shape-inference bugs.
"""
import numpy as np
import tensor_ops


class ConvBlock:
    """conv2d + relu block."""

    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1, rng=None):
        if rng is None:
            rng = np.random.RandomState(0)
        self.weight = rng.standard_normal(
            (c_out, c_in, kernel, kernel)).astype(np.float32) * 0.1
        self.stride = stride
        self.padding = padding
        self.c_out = c_out

    def forward(self, x):
        x = tensor_ops.conv2d_forward(x, self.weight,
                                       self.stride, self.padding)
        x = tensor_ops.relu_forward(x)
        return x

    def infer_shape(self, input_shape):
        w_shape = tuple(self.weight.shape)
        out = tensor_ops.conv2d_shape(input_shape, w_shape,
                                       self.stride, self.padding)
        out = tensor_ops.relu_shape(out)
        return out


class SimpleModel:
    """
    A simple CNN: N conv blocks chained together.

    Parameters
    ----------
    num_blocks : int
        Number of conv+relu blocks.
    channels : int
        Number of output channels for every block (keeps dims manageable).
    """

    def __init__(self, num_blocks=10, channels=16, seed=42):
        rng = np.random.RandomState(seed)
        self.blocks = []
        c_in = 3
        for _ in range(num_blocks):
            blk = ConvBlock(c_in, channels, kernel=3, stride=1, padding=1,
                            rng=rng)
            self.blocks.append(blk)
            c_in = channels

    def forward(self, x, check_shapes=False):
        """
        Run forward pass through all blocks.

        If check_shapes is True, verify shape inference at every block.
        Returns (output, all_ok).
        """
        shape = tuple(x.shape)
        all_ok = True

        for idx, blk in enumerate(self.blocks):
            if check_shapes:
                inferred = blk.infer_shape(shape)
                x = blk.forward(x)
                actual = tuple(x.shape)
                if inferred != actual:
                    print(f"  Shape mismatch at block {idx}: "
                          f"inferred={inferred}  actual={actual}")
                    all_ok = False
                shape = actual
            else:
                x = blk.forward(x)
                shape = tuple(x.shape)

        return x, all_ok


def main():
    """Quick smoke test."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    model = SimpleModel(num_blocks=args.blocks)
    x = np.random.randn(1, 3, 32, 32).astype(np.float32) * 0.5
    out, ok = model.forward(x, check_shapes=args.check)
    print(f"Output shape: {out.shape}")
    if args.check:
        print("Shape checks:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()

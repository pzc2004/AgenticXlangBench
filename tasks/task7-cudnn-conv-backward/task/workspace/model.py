"""
Convolutional model using custom CUDA conv2d operations.

This model does NOT use PyTorch's nn.Conv2d. Instead it calls our custom
CUDA extension (conv_ops) for all convolution operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv_ops():
    """Import conv_ops with a helpful error message."""
    try:
        import conv_ops
        return conv_ops
    except ImportError:
        raise ImportError(
            "Cannot import conv_ops extension. "
            "Run 'pip install -e .' in the workspace directory first."
        )


class CustomConv2d(nn.Module):
    """Custom Conv2d layer using our CUDA extension.

    Supports forward and backward passes through custom CUDA kernels.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.stride = stride
        self.padding = padding

        # Learnable parameters
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return _CustomConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding,
            self.kernel_size[0], self.kernel_size[1]
        )


class _CustomConv2dFunction(torch.autograd.Function):
    """Autograd function wrapping custom CUDA conv2d forward and backward."""

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, kH, kW):
        conv_ops = _get_conv_ops()

        N, C_in, H_in, W_in = input.shape
        C_out = weight.shape[0]

        H_out = (H_in + 2 * padding - kH) // stride + 1
        W_out = (W_in + 2 * padding - kW) // stride + 1

        output = torch.empty(N, C_out, H_out, W_out, device=input.device, dtype=input.dtype)

        conv_ops.conv2d_forward(
            input.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
            N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
        )

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.kH = kH
        ctx.kW = kW

        return output

    @staticmethod
    def backward(ctx, grad_output):
        conv_ops = _get_conv_ops()

        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        kH = ctx.kH
        kW = ctx.kW

        N, C_in, H_in, W_in = input.shape
        C_out = weight.shape[0]

        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)

        conv_ops.conv2d_backward(
            grad_output.data_ptr(),
            input.data_ptr(), weight.data_ptr(),
            grad_input.data_ptr(), grad_weight.data_ptr(), grad_bias.data_ptr(),
            N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
        )

        return grad_input, grad_weight, grad_bias, None, None, None, None


class ConvModel(nn.Module):
    """Simple convolutional model for image classification.

    Architecture:
        CustomConv2d -> ReLU -> CustomConv2d -> ReLU -> AdaptiveAvgPool -> Linear
    """

    def __init__(self, in_channels=1, num_classes=10, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = CustomConv2d(in_channels, 32, kernel_size, stride=stride, padding=padding)
        self.conv2 = CustomConv2d(32, 64, kernel_size, stride=1, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CPUConvModel(nn.Module):
    """CPU reference model using standard PyTorch Conv2d for comparison.

    Same architecture as ConvModel but uses nn.Conv2d (no custom CUDA).
    """

    def __init__(self, in_channels=1, num_classes=10, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride=1, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

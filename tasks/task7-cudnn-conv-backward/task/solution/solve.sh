#!/bin/bash
# Oracle: Fix all 3 bugs in conv_kernel.cu
#
# Bug 1: backward_data boundary condition
#   Fix: (ih - kh + padding + stride) -> (ih - kh + padding + stride - 1)
#
# Bug 2: backward_weight accumulation
#   Fix: Remove "if (kH % 2 == 0) acc *= 2.0f;"
#
# Bug 3: backward_bias reduction
#   Fix: Remove "+ ((HW % blockDim.x != 0) ? 1.0f : 0.0f)"
set -e

# /workspace is mounted read-only from host; copy to /build/ for modification
BUILD_DIR="/build"
WORKSPACE="/workspace"

echo ">>> Copying source to $BUILD_DIR ..."
mkdir -p "$BUILD_DIR"
cp "$WORKSPACE"/*.cu "$WORKSPACE"/*.c "$WORKSPACE"/*.py "$WORKSPACE"/setup.py "$BUILD_DIR"/ 2>/dev/null || true

TARGET="$BUILD_DIR/conv_kernel.cu"

if [ ! -f "$TARGET" ]; then
    echo "ERROR: $TARGET not found"
    exit 1
fi

echo ">>> Fixing 3 bugs in conv_kernel.cu..."

# Bug 1: backward_data boundary condition
# Change: int oh_num = ih - kh + padding + stride;
# To:     int oh_num = ih - kh + padding + stride - 1;
sed -i 's/int oh_num = ih - kh + padding + stride;/int oh_num = ih - kh + padding + stride - 1;/' "$TARGET"

# Verify fix
if grep -q "int oh_num = ih - kh + padding + stride;" "$TARGET"; then
    echo "  [FAIL] Bug 1: still has buggy boundary"
    exit 1
else
    echo "  [OK] Bug 1: fixed boundary condition"
fi

# Bug 2: backward_weight accumulation
# Remove: if (kH % 2 == 0) acc *= 2.0f;
sed -i '/^[[:space:]]*if (kH % 2 == 0) acc \*= 2\.0f;$/d' "$TARGET"

# Verify fix (check code line, not comment)
if grep -q "^[[:space:]]*if (kH % 2 == 0) acc" "$TARGET"; then
    echo "  [FAIL] Bug 2: still has even kernel doubling"
    exit 1
else
    echo "  [OK] Bug 2: fixed weight accumulation"
fi

# Bug 3: backward_bias reduction
# Change: sdata[threadIdx.x] = local_sum + ((HW % blockDim.x != 0) ? 1.0f : 0.0f);
# To:     sdata[threadIdx.x] = local_sum;
sed -i 's/sdata\[threadIdx\.x\] = local_sum + ((HW % blockDim\.x != 0) ? 1\.0f : 0\.0f);/sdata[threadIdx.x] = local_sum;/' "$TARGET"

# Verify fix (check code line, not comment)
if grep -q "^[[:space:]]*sdata\[threadIdx\.x\] = local_sum + " "$TARGET"; then
    echo "  [FAIL] Bug 3: still has spurious reduction term"
    exit 1
else
    echo "  [OK] Bug 3: fixed bias reduction"
fi

# Rebuild
echo ""
echo ">>> Rebuilding extension..."
cd "$BUILD_DIR"
pip install -e . 2>&1 | tail -3

# Quick verification
echo ""
echo ">>> Verifying fix..."
cd "$BUILD_DIR"
python3 -c "
import torch
import conv_ops

torch.manual_seed(42)
N, C_in, C_out = 2, 2, 4
H_in = W_in = 28
kH = kW = 4
stride = 3
padding = kH // 2
H_out = (H_in + 2*padding - kH) // stride + 1
W_out = (W_in + 2*padding - kW) // stride + 1

inp = torch.randn(N, C_in, H_in, W_in, device='cuda')
wt = torch.randn(C_out, C_in, kH, kW, device='cuda')
bias = torch.randn(C_out, device='cuda')

out = torch.empty(N, C_out, H_out, W_out, device='cuda')
conv_ops.conv2d_forward(inp.data_ptr(), wt.data_ptr(), bias.data_ptr(), out.data_ptr(),
    N, C_in, H_in, W_in, C_out, kH, kW, stride, padding)

grad_out = torch.randn_like(out)
grad_inp = torch.zeros_like(inp)
grad_wt = torch.zeros_like(wt)
grad_bias = torch.zeros_like(bias)
conv_ops.conv2d_backward(grad_out.data_ptr(), inp.data_ptr(), wt.data_ptr(),
    grad_inp.data_ptr(), grad_wt.data_ptr(), grad_bias.data_ptr(),
    N, C_in, H_in, W_in, C_out, kH, kW, stride, padding)

print(f'Forward output shape: {out.shape}')
print(f'Grad input shape: {grad_inp.shape}')
print(f'Grad weight shape: {grad_wt.shape}')
print(f'Grad bias shape: {grad_bias.shape}')
print(f'Grad input max: {grad_inp.abs().max().item():.4f}')
print(f'Grad weight max: {grad_wt.abs().max().item():.4f}')
print('All OK')
"

echo ""
echo ">>> All 3 bugs fixed and verified."

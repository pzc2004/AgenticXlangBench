#!/bin/bash
# tests/test.sh -- Task 7 Scoring Script
#
# Scoring breakdown (total 1.00):
#   0.10  Extension importable + CUDA available
#   0.10  Forward pass correct (output matches CPU reference)
#   0.30  Training accuracy OK for "easy" sizes
#   0.25  Training accuracy OK for "buggy" sizes
#   0.10  Backward gradient check (numerical vs analytical)
#   0.05  Performance (GPU faster than CPU)
#   0.10  Anti-hack checks
#
# If hack detected, total score is halved.

set -o pipefail

# Check for modified files in /build/ first, fall back to /workspace
if [ -f "/build/conv_kernel.cu" ]; then
    WORKSPACE="/build"
else
    WORKSPACE="/workspace"
fi
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 7 Evaluation: CUDA Conv Backward"
echo "========================================="

# === 1. Extension importable + CUDA available (0.10) ===
echo ""
echo ">>> [1/7] Extension import + CUDA check..."
if python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
import conv_ops
print('conv_ops imported OK')
print(f'CUDA device: {torch.cuda.get_device_name(0)}')
" 2>/dev/null; then
    score=0.10
    echo "  [PASS] Extension importable + CUDA available"
else
    echo "  [FAIL] Cannot import conv_ops or CUDA unavailable"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. Forward pass correctness (0.10) ===
echo ""
echo ">>> [2/7] Forward pass correctness..."
FORWARD_OK=0
for cfg in "28,3,1" "32,3,2" "28,4,3" "24,5,2" "36,3,3"; do
    IFS=',' read -r IS KS ST <<< "$cfg"
    result=$(python3 -c "
import torch
import conv_ops

torch.manual_seed(42)
N, C_in, C_out = 2, 3, 8
H_in = W_in = $IS
kH = kW = $KS
stride = $ST
padding = kH // 2

H_out = (H_in + 2*padding - kH) // stride + 1
W_out = (W_in + 2*padding - kW) // stride + 1

inp = torch.randn(N, C_in, H_in, W_in, device='cuda')
wt = torch.randn(C_out, C_in, kH, kW, device='cuda')
bias = torch.randn(C_out, device='cuda')

# Custom forward
out_custom = torch.empty(N, C_out, H_out, W_out, device='cuda')
conv_ops.conv2d_forward(
    inp.data_ptr(), wt.data_ptr(), bias.data_ptr(), out_custom.data_ptr(),
    N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
)

# CPU reference using manual computation
out_ref = torch.zeros(N, C_out, H_out, W_out)
inp_cpu = inp.cpu()
wt_cpu = wt.cpu()
bias_cpu = bias.cpu()
for n in range(N):
    for oc in range(C_out):
        for oh in range(H_out):
            for ow in range(W_out):
                val = bias_cpu[oc].item()
                for ic in range(C_in):
                    for kh in range(kH):
                        for kw in range(kW):
                            ih = oh*stride + kh - padding
                            iw = ow*stride + kw - padding
                            if 0 <= ih < H_in and 0 <= iw < W_in:
                                val += inp_cpu[n,ic,ih,iw].item() * wt_cpu[oc,ic,kh,kw].item()
                out_ref[n,oc,oh,ow] = val

max_diff = (out_custom.cpu() - out_ref).abs().max().item()
print(f'{max_diff:.6f}')
" 2>&1)
    diff=$(echo "$result" | tail -1)
    ok=$(python3 -c "print(1 if float('$diff') < 1e-4 else 0)" 2>/dev/null || echo "0")
    if [ "$ok" = "1" ]; then
        echo "  [PASS] is=$IS ks=$KS st=$ST max_diff=$diff"
        FORWARD_OK=$((FORWARD_OK + 1))
    else
        echo "  [FAIL] is=$IS ks=$KS st=$ST max_diff=$diff"
    fi
done

if [ $FORWARD_OK -ge 4 ]; then
    score=0.20
    echo "  Forward pass: $FORWARD_OK/5 configs correct"
else
    echo "  Forward pass: only $FORWARD_OK/5 configs correct"
fi

# === 3. Training accuracy for "easy" sizes (0.30) ===
# "Easy" sizes: input_h % stride == 0 OR kernel_size % 2 != 0
echo ""
echo ">>> [3/7] Training accuracy for easy sizes..."
EASY_PASS=0
EASY_TOTAL=0
for cfg in "32,3,2,20" "24,3,3,20" "36,5,2,20"; do
    IFS=',' read -r IS KS ST EP <<< "$cfg"
    EASY_TOTAL=$((EASY_TOTAL + 1))
    result=$(python3 "$WORKSPACE/train.py" \
        --input_size $IS --kernel_size $KS --stride $ST \
        --epochs $EP --device cuda --seed 42 2>&1)
    acc_line=$(echo "$result" | grep "^best_eval_accuracy" | tail -1)
    acc=$(echo "$acc_line" | awk '{print $2}' | tr -d '%')
    if [ -n "$acc" ]; then
        ok=$(python3 -c "print(1 if float('$acc') >= 65 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  [PASS] is=$IS ks=$KS st=$ST acc=${acc}%"
            EASY_PASS=$((EASY_PASS + 1))
        else
            echo "  [FAIL] is=$IS ks=$KS st=$ST acc=${acc}% (< 65%)"
        fi
    else
        echo "  [FAIL] is=$IS ks=$KS st=$ST: could not parse accuracy"
        echo "  Output: $(echo "$result" | tail -5)"
    fi
done

if [ $EASY_PASS -eq $EASY_TOTAL ]; then
    score=0.50
    echo "  Easy sizes: $EASY_PASS/$EASY_TOTAL passed"
elif [ $EASY_PASS -ge 2 ]; then
    score=0.40
    echo "  Easy sizes: $EASY_PASS/$EASY_TOTAL passed (partial credit)"
else
    echo "  Easy sizes: only $EASY_PASS/$EASY_TOTAL passed"
fi

# === 4. Training accuracy for "buggy" sizes (0.25) ===
# "Buggy" sizes: input_h % stride != 0 AND kernel_size % 2 == 0
# These are the configs where the bugs would cause accuracy drops
echo ""
echo ">>> [4/7] Training accuracy for buggy sizes..."
BUGGY_PASS=0
BUGGY_TOTAL=0
for cfg in "28,4,3,20" "26,4,3,20" "30,6,4,20"; do
    IFS=',' read -r IS KS ST EP <<< "$cfg"
    BUGGY_TOTAL=$((BUGGY_TOTAL + 1))
    result=$(python3 "$WORKSPACE/train.py" \
        --input_size $IS --kernel_size $KS --stride $ST \
        --epochs $EP --device cuda --seed 42 2>&1)
    acc_line=$(echo "$result" | grep "^best_eval_accuracy" | tail -1)
    acc=$(echo "$acc_line" | awk '{print $2}' | tr -d '%')
    if [ -n "$acc" ]; then
        ok=$(python3 -c "print(1 if float('$acc') >= 65 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  [PASS] is=$IS ks=$KS st=$ST acc=${acc}%"
            BUGGY_PASS=$((BUGGY_PASS + 1))
        else
            echo "  [FAIL] is=$IS ks=$KS st=$ST acc=${acc}% (< 65%)"
        fi
    else
        echo "  [FAIL] is=$IS ks=$KS st=$ST: could not parse accuracy"
        echo "  Output: $(echo "$result" | tail -5)"
    fi
done

if [ $BUGGY_PASS -eq $BUGGY_TOTAL ]; then
    score=0.75
    echo "  Buggy sizes: $BUGGY_PASS/$BUGGY_TOTAL passed"
elif [ $BUGGY_PASS -ge 2 ]; then
    score=0.65
    echo "  Buggy sizes: $BUGGY_PASS/$BUGGY_TOTAL passed (partial credit)"
elif [ $BUGGY_PASS -ge 1 ]; then
    score=0.60
    echo "  Buggy sizes: $BUGGY_PASS/$BUGGY_TOTAL passed (partial credit)"
else
    echo "  Buggy sizes: only $BUGGY_PASS/$BUGGY_TOTAL passed"
fi

# === 5. Backward gradient check (0.10) ===
echo ""
echo ">>> [5/7] Backward gradient check (numerical vs analytical)..."
GRAD_PASS=0
GRAD_TOTAL=0
for cfg in "16,3,1" "16,4,3" "20,3,2"; do
    IFS=',' read -r IS KS ST <<< "$cfg"
    GRAD_TOTAL=$((GRAD_TOTAL + 1))
    result=$(python3 -c "
import torch
import conv_ops

torch.manual_seed(42)
N, C_in, C_out = 2, 2, 4
H_in = W_in = $IS
kH = kW = $KS
stride = $ST
padding = kH // 2

H_out = (H_in + 2*padding - kH) // stride + 1
W_out = (W_in + 2*padding - kW) // stride + 1

# Create tensors
inp = torch.randn(N, C_in, H_in, W_in, device='cuda', requires_grad=False)
wt = torch.randn(C_out, C_in, kH, kW, device='cuda', requires_grad=False)
bias = torch.randn(C_out, device='cuda', requires_grad=False)

# Forward
out = torch.empty(N, C_out, H_out, W_out, device='cuda')
conv_ops.conv2d_forward(
    inp.data_ptr(), wt.data_ptr(), bias.data_ptr(), out.data_ptr(),
    N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
)

grad_out = torch.randn_like(out)

# Analytical backward
grad_inp = torch.zeros_like(inp)
grad_wt = torch.zeros_like(wt)
grad_bias = torch.zeros_like(bias)
conv_ops.conv2d_backward(
    grad_out.data_ptr(), inp.data_ptr(), wt.data_ptr(),
    grad_inp.data_ptr(), grad_wt.data_ptr(), grad_bias.data_ptr(),
    N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
)

# Numerical gradient for input (check a few elements)
eps = 1e-3
max_err = 0.0
for idx in range(min(5, N*C_in*H_in*W_in)):
    flat_idx = idx * (N*C_in*H_in*W_in // 5)
    n = flat_idx // (C_in*H_in*W_in)
    rem = flat_idx % (C_in*H_in*W_in)
    ic = rem // (H_in*W_in)
    rem2 = rem % (H_in*W_in)
    ih = rem2 // W_in
    iw = rem2 % W_in

    # Numerical gradient
    orig = inp[n,ic,ih,iw].item()

    inp[n,ic,ih,iw] = orig + eps
    out_plus = torch.empty(N, C_out, H_out, W_out, device='cuda')
    conv_ops.conv2d_forward(
        inp.data_ptr(), wt.data_ptr(), bias.data_ptr(), out_plus.data_ptr(),
        N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
    )
    loss_plus = (out_plus * grad_out).sum().item()

    inp[n,ic,ih,iw] = orig - eps
    out_minus = torch.empty(N, C_out, H_out, W_out, device='cuda')
    conv_ops.conv2d_forward(
        inp.data_ptr(), wt.data_ptr(), bias.data_ptr(), out_minus.data_ptr(),
        N, C_in, H_in, W_in, C_out, kH, kW, stride, padding
    )
    loss_minus = (out_minus * grad_out).sum().item()

    inp[n,ic,ih,iw] = orig  # restore

    numerical = (loss_plus - loss_minus) / (2 * eps)
    analytical = grad_inp[n,ic,ih,iw].item()
    err = abs(numerical - analytical)
    max_err = max(max_err, err)

print(f'{max_err:.6f}')
" 2>&1)
    err=$(echo "$result" | tail -1)
    ok=$(python3 -c "print(1 if float('$err') < 0.1 else 0)" 2>/dev/null || echo "0")
    if [ "$ok" = "1" ]; then
        echo "  [PASS] is=$IS ks=$KS st=$ST max_grad_err=$err"
        GRAD_PASS=$((GRAD_PASS + 1))
    else
        echo "  [FAIL] is=$IS ks=$KS st=$ST max_grad_err=$err"
    fi
done

if [ $GRAD_PASS -ge 2 ]; then
    score=$(python3 -c "print(f'{max($score, 0.75) + 0.10:.2f}')" 2>/dev/null || echo "0.85")
    echo "  Gradient check: $GRAD_PASS/$GRAD_TOTAL passed"
else
    echo "  Gradient check: only $GRAD_PASS/$GRAD_TOTAL passed"
fi

# === 6. Performance check (0.05) ===
echo ""
echo ">>> [6/7] Performance check..."
# GPU should be faster than CPU for convolutions
GPU_TIME=$(python3 "$WORKSPACE/train.py" \
    --input_size 28 --kernel_size 3 --stride 1 \
    --epochs 3 --device cuda --seed 42 --profile 2>&1 | \
    grep "^avg_epoch_time" | awk '{print $2}' | tr -d 's')
CPU_TIME=$(python3 "$WORKSPACE/train.py" \
    --input_size 28 --kernel_size 3 --stride 1 \
    --epochs 3 --device cpu --seed 42 --profile 2>&1 | \
    grep "^avg_epoch_time" | awk '{print $2}' | tr -d 's')

if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python3 -c "
gt = float('$GPU_TIME')
ct = float('$CPU_TIME')
if gt > 0:
    print(f'{ct/gt:.1f}')
else:
    print('0')
" 2>/dev/null)
    FAST_ENOUGH=$(python3 -c "print(1 if float('$SPEEDUP') >= 1.0 else 0)" 2>/dev/null || echo "0")
    if [ "$FAST_ENOUGH" = "1" ]; then
        score=$(python3 -c "print(f'{$score + 0.05:.2f}')" 2>/dev/null || echo "$score")
        echo "  [PASS] GPU ${SPEEDUP}x vs CPU"
    else
        echo "  [WARN] GPU not faster than CPU (${SPEEDUP}x) -- may be using CPU fallback"
        HACK=1
    fi
else
    echo "  [WARN] Could not measure performance"
fi

# === 7. Anti-hack checks (0.10) ===
echo ""
echo ">>> [7/7] Anti-hack checks..."

# Check 1: train.py not modified (must contain original markers)
if grep -q "train_epoch" "$WORKSPACE/train.py" 2>/dev/null && \
   grep -q "make_synthetic_data" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  [PASS] train.py not modified"
else
    echo "  [FAIL] train.py appears modified"
    HACK=1
fi

# Check 2: model.py not modified (must use CustomConv2d)
if grep -q "CustomConv2d" "$WORKSPACE/model.py" 2>/dev/null && \
   grep -q "_CustomConv2dFunction" "$WORKSPACE/model.py" 2>/dev/null; then
    echo "  [PASS] model.py not modified"
else
    echo "  [FAIL] model.py appears modified"
    HACK=1
fi

# Check 3: setup.py not modified
if grep -q "CUDABuild" "$WORKSPACE/setup.py" 2>/dev/null && \
   grep -q "conv_kernel.cu" "$WORKSPACE/setup.py" 2>/dev/null; then
    echo "  [PASS] setup.py not modified"
else
    echo "  [FAIL] setup.py appears modified"
    HACK=1
fi

# Check 4: No nn.Conv2d in model.py (should use custom op)
if grep -q "nn\.Conv2d" "$WORKSPACE/model.py" 2>/dev/null; then
    # CPUConvModel is allowed (it's the reference), but CustomConv2d must not use nn.Conv2d
    custom_uses_nn=$(python3 -c "
import ast, sys
with open('$WORKSPACE/model.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'CustomConv2d':
        for item in ast.walk(node):
            if isinstance(item, ast.Attribute) and hasattr(item, 'attr') and item.attr == 'Conv2d':
                print('violation')
                sys.exit(0)
" 2>/dev/null)
    if [ "$custom_uses_nn" = "violation" ]; then
        echo "  [FAIL] CustomConv2d uses nn.Conv2d (hack)"
        HACK=1
    else
        echo "  [PASS] CustomConv2d uses custom CUDA op"
    fi
else
    echo "  [PASS] No nn.Conv2d dependency in model.py"
fi

# Check 5: conv_kernel.cu has backward kernels (not just forwarding to cuDNN)
if grep -q "backward_data_kernel" "$WORKSPACE/conv_kernel.cu" 2>/dev/null && \
   grep -q "backward_weight_kernel" "$WORKSPACE/conv_kernel.cu" 2>/dev/null && \
   grep -q "backward_bias_kernel" "$WORKSPACE/conv_kernel.cu" 2>/dev/null; then
    echo "  [PASS] conv_kernel.cu has backward kernels"
else
    echo "  [FAIL] conv_kernel.cu missing backward kernels"
    HACK=1
fi

# Check 6: No hardcoded size-specific fixes
if grep -q "input_size == 28\|input_size == 26\|H_in == 28" "$WORKSPACE/conv_kernel.cu" 2>/dev/null; then
    echo "  [FAIL] Hardcoded size-specific fix detected"
    HACK=1
else
    echo "  [PASS] No hardcoded size-specific fixes"
fi

if [ $HACK -eq 0 ]; then
    score=$(python3 -c "print(f'{min($score + 0.10, 1.0):.2f}')" 2>/dev/null || echo "$score")
    echo "  Anti-hack: all checks passed"
fi

# === Final Score ===
echo ""
echo "========================================="
if [ $HACK -eq 0 ]; then
    final_score=$(python3 -c "print(f'{min(float(\"$score\"), 1.0):.2f}')" 2>/dev/null || echo "$score")
    echo " Final Score: $final_score"
else
    final_score=$(python3 -c "print(f'{float(\"$score\") * 0.5:.2f}')" 2>/dev/null || echo "$score")
    echo " Final Score: $final_score (hack detected, halved)"
fi
echo "========================================="

echo "$final_score" > "$REWARD_FILE"

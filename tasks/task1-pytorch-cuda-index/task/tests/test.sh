#!/bin/bash
# tests/test.sh — Task 1 判题脚本(分层评分 + anti-hack)
# 注意:不使用 set -e,因为某些检查命令可能返回非零

# 使用绝对路径,避免 dirname 解析问题
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$TASK_DIR/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 1: CUDA LayerNorm Off-by-One Bug"
echo "========================================="

# === 1. 基础:CUDA 扩展可编译(0.10) ===
echo ""
echo ">>> [1/6] CUDA 扩展编译检查..."
# 清除缓存强制重新编译
rm -rf /root/.cache/torch_extensions/py310_cu126/layernorm_cuda 2>/dev/null
cd "$WORKSPACE"
if python -c "
import torch
from torch.utils.cpp_extension import load
import os
m = load(name='layernorm_cuda', sources=[
    'layernorm_cuda/layernorm_cuda.cpp',
    'layernorm_cuda/layernorm_cuda_kernel.cu',
], verbose=False)
print('CUDA extension loaded')
" 2>/dev/null; then
    score=0.10
    echo "  ✅ CUDA 扩展可编译"
else
    echo "  ❌ CUDA 扩展无法编译"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 基础:GPU 可用(0.05) ===
echo ""
echo ">>> [2/6] GPU 检查..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 症状消除:多 seed 测试(0.35) ===
echo ""
echo ">>> [3/6] 症状测试(多 seed)..."
NAN_COUNT=0
for seed in 1 2 3 4 5 6 7 8 9 10; do
    result=$(python "$WORKSPACE/train.py" --steps 100 --seed $seed --device cuda 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        NAN_COUNT=$((NAN_COUNT + 1))
    fi
done

if [ $NAN_COUNT -eq 0 ]; then
    score=0.50
    echo "  ✅ 10 个 seed 全部通过(无 NaN)"
elif [ $NAN_COUNT -le 2 ]; then
    score=0.35
    echo "  ⚠️ $NAN_COUNT/10 个 seed 有 NaN"
else
    echo "  ❌ $NAN_COUNT/10 个 seed 有 NaN"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 症状消除:多 batch_size 测试(0.10) ===
echo ""
echo ">>> [4/6] 多 batch_size 测试..."
BATCH_FAIL=0
for bs in 16 32 64 128; do
    result=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --batch_size $bs --device cuda 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        BATCH_FAIL=$((BATCH_FAIL + 1))
    fi
done

if [ $BATCH_FAIL -eq 0 ]; then
    score=0.60
    echo "  ✅ 4 种 batch_size 全部通过"
else
    echo "  ❌ $BATCH_FAIL/4 种 batch_size 有 NaN"
fi

# === 5. 性能未退化(0.20) ===
echo ""
echo ">>> [5/6] 性能测试..."
GPU_TIME=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --device cuda --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
CPU_TIME=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --device cpu --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')

if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 2 else 1)" 2>/dev/null; then
        score=0.80
        echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x(无 CPU 回退)"
    else
        echo "  ⚠️ GPU 加速不足(${SPEEDUP}x),可能有 CPU 回退"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

# === 6. 代码修改检查(0.20) ===
echo ""
echo ">>> [6/6] 代码修改检查..."

# 检查 train.py 是否被修改(对比原始版本)
# 原始 train.py 不应该有 isnan/clip_grad/nan_to_num 等 hack
# 也不应该使用 nn.LayerNorm(应该用 CudaLayerNorm)
if grep -q "nn.LayerNorm" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ train.py 使用了 PyTorch 原生 LayerNorm(应该用 CudaLayerNorm)"
    HACK=1
else
    echo "  ✅ train.py 使用自定义 CudaLayerNorm"
fi

# 检查 CUDA 内核是否被修改(应该被修改)
if grep -q "j <= N" "$WORKSPACE/layernorm_cuda/layernorm_cuda_kernel.cu" 2>/dev/null; then
    echo "  ⚠️ CUDA 内核仍有 off-by-one bug(未修复)"
else
    echo "  ✅ CUDA 内核已修复"
fi

# Anti-hack:检查 Python 层有没有 hack
echo ""
echo ">>> Anti-hack 检查..."
if grep -rn "isnan\|isinf\|nan_to_num\|clip_grad\|clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 NaN 相关 hack"
    HACK=1
else
    echo "  ✅ 未发现 NaN hack"
fi

if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 CPU 回退"
    HACK=1
else
    echo "  ✅ 未发现 CPU 回退"
fi

if grep -rn "nn.LayerNorm" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现换用 PyTorch 原生 LayerNorm"
    HACK=1
else
    echo "  ✅ 未换用原生 LayerNorm"
fi

# === 最终评分 ===
echo ""
echo "========================================="
if [ $HACK -eq 0 ]; then
    echo " 最终分数: $score"
else
    score=$(python -c "print(f'{$score * 0.5:.2f}')")
    echo " 最终分数: $score (检测到 hack,减半)"
fi
echo "========================================="

echo "$score" > "$REWARD_FILE"

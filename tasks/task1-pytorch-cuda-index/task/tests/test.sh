#!/bin/bash
# tests/test.sh — Task 1 判题脚本(分层评分 + anti-hack)
# 注意:不使用 set -e

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 1: PyTorch LayerNorm CUDA rsqrt Bug"
echo "========================================="

# === 1. 基础:PyTorch 可导入(0.10) ===
echo ""
echo ">>> [1/6] PyTorch 检查..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ PyTorch 可导入"
else
    echo "  ❌ PyTorch 无法导入"
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

# === 4. 多 batch_size 测试(0.10) ===
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

# 检查 CUDA kernel 是否被修复
KERNEL_FILE="/build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu"
if [ -f "$KERNEL_FILE" ]; then
    if grep -q "rsqrt(wd.sigma2 + eps)" "$KERNEL_FILE"; then
        echo "  ✅ CUDA kernel 已修复(rsqrt 有 eps)"
    elif grep -q "rsqrt(wd.sigma2)" "$KERNEL_FILE"; then
        echo "  ⚠️ CUDA kernel 仍有 bug(rsqrt 无 eps)"
    else
        echo "  ⚠️ CUDA kernel 结构异常"
    fi
else
    echo "  ⚠️ 找不到 CUDA kernel 文件"
fi

# Anti-hack 检查(只检查"掩盖"NaN 的 hack,不检查"检测"NaN)
echo ""
echo ">>> Anti-hack 检查..."
if grep -rn "nan_to_num\|clip_grad\|clamp\|torch.where.*nan\|replace.*nan\|\.clamp\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 NaN 掩盖 hack"
    HACK=1
else
    echo "  ✅ 未发现 NaN 掩盖 hack"
fi

if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 CPU 回退"
    HACK=1
else
    echo "  ✅ 未发现 CPU 回退"
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

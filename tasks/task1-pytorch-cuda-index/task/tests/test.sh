#!/bin/bash
# tests/test.sh — Task 1 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  PyTorch 可导入
#   0.05  CUDA 可用
#   0.25  10 个 seed 无 NaN
#   0.15  4 种 batch_size 通过
#   0.20  GPU 比 CPU 快 2x+
#   0.25  Anti-hack: 无 NaN 掩盖 + 无 CPU 回退 + 关键文件未被修改
#
# 如果检测到 hack,总分减半。
# 不暴露任何 bug 细节。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 1 评测"
echo "========================================="

# === 1. 基础:PyTorch 可导入(0.10) ===
echo ""
echo ">>> [1/5] PyTorch 检查..."
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
echo ">>> [2/5] GPU 检查..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 核心:训练无 NaN(0.25) ===
echo ""
echo ">>> [3/5] 训练测试(多 seed)..."
NAN_COUNT=0
for seed in 1 2 3 4 5 6 7 8 9 10; do
    result=$(python "$WORKSPACE/train.py" --steps 100 --seed $seed --device cuda 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        NAN_COUNT=$((NAN_COUNT + 1))
    fi
done

if [ $NAN_COUNT -eq 0 ]; then
    score=0.40
    echo "  ✅ 10 个 seed 全部通过(无 NaN)"
elif [ $NAN_COUNT -le 2 ]; then
    score=0.25
    echo "  ⚠️ $NAN_COUNT/10 个 seed 有 NaN"
else
    echo "  ❌ $NAN_COUNT/10 个 seed 有 NaN"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 多 batch_size 测试(0.15) ===
echo ""
echo ">>> [4/5] 多 batch_size 测试..."
BATCH_FAIL=0
for bs in 16 32 64 128; do
    result=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --batch_size $bs --device cuda 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        BATCH_FAIL=$((BATCH_FAIL + 1))
    fi
done

if [ $BATCH_FAIL -eq 0 ]; then
    score=0.55
    echo "  ✅ 4 种 batch_size 全部通过"
else
    echo "  ❌ $BATCH_FAIL/4 种 batch_size 有 NaN"
fi

# === 5. 性能 + Anti-hack(0.45) ===
echo ""
echo ">>> [5/5] 性能 + Anti-hack 检查..."

# GPU vs CPU 性能
GPU_TIME=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --device cuda --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
CPU_TIME=$(python "$WORKSPACE/train.py" --steps 50 --seed 42 --device cpu --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')

if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 2 else 1)" 2>/dev/null; then
        echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x"
    else
        echo "  ⚠️ GPU 加速不足(${SPEEDUP}x)"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

# Anti-hack:检查 NaN 掩盖
if grep -rn "nan_to_num\|clip_grad\|clamp\|torch.where.*nan\|replace.*nan\|\.clamp\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 NaN 掩盖 hack"
    HACK=1
else
    echo "  ✅ 未发现 NaN 掩盖"
fi

# Anti-hack:检查 CPU 回退
if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 CPU 回退"
    HACK=1
else
    echo "  ✅ 未发现 CPU 回退"
fi

# Anti-hack:检查 train.py 关键内容未被修改
if grep -q "ImageClassifier" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ✅ train.py 未被修改"
else
    echo "  ❌ train.py 被修改"
    HACK=1
fi

# Anti-hack:检查 model.py 关键内容未被修改
if grep -q "FeatureNorm\|F.layer_norm" "$WORKSPACE/model.py" 2>/dev/null; then
    echo "  ✅ model.py 未被修改"
else
    echo "  ❌ model.py 被修改"
    HACK=1
fi

# 性能 + anti-hack 全部通过
if [ $HACK -eq 0 ]; then
    score=1.0
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

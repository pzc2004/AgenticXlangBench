#!/bin/bash
# tests/test.sh — Task 13 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  OpenCV 可导入
#   0.05  CUDA 可用
#   0.30  单次 resize PSNR 检查(Bug 1)
#   0.20  多缩放比例 PSNR 检查(Bug 2)
#   0.15  特征匹配质量检查(Bug 3)
#   0.10  性能:GPU 比 CPU 快 5x+
#   0.10  Anti-hack: 无 CPU 替代 + 无后处理 + 关键文件未被修改
#
# 如果检测到 hack,总分减半。
# 不暴露任何 bug 细节。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 13 评测"
echo "========================================="

# === 1. 基础:OpenCV 可导入(0.10) ===
echo ""
echo ">>> [1/6] OpenCV 检查..."
if python -c "import cv2; print(f'OpenCV {cv2.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ OpenCV 可导入"
else
    echo "  ❌ OpenCV 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 基础:CUDA 可用(0.05) ===
echo ""
echo ">>> [2/6] CUDA 检查..."
if python -c "import cv2; assert cv2.cuda.getCudaEnabledDeviceCount() > 0" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 单次 resize PSNR 检查(0.30) ===
echo ""
echo ">>> [3/6] 单次 resize PSNR 测试(Bug 1)..."
result=$(python "$WORKSPACE/test_resize.py" 2>&1)
echo "$result"

psnr=$(echo "$result" | grep "RESULT_PSNR" | awk '{print $2}')
mse=$(echo "$result" | grep "RESULT_MSE" | awk '{print $2}')

if [ -n "$psnr" ]; then
    psnr_ok=$(python -c "print(1 if $psnr > 40 else 0)" 2>/dev/null || echo "0")
    if [ "$psnr_ok" = "1" ]; then
        score=0.45
        echo "  ✅ PSNR = ${psnr} dB (> 40 dB)"
    else
        psnr_ok30=$(python -c "print(1 if $psnr > 30 else 0)" 2>/dev/null || echo "0")
        if [ "$psnr_ok30" = "1" ]; then
            score=0.30
            echo "  ⚠️ PSNR = ${psnr} dB (30-40 dB, 部分修复)"
        else
            echo "  ❌ PSNR = ${psnr} dB (< 30 dB)"
            echo "$score" > "$REWARD_FILE"
            exit 0
        fi
    fi
else
    echo "  ❌ 无法解析 PSNR"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 多缩放比例 PSNR 检查(0.20) ===
echo ""
echo ">>> [4/6] 多缩放比例 PSNR 测试(Bug 2)..."
avg_psnr=$(echo "$result" | grep "RESULT_AVG_PSNR" | awk '{print $2}')

if [ -n "$avg_psnr" ]; then
    avg_ok=$(python -c "print(1 if $avg_psnr > 35 else 0)" 2>/dev/null || echo "0")
    if [ "$avg_ok" = "1" ]; then
        score=0.65
        echo "  ✅ 平均 PSNR = ${avg_psnr} dB (> 35 dB)"
    else
        echo "  ⚠️ 平均 PSNR = ${avg_psnr} dB (< 35 dB)"
    fi
else
    echo "  ⚠️ 无法解析平均 PSNR"
fi

# === 5. 特征匹配质量检查(0.15) ===
echo ""
echo ">>> [5/6] 特征匹配质量测试(Bug 3)..."
inlier=$(echo "$result" | grep "RESULT_INLIER" | awk '{print $2}')

if [ -n "$inlier" ]; then
    inlier_ok=$(python -c "print(1 if $inlier > 0.80 else 0)" 2>/dev/null || echo "0")
    if [ "$inlier_ok" = "1" ]; then
        score=0.80
        echo "  ✅ Inlier 率 = $(python -c "print(f'{$inlier*100:.1f}%')") (> 80%)"
    else
        echo "  ⚠️ Inlier 率 = $(python -c "print(f'{$inlier*100:.1f}%')") (< 80%)"
    fi
else
    echo "  ⚠️ 无法解析 Inlier 率"
fi

# === 6. 性能 + Anti-hack(0.20) ===
echo ""
echo ">>> [6/6] 性能 + Anti-hack 检查..."

# 性能测试: GPU vs CPU resize
GPU_TIME=$(python -c "
import cv2, time, numpy as np
img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(img)
t0 = time.time()
for _ in range(100):
    cv2.cuda.resize(gpu_img, (960, 540))
t1 = time.time()
print(f'{(t1-t0)/100:.6f}')
" 2>/dev/null)

CPU_TIME=$(python -c "
import cv2, time, numpy as np
img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
t0 = time.time()
for _ in range(100):
    cv2.resize(img, (960, 540))
t1 = time.time()
print(f'{(t1-t0)/100:.6f}')
" 2>/dev/null)

if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 5.0 else 1)" 2>/dev/null; then
        echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x"
    else
        echo "  ⚠️ GPU 加速不足(${SPEEDUP}x)"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

# Anti-hack: 检查是否有 CPU 替代
if grep -rn "cv2\.resize\b" "$WORKSPACE/test_resize.py" 2>/dev/null | grep -v "cuda" | grep -v "cpu_result" | grep -v "CPU" | grep -v "#"; then
    echo "  ❌ 发现用 CPU resize 替代 GPU resize"
    HACK=1
else
    echo "  ✅ 未发现 CPU 替代"
fi

# Anti-hack: 检查是否有后处理校正
if grep -rn "np\.clip\|np\.round\|astype.*uint8\|pixel.*correct\|adjust" "$WORKSPACE/test_resize.py" 2>/dev/null | grep -v "#"; then
    echo "  ❌ 发现像素后处理校正"
    HACK=1
else
    echo "  ✅ 未发现后处理校正"
fi

# Anti-hack: 检查关键文件未被修改
if grep -q "create_test_image" "$WORKSPACE/test_resize.py" 2>/dev/null; then
    echo "  ✅ test_resize.py 未被修改"
else
    echo "  ❌ test_resize.py 被修改"
    HACK=1
fi

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

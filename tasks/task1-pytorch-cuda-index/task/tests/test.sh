#!/bin/bash
WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 1 评测"
echo "========================================="

echo ""
echo ">>> [1/7] 环境检查..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ PyTorch 可导入"
else
    echo "  ❌ PyTorch 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

echo ""
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

echo ""
echo ">>> [2/7] 前向检查 A..."
FWD_PASS=0
for seed in 1 2 3 4 5; do
    result=$(python -c "
import torch, torch.nn as nn, sys
sys.path.insert(0, '$WORKSPACE')
from model import ImageClassifier
torch.manual_seed($seed)
if torch.cuda.is_available(): torch.cuda.manual_seed($seed)
model_cpu = ImageClassifier().eval()
model_cuda = ImageClassifier().cuda().eval()
for (n1, p1), (n2, p2) in zip(model_cpu.named_parameters(), model_cuda.named_parameters()):
    p2.data.copy_(p1.data)
torch.manual_seed(12345)
x = torch.randn(8, 3, 32, 32)
out_cpu = model_cpu(x)
out_cuda = model_cuda(x.cuda())
abs_diff = (out_cpu - out_cuda.cpu()).abs().max().item()
max_val = max(out_cpu.abs().max().item(), out_cuda.abs().cpu().max().item(), 1e-8)
rel_diff = abs_diff / max_val
print(f'rel_diff={rel_diff:.6f}')
if rel_diff < 0.001: print('PASS')
else: print('FAIL')
" 2>&1)
    if echo "$result" | grep -q "PASS"; then
        rel=$(echo "$result" | grep "rel_diff" | sed 's/.*rel_diff=\([^ ]*\).*/\1/')
        echo "  ✅ seed=$seed: rel_diff=$rel"
        FWD_PASS=$((FWD_PASS + 1))
    else
        rel=$(echo "$result" | grep "rel_diff" | sed 's/.*rel_diff=\([^ ]*\).*/\1/')
        echo "  ❌ seed=$seed: rel_diff=$rel"
    fi
done

if [ $FWD_PASS -eq 5 ]; then
    score=$(python -c "print(f'{$score + 0.15:.2f}')")
    echo "  ✅ 全部通过"
elif [ $FWD_PASS -ge 3 ]; then
    score=$(python -c "print(f'{$score + 0.07:.2f}')")
    echo "  ⚠️ $FWD_PASS/5 通过"
else
    echo "  ❌ $FWD_PASS/5 通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

echo ""
echo ">>> [3/7] 前向检查 B..."
RACE_PASS=0
for seed in 1 2 3; do
    result=$(python -c "
import torch, torch.nn as nn, sys
sys.path.insert(0, '$WORKSPACE')
from model import ImageClassifier
torch.manual_seed($seed)
if torch.cuda.is_available(): torch.cuda.manual_seed($seed)
model = ImageClassifier().cuda().eval()
torch.manual_seed(12345)
x = torch.randn(128, 3, 32, 32).cuda()
results = []
for _ in range(50):
    out = model(x)
    results.append(out.detach().clone())
max_diff = 0
for i in range(1, 50):
    diff = (results[0] - results[i]).abs().max().item()
    max_diff = max(max_diff, diff)
print(f'max_diff={max_diff:.8f}')
if max_diff < 1e-7: print('PASS')
else: print('FAIL')
" 2>&1)
    if echo "$result" | grep -q "PASS"; then
        diff=$(echo "$result" | grep "max_diff" | sed 's/.*max_diff=\([^ ]*\).*/\1/')
        echo "  ✅ seed=$seed: max_diff=$diff"
        RACE_PASS=$((RACE_PASS + 1))
    else
        diff=$(echo "$result" | grep "max_diff" | sed 's/.*max_diff=\([^ ]*\).*/\1/')
        echo "  ❌ seed=$seed: max_diff=$diff"
    fi
done

if [ $RACE_PASS -eq 3 ]; then
    score=$(python -c "print(f'{$score + 0.05:.2f}')")
    echo "  ✅ 全部通过"
elif [ $RACE_PASS -ge 2 ]; then
    score=$(python -c "print(f'{$score + 0.03:.2f}')")
    echo "  ⚠️ $RACE_PASS/3 通过"
else
    echo "  ❌ $RACE_PASS/3 通过"
fi

echo ""
echo ">>> [4/7] 反向检查 A..."
RACE_BWD_PASS=0
for seed in 1 2 3; do
    result=$(python -c "
import torch, torch.nn as nn, sys
sys.path.insert(0, '$WORKSPACE')
from model import ImageClassifier
torch.manual_seed($seed)
if torch.cuda.is_available(): torch.cuda.manual_seed($seed)
model = ImageClassifier().cuda().eval()
torch.manual_seed(12345)
x = torch.randn(128, 3, 32, 32).cuda()
y = torch.randint(0, 10, (128,)).cuda()
grads = []
for _ in range(20):
    model.zero_grad()
    out = model(x)
    loss = nn.CrossEntropyLoss()(out, y)
    loss.backward()
    g = next(model.parameters()).grad.detach().clone()
    grads.append(g)
max_diff = 0
for i in range(1, 20):
    diff = (grads[0] - grads[i]).abs().max().item()
    max_diff = max(max_diff, diff)
print(f'max_diff={max_diff:.8f}')
if max_diff < 1e-7: print('PASS')
else: print('FAIL')
" 2>&1)
    if echo "$result" | grep -q "PASS"; then
        diff=$(echo "$result" | grep "max_diff" | sed 's/.*max_diff=\([^ ]*\).*/\1/')
        echo "  ✅ seed=$seed: max_diff=$diff"
        RACE_BWD_PASS=$((RACE_BWD_PASS + 1))
    else
        diff=$(echo "$result" | grep "max_diff" | sed 's/.*max_diff=\([^ ]*\).*/\1/')
        echo "  ❌ seed=$seed: max_diff=$diff"
    fi
done

if [ $RACE_BWD_PASS -eq 3 ]; then
    score=$(python -c "print(f'{$score + 0.05:.2f}')")
    echo "  ✅ 全部通过"
elif [ $RACE_BWD_PASS -ge 2 ]; then
    score=$(python -c "print(f'{$score + 0.03:.2f}')")
    echo "  ⚠️ $RACE_BWD_PASS/3 通过"
else
    echo "  ❌ $RACE_BWD_PASS/3 通过"
fi

echo ""
echo ">>> [5/7] 梯度检查..."
GRAD_PASS=0
for seed in 1 2 3 4 5; do
    result=$(python -c "
import torch, torch.nn as nn, sys
sys.path.insert(0, '$WORKSPACE')
from model import ImageClassifier
torch.manual_seed($seed)
if torch.cuda.is_available(): torch.cuda.manual_seed($seed)
model_cpu = ImageClassifier().eval()
model_cuda = ImageClassifier().cuda().eval()
for (n1, p1), (n2, p2) in zip(model_cpu.named_parameters(), model_cuda.named_parameters()):
    p2.data.copy_(p1.data)
torch.manual_seed(12345)
x = torch.randn(8, 3, 32, 32)
y = torch.randint(0, 10, (8,))
x_cpu = x.clone().requires_grad_(True)
x_cuda = x.clone().cuda().requires_grad_(True)
out_cpu = model_cpu(x_cpu)
out_cuda = model_cuda(x_cuda)
loss_cpu = nn.CrossEntropyLoss()(out_cpu, y)
loss_cuda = nn.CrossEntropyLoss()(out_cuda, y.cuda())
loss_cpu.backward()
loss_cuda.backward()
abs_diff = (x_cpu.grad - x_cuda.grad.cpu()).abs().max().item()
max_val = max(x_cpu.grad.abs().max().item(), x_cuda.grad.cpu().abs().max().item(), 1e-8)
rel_diff = abs_diff / max_val
print(f'rel_diff={rel_diff:.6f}')
if rel_diff < 0.001: print('PASS')
else: print('FAIL')
" 2>&1)
    if echo "$result" | grep -q "PASS"; then
        rel=$(echo "$result" | grep "rel_diff" | sed 's/.*rel_diff=\([^ ]*\).*/\1/')
        echo "  ✅ seed=$seed: rel_diff=$rel"
        GRAD_PASS=$((GRAD_PASS + 1))
    else
        rel=$(echo "$result" | grep "rel_diff" | sed 's/.*rel_diff=\([^ ]*\).*/\1/')
        echo "  ❌ seed=$seed: rel_diff=$rel"
    fi
done

if [ $GRAD_PASS -eq 5 ]; then
    score=$(python -c "print(f'{$score + 0.15:.2f}')")
    echo "  ✅ 全部通过"
elif [ $GRAD_PASS -ge 3 ]; then
    score=$(python -c "print(f'{$score + 0.07:.2f}')")
    echo "  ⚠️ $GRAD_PASS/5 通过"
else
    echo "  ❌ $GRAD_PASS/5 通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

echo ""
echo ">>> [6/8] Kernel 级带电检查（GELU-tanh / softmax / BN / Dropout）..."
KPASS=0
KTOTAL=6

# K1: GELU tanh approximate（触发 Gelu tanh 路径 bug）
r=$(python -c "
import torch, torch.nn.functional as F
torch.manual_seed(0); x=torch.randn(64,512)
yc=F.gelu(x,approximate='tanh'); yg=F.gelu(x.cuda(),approximate='tanh')
d=(yc-yg.cpu()).abs().max().item()/max(yc.abs().max().item(),1e-8)
print('PASS' if d<1e-3 else f'FAIL rel={d:.5f}')
" 2>&1)
echo "  GELU-tanh: $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

# K2: 4D spatial softmax 前向+反向（dim=1, inner_size>1）
r=$(python -c "
import torch, torch.nn.functional as F
torch.manual_seed(0); x=torch.randn(8,16,32,32)
xc=x.clone().requires_grad_(True); xg=x.clone().cuda().requires_grad_(True)
yc=F.softmax(xc,dim=1); yg=F.softmax(xg,dim=1)
fd=(yc-yg.cpu()).abs().max().item()
yc.sum().backward(); yg.sum().backward()
bd=(xc.grad-xg.grad.cpu()).abs().max().item()
print('PASS' if fd<1e-4 and bd<1e-4 else f'FAIL fd={fd:.6f} bd={bd:.6f}')
" 2>&1)
echo "  spatial-softmax(dim=1): $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

# K3: 2D 大 softmax 反向（dim=-1, dim_size>1024，走 blockReduce）
r=$(python -c "
import torch, torch.nn.functional as F
torch.manual_seed(0); x=torch.randn(64,2048); g=torch.randn(64,2048)
xc=x.clone().requires_grad_(True); xg=x.clone().cuda().requires_grad_(True)
(F.softmax(xc,dim=-1)*g).sum().backward()
(F.softmax(xg,dim=-1)*g.cuda()).sum().backward()
bd=(xc.grad-xg.grad.cpu()).abs().max().item()
print('PASS' if bd<1e-4 else f'FAIL bd={bd:.6f}')
" 2>&1)
echo "  softmax-bwd(2D large): $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

# K4: BatchNorm train running 统计更新（触发 update_stats_and_invert running_var bug）
r=$(python -c "
import torch
torch.manual_seed(0); x=torch.randn(32,16,8,8)
bn_c=torch.nn.BatchNorm2d(16); bn_g=torch.nn.BatchNorm2d(16).cuda()
bn_g.load_state_dict(bn_c.state_dict())
bn_c.train(); bn_g.train()
for _ in range(5):
    bn_c(x); bn_g(x.cuda())
dv=(bn_c.running_var-bn_g.running_var.cpu()).abs().max().item()
dm=(bn_c.running_mean-bn_g.running_mean.cpu()).abs().max().item()
print('PASS' if dv<1e-3 and dm<1e-3 else f'FAIL var={dv:.5f} mean={dm:.5f}')
" 2>&1)
echo "  BN-train-running_stats: $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

# K5: BatchNorm eval invstd（small var 放大 eps 错误，触发 eps×100 bug）
r=$(python -c "
import torch
torch.manual_seed(0); x=torch.randn(8,16,8,8)
bn_c=torch.nn.BatchNorm2d(16).eval(); bn_g=torch.nn.BatchNorm2d(16).cuda().eval()
with torch.no_grad():
    for bn in (bn_c,bn_g):
        bn.running_var.fill_(1e-3); bn.running_mean.zero_()
yc=bn_c(x); yg=bn_g(x.cuda())
d=(yc-yg.cpu()).abs().max().item()/max(yc.abs().max().item(),1e-8)
print('PASS' if d<1e-3 else f'FAIL rel={d:.5f}')
" 2>&1)
echo "  BN-eval-invstd: $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

# K6: Dropout train 缩放（固定 seed，检查保留元素缩放值，触发 scale bug）
r=$(python -c "
import torch, torch.nn.functional as F
torch.manual_seed(0); x=torch.ones(100000).cuda()
y=F.dropout(x,p=0.1,training=True)
nz=y[y!=0]
m=nz.mean().item()  # 正确 ≈ 1/0.9 = 1.1111
print('PASS' if abs(m-1.0/0.9)<0.005 else f'FAIL scale={m:.5f}')
" 2>&1)
echo "  Dropout-train-scale: $r"; echo "$r" | grep -q PASS && KPASS=$((KPASS+1))

score=$(python -c "print(f'{$score + 0.15*$KPASS/$KTOTAL:.4f}')")
echo "  >>> 带电检查通过 $KPASS/$KTOTAL（权重 0.15）"

echo ""
echo ">>> [7/8] 性能检查..."
GPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cuda --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
CPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cpu --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 1.5 else 1)" 2>/dev/null; then
        score=$(python -c "print(f'{$score + 0.05:.2f}')")
        echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x"
    else
        echo "  ⚠️ GPU 加速不足(${SPEEDUP}x)"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

echo ""
echo ">>> [8/8] 完整性检查..."
if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan\|replace.*nan\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现异常处理"
    HACK=1
else
    echo "  ✅ 未发现异常处理"
fi
if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现设备回退"
    HACK=1
else
    echo "  ✅ 未发现设备回退"
fi
if grep -q "ImageClassifier" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ✅ train.py 完整"
else
    echo "  ❌ train.py 被修改"
    HACK=1
fi
if grep -q "FeatureNorm\|F.layer_norm" "$WORKSPACE/model.py" 2>/dev/null && \
   grep -q "BatchNorm2d\|GroupNorm" "$WORKSPACE/model.py" 2>/dev/null; then
    score=$(python -c "print(f'{$score + 0.25:.2f}')")
    echo "  ✅ model.py 完整"
else
    echo "  ❌ model.py 被修改"
    HACK=1
fi

echo ""
echo "========================================="
if [ $HACK -eq 1 ]; then
    score=$(python -c "print(f'{$score * 0.5:.2f}')")
    echo " 最终分数: $score (异常)"
else
    echo " 最终分数: $score"
fi
echo "========================================="
echo "$score" > "$REWARD_FILE"

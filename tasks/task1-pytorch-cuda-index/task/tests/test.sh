#!/bin/bash
WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
HISTORY_FILE="/logs/verifier/history.log"   # 受保护(root 700)，累积每次判分的分数，供 run.sh 取 max
mkdir -p "$(dirname "$REWARD_FILE")"

# 写当前分：覆盖 reward.txt(本次) + 追加 history.log(历史最大值用)。
# agent 每跑一次 grade(setuid→root 跑本脚本)就诚实落一条，无法伪造(目录 root:700)。
write_reward() { echo "$1" > "$REWARD_FILE"; echo "$1" >> "$HISTORY_FILE"; }

score=0
HACK=0

echo "========================================="
echo " Task 1 评测（v2：删除型带电）"
echo "========================================="

echo ""
echo ">>> [1/6] 环境检查..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    score=0.05
    echo "  ✅ PyTorch 可导入"
else
    echo "  ❌ PyTorch 无法导入"; write_reward "$score"; exit 0
fi
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.10
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"; write_reward "$score"; exit 0
fi

echo ""
echo ">>> [2/6] 模型前向 GPU-vs-CPU (8,3,32,32)..."
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
out_cpu = model_cpu(x); out_cuda = model_cuda(x.cuda())
abs_diff = (out_cpu - out_cuda.cpu()).abs().max().item()
max_val = max(out_cpu.abs().max().item(), out_cuda.abs().cpu().max().item(), 1e-8)
print('PASS' if abs_diff/max_val < 0.001 else 'FAIL')
" 2>&1)
    echo "$result" | grep -q "PASS" && FWD_PASS=$((FWD_PASS + 1))
done
if [ $FWD_PASS -eq 5 ]; then score=$(python -c "print(f'{$score + 0.10:.4f}')"); echo "  ✅ 5/5"
elif [ $FWD_PASS -ge 3 ]; then score=$(python -c "print(f'{$score + 0.05:.4f}')"); echo "  ⚠️ $FWD_PASS/5"
else echo "  ❌ $FWD_PASS/5"; write_reward "$score"; exit 0; fi

echo ""
echo ">>> [3/6] 模型梯度 GPU-vs-CPU (8,3,32,32)..."
GRAD_PASS=0
for seed in 1 2 3 4 5; do
    result=$(python -c "
import torch, torch.nn as nn, sys
sys.path.insert(0, '$WORKSPACE')
from model import ImageClassifier
torch.manual_seed($seed)
if torch.cuda.is_available(): torch.cuda.manual_seed($seed)
model_cpu = ImageClassifier().eval(); model_cuda = ImageClassifier().cuda().eval()
for (n1, p1), (n2, p2) in zip(model_cpu.named_parameters(), model_cuda.named_parameters()):
    p2.data.copy_(p1.data)
torch.manual_seed(12345)
x = torch.randn(8, 3, 32, 32); y = torch.randint(0, 10, (8,))
x_cpu = x.clone().requires_grad_(True); x_cuda = x.clone().cuda().requires_grad_(True)
out_cpu = model_cpu(x_cpu); out_cuda = model_cuda(x_cuda)
nn.CrossEntropyLoss()(out_cpu, y).backward()
nn.CrossEntropyLoss()(out_cuda, y.cuda()).backward()
abs_diff = (x_cpu.grad - x_cuda.grad.cpu()).abs().max().item()
max_val = max(x_cpu.grad.abs().max().item(), x_cuda.grad.cpu().abs().max().item(), 1e-8)
print('PASS' if abs_diff/max_val < 0.001 else 'FAIL')
" 2>&1)
    echo "$result" | grep -q "PASS" && GRAD_PASS=$((GRAD_PASS + 1))
done
if [ $GRAD_PASS -eq 5 ]; then score=$(python -c "print(f'{$score + 0.10:.4f}')"); echo "  ✅ 5/5"
elif [ $GRAD_PASS -ge 3 ]; then score=$(python -c "print(f'{$score + 0.05:.4f}')"); echo "  ⚠️ $GRAD_PASS/5"
else echo "  ❌ $GRAD_PASS/5"; write_reward "$score"; exit 0; fi

echo ""
echo ">>> [4/6] 确定性（race）检查：前向+反向重复一致性..."
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
x = torch.randn(128, 3, 32, 32).cuda(); y = torch.randint(0,10,(128,)).cuda()
outs=[]; grads=[]
for _ in range(40):
    model.zero_grad()
    xx=x.clone().requires_grad_(True)
    o=model(xx); outs.append(o.detach().clone())
    nn.CrossEntropyLoss()(o,y).backward(); grads.append(xx.grad.detach().clone())
md=max((outs[0]-outs[i]).abs().max().item() for i in range(1,40))
gd=max((grads[0]-grads[i]).abs().max().item() for i in range(1,40))
print('PASS' if md<1e-7 and gd<1e-7 else 'FAIL')
" 2>&1)
    echo "$result" | grep -q "PASS" && RACE_PASS=$((RACE_PASS + 1))
done
if [ $RACE_PASS -eq 3 ]; then score=$(python -c "print(f'{$score + 0.05:.4f}')"); echo "  ✅ 3/3"
elif [ $RACE_PASS -ge 2 ]; then score=$(python -c "print(f'{$score + 0.03:.4f}')"); echo "  ⚠️ $RACE_PASS/3"
else echo "  ❌ $RACE_PASS/3"; fi

echo ""
echo ">>> [5/6] Kernel 级带电检查（删除型 + 条件/数值型，GPU-vs-CPU 多 warp 形状）..."
# 每个删除型 bug 都对应至少一项：用强制多 warp 归约的形状 + 多次重复抓 flaky race。
KOUT=$(python - <<'PYEOF' 2>&1
import torch, torch.nn as nn, torch.nn.functional as F
torch.backends.cudnn.enabled = False  # 强制走 native CUDA kernel（非 cudnn），命中被注入的 .cu
REP = 8  # 归约类重复多次，提升 flaky race 命中率（取最差）

def rel(a, b):
    d = (a - b).abs().max().item()
    m = max(a.abs().max().item(), b.abs().max().item(), 1e-8)
    return d / m

results = []
def check(name, fn, thr=1e-3):
    try:
        ok = fn(thr)
    except Exception as e:
        ok = False
        print(f"  [ERR] {name}: {e}")
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'} {name}")

# ---------- 激活 forward+backward（含删项 Bug16-20,24 + 条件/数值 Bug32-35,37-39）----------
def act_check(actfn, shape=(64,512), needs_weight=False):
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(*shape)
        worst = 0.0
        for _ in range(2):
            xc = x.clone().requires_grad_(True)
            xg = x.clone().cuda().requires_grad_(True)
            if needs_weight:
                wc = torch.tensor([0.25]); wg = torch.tensor([0.25]).cuda()
                wc.requires_grad_(True); wg.requires_grad_(True)
                yc = F.prelu(xc, wc); yg = F.prelu(xg, wg)
            else:
                yc = actfn(xc); yg = actfn(xg)
            worst = max(worst, rel(yc, yg.cpu()))
            g = torch.randn(*shape)
            (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
            worst = max(worst, rel(xc.grad, xg.grad.cpu()))
            if needs_weight:
                worst = max(worst, rel(wc.grad, wg.grad.cpu()))
        return worst < thr
    return f

check("act:silu",      act_check(F.silu))
check("act:gelu_tanh", act_check(lambda x: F.gelu(x, approximate='tanh')))
check("act:elu",       act_check(F.elu))
check("act:leakyrelu", act_check(lambda x: F.leaky_relu(x, 0.1)))
check("act:hardswish", act_check(F.hardswish))
check("act:prelu",     act_check(None, needs_weight=True))

# ---------- LayerNorm forward+backward（Bug1-8,21,25-28,42,43）----------
def ln_check(N, M=256):
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(M, N); w = torch.randn(N); b = torch.randn(N)
        worst = 0.0
        for _ in range(REP):
            xc = x.clone().requires_grad_(True); wc = w.clone().requires_grad_(True); bc = b.clone().requires_grad_(True)
            xg = x.clone().cuda().requires_grad_(True); wg = w.clone().cuda().requires_grad_(True); bg = b.clone().cuda().requires_grad_(True)
            yc = F.layer_norm(xc, (N,), wc, bc); yg = F.layer_norm(xg, (N,), wg, bg)
            worst = max(worst, rel(yc, yg.cpu()))
            g = torch.randn(M, N)
            (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
            worst = max(worst, rel(xc.grad, xg.grad.cpu()),
                        rel(wc.grad, wg.grad.cpu()), rel(bc.grad, bg.grad.cpu()))
        return worst < thr
    return f

check("ln:N1024(vec+dgamma32x32)", ln_check(1024))   # 向量化 fwd/bwd + dgamma 32x32
check("ln:N1000(dgamma_fallback)", ln_check(1000))   # dgamma fallback (N%32!=0)
check("ln:N1023(nonvec_compute_gI)", ln_check(1023)) # 非向量化 fwd/bwd compute_gI

# ---------- SoftMax（冷算子 Bug9-14）----------
def softmax2d_check(C):
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(64, C)
        worst = 0.0
        for _ in range(REP):
            xc = x.clone().requires_grad_(True); xg = x.clone().cuda().requires_grad_(True)
            yc = F.softmax(xc, dim=-1); yg = F.softmax(xg, dim=-1)
            worst = max(worst, (yc - yg.cpu()).abs().max().item())
            g = torch.randn(64, C)
            (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
            worst = max(worst, (xc.grad - xg.grad.cpu()).abs().max().item())
        return worst < 1e-4
    return f

def softmax_spatial_check():
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(8, 128, 16, 16)
        worst = 0.0
        for _ in range(REP):
            xc = x.clone().requires_grad_(True); xg = x.clone().cuda().requires_grad_(True)
            yc = F.softmax(xc, dim=1); yg = F.softmax(xg, dim=1)
            worst = max(worst, (yc - yg.cpu()).abs().max().item())
            g = torch.randn(8, 128, 16, 16)
            (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
            worst = max(worst, (xc.grad - xg.grad.cpu()).abs().max().item())
        return worst < 1e-4
    return f

check("softmax:2d_C2048(fwd_blockReduceWarp+bwd_blockReduce)", softmax2d_check(2048))
check("softmax:spatial_4D_dim1(spatialBlockReduceX)", softmax_spatial_check())

# ---------- GroupNorm（Bug15,31,41）----------
def gn_check(shape, groups):
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(*shape); C = shape[1]
        gn_c = nn.GroupNorm(groups, C); gn_g = nn.GroupNorm(groups, C).cuda()
        gn_g.load_state_dict(gn_c.state_dict())
        worst = 0.0
        for _ in range(REP):
            gn_c.zero_grad(); gn_g.zero_grad()
            xc = x.clone().requires_grad_(True); xg = x.clone().cuda().requires_grad_(True)
            yc = gn_c(xc); yg = gn_g(xg)
            worst = max(worst, rel(yc, yg.cpu()))
            g = torch.randn(*shape)
            (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
            worst = max(worst, rel(xc.grad, xg.grad.cpu()), rel(gn_c.weight.grad, gn_g.weight.grad.cpu()))
        return worst < thr
    return f

check("gn:N256_HxW1(GammaBeta1d_Kernel2)", gn_check((256, 256), 8))
check("gn:N256_HxW64(general_Kernel2)", gn_check((256, 64, 8, 8), 8))

# ---------- BatchNorm eval forward+backward（Bug22,23,29）----------
def bn_eval_check():
    def f(thr):
        torch.manual_seed(0)
        x = torch.randn(32, 16, 8, 8)
        bn_c = nn.BatchNorm2d(16).eval(); bn_g = nn.BatchNorm2d(16).cuda().eval()
        bn_g.load_state_dict(bn_c.state_dict())
        with torch.no_grad():
            for bn in (bn_c, bn_g):
                bn.running_var.fill_(0.7); bn.running_mean.fill_(0.1)
        xc = x.clone().requires_grad_(True); xg = x.clone().cuda().requires_grad_(True)
        yc = bn_c(xc); yg = bn_g(xg)
        worst = rel(yc, yg.cpu())
        g = torch.randn(32, 16, 8, 8)
        (yc * g).sum().backward(); (yg * g.cuda()).sum().backward()
        worst = max(worst, rel(xc.grad, xg.grad.cpu()))
        return worst < thr
    return f

# ---------- BatchNorm train running stats（Bug30,40）----------
def bn_train_check():
    def f(thr):
        torch.manual_seed(0); x = torch.randn(32, 16, 8, 8)
        bn_c = nn.BatchNorm2d(16); bn_g = nn.BatchNorm2d(16).cuda()
        bn_g.load_state_dict(bn_c.state_dict()); bn_c.train(); bn_g.train()
        for _ in range(5):
            bn_c(x); bn_g(x.cuda())
        dv = rel(bn_c.running_var, bn_g.running_var.cpu())
        dm = rel(bn_c.running_mean, bn_g.running_mean.cpu())
        return dv < thr and dm < thr
    return f

# ---------- BatchNorm eval invstd（Bug29 eps×100，小 var 放大错误）----------
def bn_eval_invstd_check():
    def f(thr):
        torch.manual_seed(0); x = torch.randn(8, 16, 8, 8)
        bn_c = nn.BatchNorm2d(16).eval(); bn_g = nn.BatchNorm2d(16).cuda().eval()
        with torch.no_grad():
            for bn in (bn_c, bn_g):
                bn.running_var.fill_(1e-3); bn.running_mean.zero_()
        yc = bn_c(x); yg = bn_g(x.cuda())
        return rel(yc, yg.cpu()) < thr
    return f

check("bn:eval_fwd_bwd(factor1/2)", bn_eval_check())
check("bn:train_running_stats", bn_train_check())
check("bn:eval_invstd_eps", bn_eval_invstd_check())

# ---------- Dropout train 缩放（Bug36）----------
def dropout_check():
    def f(thr):
        torch.manual_seed(0); x = torch.ones(100000).cuda()
        y = F.dropout(x, p=0.1, training=True)
        nz = y[y != 0]; m = nz.mean().item()
        return abs(m - 1.0/0.9) < 0.005
    return f
check("dropout:train_scale", dropout_check())

npass = sum(1 for _, ok in results if ok)
ntot = len(results)
print(f"KPASS={npass}")
print(f"KTOTAL={ntot}")
PYEOF
)
echo "$KOUT"
KPASS=$(echo "$KOUT" | grep "^KPASS=" | cut -d= -f2)
KTOTAL=$(echo "$KOUT" | grep "^KTOTAL=" | cut -d= -f2)
[ -z "$KPASS" ] && KPASS=0
[ -z "$KTOTAL" ] && KTOTAL=1
score=$(python -c "print(f'{$score + 0.50*$KPASS/$KTOTAL:.4f}')")
echo "  >>> Kernel 带电检查通过 $KPASS/$KTOTAL（权重 0.50）"

echo ""
echo ">>> [6/6] 性能 + 完整性（反 hack）..."
GPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cuda --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
CPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cpu --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 1.5 else 1)" 2>/dev/null; then
        score=$(python -c "print(f'{$score + 0.05:.4f}')"); echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x"
    else echo "  ⚠️ GPU 加速不足(${SPEEDUP}x)"; HACK=1; fi
else echo "  ⚠️ 性能测试无法运行"; fi

if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan\|replace.*nan\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现异常处理"; HACK=1
fi
if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现设备回退"; HACK=1
fi
if ! grep -q "ImageClassifier" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ train.py 被修改"; HACK=1
fi
if grep -q "FeatureNorm\|F.layer_norm" "$WORKSPACE/model.py" 2>/dev/null && \
   grep -q "BatchNorm2d\|GroupNorm" "$WORKSPACE/model.py" 2>/dev/null; then
    score=$(python -c "print(f'{$score + 0.10:.4f}')"); echo "  ✅ model.py 完整"
else echo "  ❌ model.py 被修改"; HACK=1; fi

echo ""
echo "========================================="
if [ $HACK -eq 1 ]; then
    score=$(python -c "print(f'{$score * 0.5:.4f}')")
    echo " 最终分数: $score (异常)"
else
    echo " 最终分数: $score"
fi
echo "========================================="
write_reward "$score"

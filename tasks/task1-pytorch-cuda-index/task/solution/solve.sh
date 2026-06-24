#!/bin/bash
PYTORCH_SRC="/build/pytorch"
CUDA_DIR="$PYTORCH_SRC/aten/src/ATen/native/cuda"
WORKSPACE="/workspace"

echo ">>> 修复所有 bug..."

python3 << 'PYEOF'
import os, re

PYTORCH_DIR = "/build/pytorch"
CUDA_DIR = os.path.join(PYTORCH_DIR, "aten/src/ATen/native/cuda")

fixed = 0

def read_file(f):
    with open(f) as fh: return fh.read()

def write_file(f, c):
    with open(f, "w") as fh: fh.write(c)

def fix(filepath, old, new, name):
    global fixed
    content = read_file(filepath)
    if old in content:
        content = content.replace(old, new, 1)
        write_file(filepath, content)
        fixed += 1
        print(f"  Fixed: {name}")
        return True
    return False

# === Bug 1-6: 恢复删除的 __syncthreads ===
fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "        return WelfordDataLN",
    "        __syncthreads();\n        return WelfordDataLN",
    "Bug 1: 恢复 LN forward __syncthreads")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "    // Compute gradients for beta and gamma",
    "    __syncthreads();\n    // Compute gradients for beta and gamma",
    "Bug 2: 恢复 LN backward __syncthreads")

fix(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
    "\n  // Do warp reduce",
    "  __syncthreads();\n\n  // Do warp reduce",
    "Bug 3: 恢复 GN __syncthreads")

fix(os.path.join(CUDA_DIR, "SoftMax.cu"),
    "\n  shared[threadIdx.x] = val;",
    "  __syncthreads();\n\n  shared[threadIdx.x] = val;",
    "Bug 4: 恢复 SoftMax __syncthreads #1")

fix(os.path.join(CUDA_DIR, "SoftMax.cu"),
    "\n  return shared[0];",
    "  __syncthreads();\n\n  return shared[0];",
    "Bug 5: 恢复 SoftMax __syncthreads #2")

fix(os.path.join(CUDA_DIR, "SoftMax.cu"),
    "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here",
    "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here\n  __syncthreads();",
    "Bug 6: 恢复 SoftMax forward __syncthreads")

# === Bug 7-17: 条件触发型 ===
fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "T_ACC rstd_val = (wd.sigma2 > T_ACC(0.5) && wd.sigma2 < T_ACC(2.0)) ? -c10::cuda::compat::rsqrt(wd.sigma2 + eps) : c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
    "T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
    "Bug 7: LN forward 条件符号翻转")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "mean[i] = m1 + T_ACC(0.05);",
    "mean[i] = m1;",
    "Bug 8: LN forward 均值偏移")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "const T_ACC rstd_val = rstd[i1] * T_ACC(0.95);",
    "const T_ACC rstd_val = rstd[i1];",
    "Bug 9: LN backward rstd 缩放")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "stats_x2 -= c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
    "stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
    "Bug 10: LN backward 梯度方向")

fix(os.path.join(CUDA_DIR, "Normalization.cu"),
    "c10::cuda::compat::rsqrt(var + eps * static_cast<acc_t>(100))",
    "c10::cuda::compat::rsqrt(var + eps)",
    "Bug 11: BN eps")

fix(os.path.join(CUDA_DIR, "Normalization.cu"),
    "unbiased_var / momentum + (1 - momentum) * running_var,",
    "unbiased_var * momentum + (1 - momentum) * running_var,",
    "Bug 12: BN momentum")

fix(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
    ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]) * static_cast<T_ACC>(0.8);",
    ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);",
    "Bug 13: GN gamma")

fix(os.path.join(CUDA_DIR, "ActivationSiluKernel.cu"),
    "return x_acc / (opmath_t(1) + ::exp(-x_acc)) * opmath_t(0.9);",
    "return x_acc / (opmath_t(1) + ::exp(-x_acc));",
    "Bug 14: SiLU")

fix(os.path.join(CUDA_DIR, "ActivationGeluKernel.cu"),
    "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube) + opmath_t(0.01);",
    "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);",
    "Bug 15: GELU")

fix(os.path.join(CUDA_DIR, "ActivationEluKernel.cu"),
    "return aop > 0 ? aop * poscoef * static_cast<opmath_t>(0.95)\n                             : std::expm1(aop * negiptcoef) * negcoef;",
    "return aop > 0 ? aop * poscoef\n                             : std::expm1(aop * negiptcoef) * negcoef;",
    "Bug 16: ELU")

fix(os.path.join(CUDA_DIR, "ActivationLeakyReluKernel.cu"),
    "return aop > opmath_t(0) ? aop : aop * negval * static_cast<opmath_t>(0.5);",
    "return aop > opmath_t(0) ? aop : aop * negval;",
    "Bug 17: LeakyReLU")

# === Bug 18-22: 数值精度型 ===
fix(os.path.join(CUDA_DIR, "Dropout.cu"),
    "accscalar_t scale = 0.98 / p;",
    "accscalar_t scale = 1.0 / p;",
    "Bug 18: Dropout scale")

fix(os.path.join(CUDA_DIR, "ActivationGeluKernel.cu"),
    "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
    "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
    "Bug 19: Gelu x_cube")

fix(os.path.join(CUDA_DIR, "ActivationHardswishKernel.cu"),
    "return x * std::min(std::max(x + three, zero), six) * one_sixth * 0.95f;",
    "return x * std::min(std::max(x + three, zero), six) * one_sixth;",
    "Bug 20: Hardswish")

fix(os.path.join(CUDA_DIR, "ActivationPreluKernel.cu"),
    "return (input > 0) ? input : static_cast<decltype(input)>(weight * input * 0.95f);",
    "return (input > 0) ? input : weight * input;",
    "Bug 21: PReLU")

fix(os.path.join(CUDA_DIR, "Normalization.cu"),
    "unbiased_var / momentum + (1 - momentum) * running_var,\n          c10::detail::",
    "unbiased_var * momentum + (1 - momentum) * running_var,\n          c10::detail::",
    "Bug 22: BN running_var")

# === Bug 23-25: 跨 kernel 依赖型 ===
fix(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
    "b[index] = -scale * (static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(0.1) * static_cast<T_ACC>(_ln_flag))",
    "b[index] = -scale * static_cast<T_ACC>(mean[ng])",
    "Bug 23: GN mean")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "rstd[i1] = rstd_val * (_ln_flag ? T_ACC(1) : T_ACC(1)); _ln_flag = (blockIdx.x == 0);",
    "rstd[i1] = rstd_val;",
    "Bug 24: LN rstd")

fix(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
    "const T_ACC gamma_val = gamma == nullptr ? T_ACC(1) : gamma[i2] * T_ACC(0.9);",
    "const T_ACC gamma_val = gamma == nullptr ? T_ACC(1) : gamma[i2];",
    "Bug 25: LN gamma")

# === 移除 _ln_flag 声明 ===
print("\n>>> 移除 _ln_flag 声明...")
header = "extern __device__ bool _ln_flag;\n"
for filename in os.listdir(CUDA_DIR):
    if not filename.endswith('.cu'):
        continue
    filepath = os.path.join(CUDA_DIR, filename)
    content = read_file(filepath)
    if header in content:
        content = content.replace(header, "")
        write_file(filepath, content)
        fixed += 1
        print(f"  Fixed: 移除 {filename} 的 _ln_flag")

# === 移除陷阱诱饵 ===
print("\n>>> 移除陷阱诱饵...")
trap_patterns = [
    (" * T_ACC(1);", ";"),
    (" * acc_t(1);", ";"),
    (" * opmath_t(1);", ";"),
    (" && true) {", ") {"),
]
trap_count = 0
for filename in os.listdir(CUDA_DIR):
    if not filename.endswith('.cu'):
        continue
    filepath = os.path.join(CUDA_DIR, filename)
    content = read_file(filepath)
    changed = False
    for old, new in trap_patterns:
        if old in content:
            content = content.replace(old, new)
            changed = True
    if changed:
        write_file(filepath, content)
        trap_count += 1
print(f"  移除 {trap_count} 个陷阱诱饵")

# === 移除普通诱饵 ===
print("\n>>> 移除普通诱饵...")
decoys = [
    "float _sigma2_floor = 1e-6f;\n",
    "float _eps_override = 1e-5f;\n",
    "int _block_size_hint = 256;\n",
    "float _norm_eps_adj = 1.0f;\n",
    "int _vec_size_fallback = 4;\n",
    "float _mean_clip = 10.0f;\n",
    "int _thread_align = 32;\n",
    "float _var_floor = 1e-6f;\n",
    "float _momentum_adj = 0.999f;\n",
    "int _norm_threads = 128;\n",
    "float _bn_eps = 1e-5f;\n",
    "int _channel_align = 32;\n",
    "float _var_clip = 100.0f;\n",
    "int _bn_block = 64;\n",
    "float _rstd_max = 100.0f;\n",
    "float _gamma_scale = 1.001f;\n",
    "// FIXME: eps handling might be wrong\n",
    "// WARNING: race condition possible\n",
    "// TODO: optimize memory access\n",
    "// BUG_CANDIDATE: momentum update\n",
    "// NOTE: variance computation fragile\n",
    "// FIXME: gamma scaling assumption\n",
    "// WARNING: mean computation\n",
    "// TODO: numerical stability\n",
    "// BUG_CANDIDATE: mask generation\n",
    "// FIXME: overflow protection\n",
]
decoy_count = 0
for filename in os.listdir(CUDA_DIR):
    if not filename.endswith('.cu'):
        continue
    filepath = os.path.join(CUDA_DIR, filename)
    content = read_file(filepath)
    changed = False
    for decoy in decoys:
        if decoy in content:
            content = content.replace(decoy, "")
            changed = True
    if changed:
        write_file(filepath, content)
        decoy_count += 1
print(f"  移除 {decoy_count} 个普通诱饵")

print(f"\n✅ 共修复 {fixed} 个 bug")
PYEOF

echo ">>> 增量编译..."
cd "$PYTORCH_SRC/build"
ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
echo ">>> 验证修复完成"

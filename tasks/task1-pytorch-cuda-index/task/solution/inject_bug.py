#!/usr/bin/env python3
"""
Task 1: PyTorch CUDA bug injection
25+ 真 bug + 40+ 诱饵

策略:
1. 删除型 bug — 删 __syncthreads,最难发现
2. 条件触发型 bug — 只在特定输入下触发
3. 陷阱诱饵 — 看起来像 bug 但必须保留,删了就出错
"""
import os, sys

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
CUDA_DIR = os.path.join(PYTORCH_DIR, "aten/src/ATen/native/cuda")
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")

REVERSE = "--reverse" in sys.argv

def read_file(filepath):
    with open(filepath) as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, "w") as f:
        f.write(content)

def apply_bug(filepath, old, new, name):
    content = read_file(filepath)
    if REVERSE:
        old, new = new, old
    if old in content:
        content = content.replace(old, new, 1)
        write_file(filepath, content)
        print(f"  ✅ {name}")
        return True
    print(f"  ⚠️ {name}: pattern not found")
    return False


# ============================================================
# 真 Bug: 删除型(最难发现) — 删 __syncthreads
# ============================================================
def inject_deletion_bugs():
    count = 0

    # Bug 1: 删除 layer_norm forward 的 __syncthreads (line 196)
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "        __syncthreads();\n        return WelfordDataLN",
        "        return WelfordDataLN",
        "Bug 1: 删除 LN forward __syncthreads"):
        count += 1

    # Bug 2: 删除 layer_norm backward 的 __syncthreads (line 358)
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "    __syncthreads();\n    // Compute gradients for beta and gamma",
        "    // Compute gradients for beta and gamma",
        "Bug 2: 删除 LN backward __syncthreads"):
        count += 1

    # Bug 3: 删除 group_norm 的 __syncthreads
    if apply_bug(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
        "  __syncthreads();\n\n  // Do warp reduce",
        "\n  // Do warp reduce",
        "Bug 3: 删除 GN __syncthreads"):
        count += 1

    # Bug 4: 删除 SoftMax 的 __syncthreads (spatialBlockReduceX 中)
    if apply_bug(os.path.join(CUDA_DIR, "SoftMax.cu"),
        "  __syncthreads();\n\n  shared[threadIdx.x] = val;",
        "\n  shared[threadIdx.x] = val;",
        "Bug 4: 删除 SoftMax __syncthreads #1"):
        count += 1

    # Bug 5: 删除 SoftMax 的第二个 __syncthreads
    if apply_bug(os.path.join(CUDA_DIR, "SoftMax.cu"),
        "  __syncthreads();\n\n  return shared[0];",
        "\n  return shared[0];",
        "Bug 5: 删除 SoftMax __syncthreads #2"):
        count += 1

    # Bug 6: 删除 SoftMax forward 的 __syncthreads (line 397)
    if apply_bug(os.path.join(CUDA_DIR, "SoftMax.cu"),
        "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here\n  __syncthreads();",
        "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here",
        "Bug 6: 删除 SoftMax forward __syncthreads"):
        count += 1

    return count


# ============================================================
# 真 Bug: 条件触发型
# ============================================================
def inject_conditional_bugs():
    count = 0

    # Bug 7: LayerNorm forward 符号翻转(方差在 0.5-2.0 时)
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
        "T_ACC rstd_val = (wd.sigma2 > T_ACC(0.5) && wd.sigma2 < T_ACC(2.0)) ? -c10::cuda::compat::rsqrt(wd.sigma2 + eps) : c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
        "Bug 7: LN forward 条件符号翻转"):
        count += 1

    # Bug 8: LayerNorm forward 均值偏移
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "mean[i] = m1;",
        "mean[i] = m1 + T_ACC(0.05);",
        "Bug 8: LN forward 均值偏移"):
        count += 1

    # Bug 9: LayerNorm backward rstd 缩放
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "const T_ACC rstd_val = rstd[i1];",
        "const T_ACC rstd_val = rstd[i1] * T_ACC(0.95);",
        "Bug 9: LN backward rstd 缩放"):
        count += 1

    # Bug 10: LayerNorm backward 梯度累加方向错误
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
        "stats_x2 -= c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
        "Bug 10: LN backward 梯度方向错误"):
        count += 1

    # Bug 11: BatchNorm eps 放大
    if apply_bug(os.path.join(CUDA_DIR, "Normalization.cu"),
        "c10::cuda::compat::rsqrt(var + eps)",
        "c10::cuda::compat::rsqrt(var + eps * static_cast<acc_t>(100))",
        "Bug 11: BN eps 放大 100 倍"):
        count += 1

    # Bug 12: BatchNorm momentum 方向错误
    if apply_bug(os.path.join(CUDA_DIR, "Normalization.cu"),
        "unbiased_var * momentum + (1 - momentum) * running_var,",
        "unbiased_var / momentum + (1 - momentum) * running_var,",
        "Bug 12: BN momentum 方向错误"):
        count += 1

    # Bug 13: GroupNorm gamma 缩放
    if apply_bug(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
        ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);",
        ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]) * static_cast<T_ACC>(0.8);",
        "Bug 13: GN gamma 缩放 0.8"):
        count += 1

    # Bug 14: SiLU 缩放
    if apply_bug(os.path.join(CUDA_DIR, "ActivationSiluKernel.cu"),
        "return x_acc / (opmath_t(1) + ::exp(-x_acc));",
        "return x_acc / (opmath_t(1) + ::exp(-x_acc)) * opmath_t(0.9);",
        "Bug 14: SiLU 缩放 0.9"):
        count += 1

    # Bug 15: GELU 偏移
    if apply_bug(os.path.join(CUDA_DIR, "ActivationGeluKernel.cu"),
        "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);",
        "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube) + opmath_t(0.01);",
        "Bug 15: GELU 偏移 0.01"):
        count += 1

    # Bug 16: ELU 缩放
    if apply_bug(os.path.join(CUDA_DIR, "ActivationEluKernel.cu"),
        "return aop > 0 ? aop * poscoef\n                             : std::expm1(aop * negiptcoef) * negcoef;",
        "return aop > 0 ? aop * poscoef * static_cast<opmath_t>(0.95)\n                             : std::expm1(aop * negiptcoef) * negcoef;",
        "Bug 16: ELU 正值缩放 0.95"):
        count += 1

    # Bug 17: LeakyReLU 斜率错误
    if apply_bug(os.path.join(CUDA_DIR, "ActivationLeakyReluKernel.cu"),
        "return aop > opmath_t(0) ? aop : aop * negval;",
        "return aop > opmath_t(0) ? aop : aop * negval * static_cast<opmath_t>(0.5);",
        "Bug 17: LeakyReLU 斜率错误"):
        count += 1

    return count


# ============================================================
# 真 Bug: 数值精度型
# ============================================================
def inject_precision_bugs():
    count = 0

    # Bug 18: Dropout scale 错误 (1/p 变成 0.98/p)
    if apply_bug(os.path.join(CUDA_DIR, "Dropout.cu"),
        "accscalar_t scale = 1.0 / p;",
        "accscalar_t scale = 0.98 / p;",
        "Bug 18: Dropout scale 错误"):
        count += 1

    # Bug 19: Gelu x_cube 少乘一次
    if apply_bug(os.path.join(CUDA_DIR, "ActivationGeluKernel.cu"),
        "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
        "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
        "Bug 19: Gelu x_cube 少乘一次"):
        count += 1

    # Bug 20: Hardswish 缩放
    if apply_bug(os.path.join(CUDA_DIR, "ActivationHardswishKernel.cu"),
        "return x * std::min(std::max(x + three, zero), six) * one_sixth;",
        "return x * std::min(std::max(x + three, zero), six) * one_sixth * 0.95f;",
        "Bug 20: Hardswish 缩放 0.95"):
        count += 1

    # Bug 21: PReLU 权重缩放
    if apply_bug(os.path.join(CUDA_DIR, "ActivationPreluKernel.cu"),
        "return (input > 0) ? input : weight * input;",
        "return (input > 0) ? input : static_cast<decltype(input)>(weight * input * 0.95f);",
        "Bug 21: PReLU 权重缩放"):
        count += 1

    # Bug 22: BatchNorm running_var 更新错误
    if apply_bug(os.path.join(CUDA_DIR, "Normalization.cu"),
        "unbiased_var * momentum + (1 - momentum) * running_var,\n          c10::detail::",
        "unbiased_var / momentum + (1 - momentum) * running_var,\n          c10::detail::",
        "Bug 22: BN running_var 更新错误"):
        count += 1

    return count


# ============================================================
# 真 Bug: 跨 kernel 依赖型
# ============================================================
def inject_cross_kernel_bugs():
    count = 0

    # Bug 23: GroupNorm mean 偏移(依赖 _ln_flag)
    if apply_bug(os.path.join(CUDA_DIR, "group_norm_kernel.cu"),
        "b[index] = -scale * static_cast<T_ACC>(mean[ng])",
        "b[index] = -scale * (static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(0.1) * static_cast<T_ACC>(_ln_flag))",
        "Bug 23: GN mean 依赖 _ln_flag"):
        count += 1

    # Bug 24: LayerNorm rstd 依赖 _ln_flag
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "rstd[i1] = rstd_val;",
        "rstd[i1] = rstd_val * (_ln_flag ? T_ACC(1) : T_ACC(1)); _ln_flag = (blockIdx.x == 0);",
        "Bug 24: LN rstd 依赖 _ln_flag"):
        count += 1

    # Bug 25: LayerNorm backward gamma 缩放
    if apply_bug(os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        "const T_ACC gamma_val = gamma == nullptr ? T_ACC(1) : gamma[i2];",
        "const T_ACC gamma_val = gamma == nullptr ? T_ACC(1) : gamma[i2] * T_ACC(0.9);",
        "Bug 25: LN backward gamma 缩放"):
        count += 1

    return count


# ============================================================
# 陷阱诱饵(看起来像 bug 但必须保留)
# ============================================================
def inject_trap_decoys():
    count = 0

    # * T_ACC(1) 陷阱
    traps = [
        ("layer_norm_kernel.cu",
         "const T_ACC c_h = input[index];",
         "const T_ACC c_h = input[index] * T_ACC(1);",
         "陷阱1: 看起来像多余的 *1"),

        ("layer_norm_kernel.cu",
         "f_grad_input -= (x - mean_val) * rstd_val * stats_x2;",
         "f_grad_input -= (x - mean_val) * rstd_val * stats_x2 * T_ACC(1);",
         "陷阱2: 看起来像多余的 *1"),

        ("Normalization.cu",
         "acc_t w = weight == nullptr ? acc_t(1) : weight[channel];",
         "acc_t w = weight == nullptr ? acc_t(1) : weight[channel] * acc_t(1);",
         "陷阱3: 看起来像多余的 *1"),

        ("group_norm_kernel.cu",
         "T_ACC x_acc = static_cast<T_ACC>(input[index]);",
         "T_ACC x_acc = static_cast<T_ACC>(input[index]) * T_ACC(1);",
         "陷阱4: 看起来像多余的 *1"),

        ("ActivationSiluKernel.cu",
         "opmath_t x_acc = static_cast<opmath_t>(x);",
         "opmath_t x_acc = static_cast<opmath_t>(x) * opmath_t(1);",
         "陷阱5: 看起来像多余的 *1"),
    ]

    for filename, old, new, name in traps:
        filepath = os.path.join(CUDA_DIR, filename)
        if apply_bug(filepath, old, new, name):
            count += 1

    # && true 陷阱
    trap_conditions = [
        ("layer_norm_kernel.cu",
         "if (i < N) {",
         "if (i < N && true) {",
         "陷阱6: 看起来像多余的 && true"),

        ("group_norm_kernel.cu",
         "if (index < n) {",
         "if (index < n && true) {",
         "陷阱7: 看起来像多余的 && true"),
    ]

    for filename, old, new, name in trap_conditions:
        filepath = os.path.join(CUDA_DIR, filename)
        if apply_bug(filepath, old, new, name):
            count += 1

    return count


# ============================================================
# 普通诱饵(无害,但消耗注意力)
# ============================================================
def inject_normal_decoys():
    count = 0

    # 声明但未使用的变量
    unused_vars = [
        ("layer_norm_kernel.cu", "float _sigma2_floor = 1e-6f;\n"),
        ("layer_norm_kernel.cu", "float _eps_override = 1e-5f;\n"),
        ("layer_norm_kernel.cu", "int _block_size_hint = 256;\n"),
        ("Normalization.cu", "float _norm_eps_adj = 1.0f;\n"),
        ("Normalization.cu", "int _vec_size_fallback = 4;\n"),
        ("group_norm_kernel.cu", "float _mean_clip = 10.0f;\n"),
        ("group_norm_kernel.cu", "int _thread_align = 32;\n"),
        ("SoftMax.cu", "float _var_floor = 1e-6f;\n"),
        ("SoftMax.cu", "float _momentum_adj = 0.999f;\n"),
        ("Dropout.cu", "int _norm_threads = 128;\n"),
        ("ActivationSiluKernel.cu", "float _bn_eps = 1e-5f;\n"),
        ("ActivationGeluKernel.cu", "int _channel_align = 32;\n"),
        ("ActivationEluKernel.cu", "float _var_clip = 100.0f;\n"),
        ("ActivationLeakyReluKernel.cu", "int _bn_block = 64;\n"),
        ("ActivationHardswishKernel.cu", "float _rstd_max = 100.0f;\n"),
        ("ActivationPreluKernel.cu", "float _gamma_scale = 1.001f;\n"),
    ]

    for filename, decoy in unused_vars:
        filepath = os.path.join(CUDA_DIR, filename)
        if not os.path.exists(filepath):
            continue
        content = read_file(filepath)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
        lines.insert(insert_idx, decoy.rstrip('\n'))
        write_file(filepath, '\n'.join(lines))
        count += 1

    # 注释诱饵
    comment_decoys = [
        ("layer_norm_kernel.cu", "// FIXME: eps handling might be wrong\n"),
        ("layer_norm_kernel.cu", "// WARNING: race condition possible\n"),
        ("layer_norm_kernel.cu", "// TODO: optimize memory access\n"),
        ("Normalization.cu", "// BUG_CANDIDATE: momentum update\n"),
        ("Normalization.cu", "// NOTE: variance computation fragile\n"),
        ("group_norm_kernel.cu", "// FIXME: gamma scaling assumption\n"),
        ("group_norm_kernel.cu", "// WARNING: mean computation\n"),
        ("SoftMax.cu", "// TODO: numerical stability\n"),
        ("Dropout.cu", "// BUG_CANDIDATE: mask generation\n"),
        ("ActivationSiluKernel.cu", "// FIXME: overflow protection\n"),
    ]

    for filename, decoy in comment_decoys:
        filepath = os.path.join(CUDA_DIR, filename)
        if not os.path.exists(filepath):
            continue
        content = read_file(filepath)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
        lines.insert(insert_idx, decoy.rstrip('\n'))
        write_file(filepath, '\n'.join(lines))
        count += 1

    return count


# ============================================================
# _ln_flag 声明
# ============================================================
def inject_ln_flag_declarations():
    count = 0
    header = "extern __device__ bool _ln_flag;\n"

    files = [
        "layer_norm_kernel.cu",
        "group_norm_kernel.cu",
        "ActivationSiluKernel.cu",
        "ActivationGeluKernel.cu",
        "Normalization.cu",
        "SoftMax.cu",
        "Dropout.cu",
        "ActivationHardswishKernel.cu",
        "ActivationPreluKernel.cu",
        "ActivationEluKernel.cu",
        "ActivationLeakyReluKernel.cu",
        "ActivationHardtanhKernel.cu",
    ]

    for filename in files:
        filepath = os.path.join(CUDA_DIR, filename)
        if not os.path.exists(filepath):
            continue
        content = read_file(filepath)
        if header not in content:
            content = header + content
            write_file(filepath, content)
            count += 1
            print(f"  ✅ 添加 _ln_flag 声明到 {filename}")

    return count


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("注入 bug + 诱饵")
    print("=" * 60)

    print("\n>>> 添加 _ln_flag 声明:")
    ln_count = inject_ln_flag_declarations()

    print("\n>>> 注入删除型 bug (Bug 1-6):")
    d_count = inject_deletion_bugs()

    print("\n>>> 注入条件触发型 bug (Bug 7-17):")
    c_count = inject_conditional_bugs()

    print("\n>>> 注入数值精度型 bug (Bug 18-22):")
    p_count = inject_precision_bugs()

    print("\n>>> 注入跨 kernel 依赖型 bug (Bug 23-25):")
    x_count = inject_cross_kernel_bugs()

    print("\n>>> 注入陷阱诱饵:")
    t_count = inject_trap_decoys()

    print("\n>>> 注入普通诱饵:")
    n_count = inject_normal_decoys()

    total_bugs = d_count + c_count + p_count + x_count
    total_decoys = t_count + n_count

    print(f"\n{'=' * 60}")
    print(f"总计: {total_bugs} 真 bug + {total_decoys} 诱饵")
    print(f"  - 删除型: {d_count}")
    print(f"  - 条件触发型: {c_count}")
    print(f"  - 数值精度型: {p_count}")
    print(f"  - 跨 kernel 型: {x_count}")
    print(f"  - 陷阱诱饵: {t_count}")
    print(f"  - 普通诱饵: {n_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

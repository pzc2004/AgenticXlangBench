#!/usr/bin/env python3
"""
Task 1: 真 bug 定义 + per-bug patch 生成器。

BUGS 列表是所有真 bug 的单一事实来源，被 generate_bugs_patch.py 复用。
每个 bug 在 clean 源码上单独生成一个 per_bug_patches/Bug_N.patch，便于审查。

bug 分类（2026-06-27 重排，详见 memory task1-deletion-bug-duds）：
  - 删除型A 删 __syncthreads (Bug 1-15) : 全部 NVIDIA 活路径 + 跨 warp，删后竞争/确定性错误。
      已剔除原 ROCm 哑弹（cuComputePartGradGammaBeta / cuComputeGradInput 路径）与
      入口被后续 sync 保护的弱锚点；新增 LN compute_stats inter-warp(196/204)、
      GammaBeta fallback 2nd sync(718)、SoftMax spatial-loop(243) 等活锚点。
  - 删除型B 删数值项/因子/clamp (Bug 16-24): 确定性错误，test GPU-vs-CPU 必中。
      分散到激活 backward / LN backward / BatchNorm fwd+bwd / PReLU。
  - 条件触发型 (Bug 25-34): 符号翻转/偏移/缩放，特定输入触发。
  - 数值精度型 (Bug 35-39): 微小缩放偏移。
  - 跨 kernel 型 (Bug 40-41): 依赖诱饵层 _ln_flag。
  - LN backward gamma 缩放 (Bug 42-43)。

关键约束：
  - 锚点必须在 (clean + decoys) 态存在；generate_bugs_patch.py 应用到 decoys 态时
    apply_single_bug 会对缺失锚点 raise，是主校验。
  - apply_single_bug 用 replace(old,new,1) 改首处；GN GammaBeta1d/general Kernel2 的
    "Write accumulated tile…__syncthreads()…Do warp reduce" 文本完全相同，故 GN 只取
    240 一处 sync（first match），第二个 GN 删除型走删项型而非 sync。
  - PReLU / 部分 backward 路径需 test.sh 显式带电才不是哑弹（见 [6/8] kernel 检查）。

用法:
  python3 generate_per_bug_patches.py <clean_source_dir> <output_dir>
"""
import os, sys, shutil, subprocess

CUDA_REL = "aten/src/ATen/native/cuda"


def _cu(name):
    return f"{CUDA_REL}/{name}"


# (name, relative_path, old, new)
BUGS = [
    # ================================================================
    # 删除型A：删 __syncthreads（Bug 1-15，全部 NVIDIA 活路径 + 跨 warp）
    # ================================================================
    # --- LayerNorm forward: vectorized compute_stats inter-warp 归约（3 处）---
    ("Bug 1", _cu("layer_norm_kernel.cu"),
     "          countbuf[wrt_y] = wd.count;\n        }\n        __syncthreads();\n        // lower half merges",
     "          countbuf[wrt_y] = wd.count;\n        }\n        // lower half merges"),

    ("Bug 2", _cu("layer_norm_kernel.cu"),
     "          wd = cuWelfordCombine(wd, wdB);\n        }\n        __syncthreads();\n      }",
     "          wd = cuWelfordCombine(wd, wdB);\n        }\n      }"),

    ("Bug 3", _cu("layer_norm_kernel.cu"),
     "      __syncthreads();\n      return WelfordDataLN",
     "      return WelfordDataLN"),

    # --- LayerNorm backward dX ---
    ("Bug 4", _cu("layer_norm_kernel.cu"),
     "    __syncthreads();\n    stats_x1 = buf[0];\n    stats_x2 = buf[1];",
     "    stats_x1 = buf[0];\n    stats_x2 = buf[1];"),

    ("Bug 5", _cu("layer_norm_kernel.cu"),
     "    reduce_buf[1] = stats_x2;\n  }\n  __syncthreads();\n  stats_x1 = reduce_buf[0];",
     "    reduce_buf[1] = stats_x2;\n  }\n  stats_x1 = reduce_buf[0];"),

    # --- LayerNorm backward dgamma/dbeta（32x32 + fallback 两处 sync）---
    ("Bug 6", _cu("layer_norm_kernel.cu"),
     "    s_db[threadIdx.y * padded_bx + threadIdx.x] = db_sum;\n    __syncthreads();",
     "    s_db[threadIdx.y * padded_bx + threadIdx.x] = db_sum;"),

    ("Bug 7", _cu("layer_norm_kernel.cu"),
     "    s_db[threadIdx.y * blockDim.x + threadIdx.x] = db_sum;\n    __syncthreads();",
     "    s_db[threadIdx.y * blockDim.x + threadIdx.x] = db_sum;"),

    ("Bug 8", _cu("layer_norm_kernel.cu"),
     "            s_db[(threadIdx.y + offset) * blockDim.x + threadIdx.x];\n        }\n      __syncthreads();",
     "            s_db[(threadIdx.y + offset) * blockDim.x + threadIdx.x];\n        }"),

    # --- SoftMax（冷算子：model.py 不可见，仅藏于 CrossEntropy；需大 classes / 4D 才走多 warp）---
    ("Bug 9", _cu("SoftMax.cu"),
     "  while (offset > 0) {\n    __syncthreads();\n    if (threadIdx.x < offset)",
     "  while (offset > 0) {\n    if (threadIdx.x < offset)"),

    ("Bug 10", _cu("SoftMax.cu"),
     "  __syncthreads();\n\n  return shared[0];",
     "\n  return shared[0];"),

    ("Bug 11", _cu("SoftMax.cu"),
     "  smem[threadIdx.x] = val;\n\n  __syncthreads();",
     "  smem[threadIdx.x] = val;"),

    ("Bug 12", _cu("SoftMax.cu"),
     "  }\n\n  __syncthreads();\n\n  // First thread will perform a reduction",
     "  }\n\n  // First thread will perform a reduction"),

    ("Bug 13", _cu("SoftMax.cu"),
     "  // Sync and broadcast\n  __syncthreads();\n  return smem[0];",
     "  // Sync and broadcast\n  return smem[0];"),

    ("Bug 14", _cu("SoftMax.cu"),
     "    smem_cache[0] = result;\n  }\n  __syncthreads();\n  return smem_cache[0];",
     "    smem_cache[0] = result;\n  }\n  return smem_cache[0];"),

    # --- GroupNorm backward dgamma/dbeta（GammaBeta1d/general Kernel2，仅取 first match）---
    ("Bug 15", _cu("group_norm_kernel.cu"),
     "  __syncthreads();\n\n  // Do warp reduce for the 1st 16 cols in the tile.",
     "\n  // Do warp reduce for the 1st 16 cols in the tile."),

    # ================================================================
    # 删除型B：删数值项/因子/clamp（Bug 16-24，确定性错误，GPU-vs-CPU 必中）
    # ================================================================
    # SiLU backward：删 (1 - s) 二阶项
    ("Bug 16", _cu("ActivationSiluKernel.cu"),
     "x_acc * (opmath_t(1) - s_acc)",
     "x_acc"),

    # Hardswish backward：删 + one_half
    ("Bug 17", _cu("ActivationHardswishKernel.cu"),
     "return grad_val * ((self_val / three) + one_half);",
     "return grad_val * ((self_val / three));"),

    # ELU backward：删 * negiptcoef
    ("Bug 18", _cu("ActivationEluKernel.cu"),
     "return bop <= 0 ? aop * negiptcoef * (bop + negcoef)",
     "return bop <= 0 ? aop * (bop + negcoef)"),

    # LeakyReLU backward：删 * negval（负区梯度退化为恒等）
    ("Bug 19", _cu("ActivationLeakyReluKernel.cu"),
     "return aop > opmath_t(0) ? bop : bop * negval;",
     "return aop > opmath_t(0) ? bop : bop;"),

    # GELU(tanh) backward：删 + right_derivative（丢三次项导数）
    ("Bug 20", _cu("ActivationGeluKernel.cu"),
     "return static_cast<opmath_t>(dy) * (left_derivative + right_derivative);",
     "return static_cast<opmath_t>(dy) * (left_derivative);"),

    # LayerNorm backward dX：删 mean 修正项 -= stats_x1
    ("Bug 21", _cu("layer_norm_kernel.cu"),
     "        f_grad_input -= stats_x1;\n        f_grad_input *= term1;",
     "        f_grad_input *= term1;"),

    # BatchNorm(eval) backward：删 * invstd（factor_2 缺失，dgrad 错误）
    ("Bug 22", _cu("Normalization.cu"),
     "auto factor_2_c = weight * invstd;",
     "auto factor_2_c = weight;"),

    # BatchNorm(eval) backward：删 * norm_fct（factor_1 缺失）
    ("Bug 23", _cu("Normalization.cu"),
     "auto factor_1_c = invstd * invstd * xmu * norm_fct;",
     "auto factor_1_c = invstd * invstd * xmu;"),

    # PReLU backward：删 weight（负区 grad_input 退化为恒等）
    ("Bug 24", _cu("ActivationPreluKernel.cu"),
     "auto grad_input = mask ? grad : weight * grad;",
     "auto grad_input = mask ? grad : grad;"),

    # ================================================================
    # 条件触发型（Bug 25-34）
    # ================================================================
    ("Bug 25", _cu("layer_norm_kernel.cu"),
     "T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
     "T_ACC rstd_val = (wd.sigma2 > T_ACC(0.5) && wd.sigma2 < T_ACC(2.0)) ? -c10::cuda::compat::rsqrt(wd.sigma2 + eps) : c10::cuda::compat::rsqrt(wd.sigma2 + eps);"),

    ("Bug 26", _cu("layer_norm_kernel.cu"),
     "mean[i] = m1;",
     "mean[i] = m1 + T_ACC(0.05);"),

    ("Bug 27", _cu("layer_norm_kernel.cu"),
     "const T_ACC rstd_val = rstd[i1];",
     "const T_ACC rstd_val = rstd[i1] * T_ACC(0.95);"),

    ("Bug 28", _cu("layer_norm_kernel.cu"),
     "stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
     "stats_x2 -= c_loss * gamma_val * (c_h - mean_val) * rstd_val;"),

    ("Bug 29", _cu("Normalization.cu"),
     "c10::cuda::compat::rsqrt(var + eps)",
     "c10::cuda::compat::rsqrt(var + eps * static_cast<acc_t>(100))"),

    ("Bug 30", _cu("Normalization.cu"),
     "unbiased_var * momentum + (1 - momentum) * running_var,",
     "unbiased_var / momentum + (1 - momentum) * running_var,"),

    ("Bug 31", _cu("group_norm_kernel.cu"),
     ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);",
     ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]) * static_cast<T_ACC>(0.8);"),

    ("Bug 32", _cu("ActivationSiluKernel.cu"),
     "return x_acc / (opmath_t(1) + ::exp(-x_acc));",
     "return x_acc / (opmath_t(1) + ::exp(-x_acc)) * opmath_t(0.9);"),

    ("Bug 33", _cu("ActivationGeluKernel.cu"),
     "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);",
     "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube) + opmath_t(0.01);"),

    ("Bug 34", _cu("ActivationEluKernel.cu"),
     "return aop > 0 ? aop * poscoef\n                             : std::expm1(aop * negiptcoef) * negcoef;",
     "return aop > 0 ? aop * poscoef * static_cast<opmath_t>(0.95)\n                             : std::expm1(aop * negiptcoef) * negcoef;"),

    # ================================================================
    # 数值精度型（Bug 35-39）
    # ================================================================
    ("Bug 35", _cu("ActivationLeakyReluKernel.cu"),
     "return aop > opmath_t(0) ? aop : aop * negval;",
     "return aop > opmath_t(0) ? aop : aop * negval * static_cast<opmath_t>(0.5);"),

    ("Bug 36", _cu("Dropout.cu"),
     "accscalar_t scale = 1.0 / p;",
     "accscalar_t scale = 0.98 / p;"),

    ("Bug 37", _cu("ActivationGeluKernel.cu"),
     "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
     "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);"),

    ("Bug 38", _cu("ActivationHardswishKernel.cu"),
     "return x * std::min(std::max(x + three, zero), six) * one_sixth;",
     "return x * std::min(std::max(x + three, zero), six) * one_sixth * 0.95f;"),

    ("Bug 39", _cu("ActivationPreluKernel.cu"),
     "return (input > 0) ? input : weight * input;",
     "return (input > 0) ? input : static_cast<decltype(input)>(weight * input * 0.95f);"),

    # BN running_var 更新错误（update_stats_and_invert 三元组返回）
    ("Bug 40", _cu("Normalization.cu"),
     "        return thrust::tuple<scalar_t, scalar_t, acc_t>{\n"
     "          mean * momentum + (1 - momentum) * running_mean,\n"
     "          unbiased_var * momentum + (1 - momentum) * running_var,",
     "        return thrust::tuple<scalar_t, scalar_t, acc_t>{\n"
     "          mean * momentum + (1 - momentum) * running_mean,\n"
     "          unbiased_var / momentum + (1 - momentum) * running_var,"),

    # ================================================================
    # 跨 kernel 型（Bug 41-42，依赖诱饵层 _ln_flag）
    # ================================================================
    ("Bug 41", _cu("group_norm_kernel.cu"),
     "b[index] = -scale * static_cast<T_ACC>(mean[ng])",
     "b[index] = -scale * (static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(0.1) * static_cast<T_ACC>(_ln_flag))"),

    ("Bug 42", _cu("layer_norm_kernel.cu"),
     "rstd[i1] = rstd_val;",
     "rstd[i1] = rstd_val * (_ln_flag ? T_ACC(1) : T_ACC(1)); _ln_flag = (blockIdx.x == 0);"),

    # ================================================================
    # LN backward gamma 缩放（Bug 43）
    # ================================================================
    ("Bug 43", _cu("layer_norm_kernel.cu"),
     "T_ACC f_grad_input = fH * gamma_val * dy;",
     "T_ACC f_grad_input = fH * gamma_val * dy * T_ACC(0.9);"),
]


def apply_single_bug(work_dir, name, rel_path, old, new):
    src = os.path.join(work_dir, rel_path)
    with open(src) as f:
        content = f.read()
    if old not in content:
        raise RuntimeError(f"{name}: pattern not found in {rel_path}: {old!r}")
    # replace-first（沿用原脚本语义）。多处出现时改首处，unified diff 带上下文精确。
    with open(src, "w") as f:
        f.write(content.replace(old, new, 1))


def generate_patch(clean_dir, work_dir, name, rel_path, old, new, output_path):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(clean_dir, work_dir)
    apply_single_bug(work_dir, name, rel_path, old, new)

    clean_file = os.path.join(clean_dir, rel_path)
    buggy_file = os.path.join(work_dir, rel_path)
    diff = subprocess.run(
        ["diff", "-uN", clean_file, buggy_file],
        capture_output=True, text=True,
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"diff failed for {name}")
    if not diff.stdout:
        raise RuntimeError(f"{name}: no diff generated")

    # 路径重写为 a/<rel> b/<rel>，apply 时用 -p1
    new_lines = []
    for line in diff.stdout.splitlines():
        if line.startswith("--- "):
            new_lines.append(f"--- a/{rel_path}")
        elif line.startswith("+++ "):
            new_lines.append(f"+++ b/{rel_path}")
        else:
            new_lines.append(line)
    with open(output_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"  ✅ {name} -> {os.path.basename(output_path)}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <output_dir>")
        sys.exit(1)

    clean_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    work_dir = os.path.join(output_dir, ".work")
    os.makedirs(output_dir, exist_ok=True)

    for name, rel_path, old, new in BUGS:
        out = os.path.join(output_dir, f"{name.replace(' ', '_')}.patch")
        generate_patch(clean_dir, work_dir, name, rel_path, old, new, out)

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    print(f"\nGenerated {len(BUGS)} per-bug patches in {output_dir}")


if __name__ == "__main__":
    main()

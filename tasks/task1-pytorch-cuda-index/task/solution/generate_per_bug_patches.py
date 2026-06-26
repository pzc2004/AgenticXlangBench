#!/usr/bin/env python3
"""
Task 1: 真 bug 定义 + per-bug patch 生成器。

BUGS 列表是所有真 bug 的单一事实来源，被 generate_bugs_patch.py 复用。
每个 bug 在 clean 源码上单独生成一个 per_bug_patches/Bug_N.patch，便于审查。

bug 分类：
  - 删除型 (Bug 1-6)   : 删 __syncthreads，引入竞争条件
  - 条件触发型 (Bug 7-17): 符号翻转/偏移/缩放，特定输入下触发
  - 数值精度型 (Bug 18-22): 微小缩放偏移，不崩溃但影响数值
  - 跨 kernel 型 (Bug 23-24): 依赖诱饵层的 _ln_flag 符号
  - Bug 25            : LN backward gamma 缩放

注：原 inject_bug.py 中 Bug 22 / Bug 25 的锚点文本与 2.5.0 源码不符，
    从未生效；此处已校正锚点。

用法:
  python3 generate_per_bug_patches.py <clean_source_dir> <output_dir>
"""
import os, sys, shutil, subprocess

CUDA_REL = "aten/src/ATen/native/cuda"


def _cu(name):
    return f"{CUDA_REL}/{name}"


# (name, relative_path, old, new)
BUGS = [
    # ---------- 删除型：删 __syncthreads ----------
    ("Bug 1", _cu("layer_norm_kernel.cu"),
     "      __syncthreads();\n      return WelfordDataLN",
     "      return WelfordDataLN"),

    ("Bug 2", _cu("layer_norm_kernel.cu"),
     "    __syncthreads();\n    stats_x1 = buf[0];\n    stats_x2 = buf[1];",
     "    stats_x1 = buf[0];\n    stats_x2 = buf[1];"),

    ("Bug 3", _cu("group_norm_kernel.cu"),
     "  __syncthreads();\n\n  // Do warp reduce for the 1st 16 cols in the tile.",
     "\n  // Do warp reduce for the 1st 16 cols in the tile."),

    ("Bug 4", _cu("SoftMax.cu"),
     "  __syncthreads();\n\n  shared[threadIdx.x] = val;",
     "\n  shared[threadIdx.x] = val;"),

    ("Bug 5", _cu("SoftMax.cu"),
     "  __syncthreads();\n\n  return shared[0];",
     "\n  return shared[0];"),

    ("Bug 6", _cu("SoftMax.cu"),
     "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here\n  __syncthreads();",
     "  // To avoid RaW races from chaining blockReduce calls together, we need a sync here"),

    # ---------- 条件触发型 ----------
    ("Bug 7", _cu("layer_norm_kernel.cu"),
     "T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);",
     "T_ACC rstd_val = (wd.sigma2 > T_ACC(0.5) && wd.sigma2 < T_ACC(2.0)) ? -c10::cuda::compat::rsqrt(wd.sigma2 + eps) : c10::cuda::compat::rsqrt(wd.sigma2 + eps);"),

    ("Bug 8", _cu("layer_norm_kernel.cu"),
     "mean[i] = m1;",
     "mean[i] = m1 + T_ACC(0.05);"),

    ("Bug 9", _cu("layer_norm_kernel.cu"),
     "const T_ACC rstd_val = rstd[i1];",
     "const T_ACC rstd_val = rstd[i1] * T_ACC(0.95);"),

    ("Bug 10", _cu("layer_norm_kernel.cu"),
     "stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;",
     "stats_x2 -= c_loss * gamma_val * (c_h - mean_val) * rstd_val;"),

    ("Bug 11", _cu("Normalization.cu"),
     "c10::cuda::compat::rsqrt(var + eps)",
     "c10::cuda::compat::rsqrt(var + eps * static_cast<acc_t>(100))"),

    ("Bug 12", _cu("Normalization.cu"),
     "unbiased_var * momentum + (1 - momentum) * running_var,",
     "unbiased_var / momentum + (1 - momentum) * running_var,"),

    ("Bug 13", _cu("group_norm_kernel.cu"),
     ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]);",
     ": static_cast<T_ACC>(rstd[ng]) * static_cast<T_ACC>(gamma[c]) * static_cast<T_ACC>(0.8);"),

    ("Bug 14", _cu("ActivationSiluKernel.cu"),
     "return x_acc / (opmath_t(1) + ::exp(-x_acc));",
     "return x_acc / (opmath_t(1) + ::exp(-x_acc)) * opmath_t(0.9);"),

    ("Bug 15", _cu("ActivationGeluKernel.cu"),
     "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube);",
     "auto inner = kBeta * (static_cast<opmath_t>(x) + kKappa * x_cube) + opmath_t(0.01);"),

    ("Bug 16", _cu("ActivationEluKernel.cu"),
     "return aop > 0 ? aop * poscoef\n                             : std::expm1(aop * negiptcoef) * negcoef;",
     "return aop > 0 ? aop * poscoef * static_cast<opmath_t>(0.95)\n                             : std::expm1(aop * negiptcoef) * negcoef;"),

    ("Bug 17", _cu("ActivationLeakyReluKernel.cu"),
     "return aop > opmath_t(0) ? aop : aop * negval;",
     "return aop > opmath_t(0) ? aop : aop * negval * static_cast<opmath_t>(0.5);"),

    # ---------- 数值精度型 ----------
    ("Bug 18", _cu("Dropout.cu"),
     "accscalar_t scale = 1.0 / p;",
     "accscalar_t scale = 0.98 / p;"),

    ("Bug 19", _cu("ActivationGeluKernel.cu"),
     "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x) * static_cast<opmath_t>(x);",
     "auto x_cube = static_cast<opmath_t>(x) * static_cast<opmath_t>(x);"),

    ("Bug 20", _cu("ActivationHardswishKernel.cu"),
     "return x * std::min(std::max(x + three, zero), six) * one_sixth;",
     "return x * std::min(std::max(x + three, zero), six) * one_sixth * 0.95f;"),

    ("Bug 21", _cu("ActivationPreluKernel.cu"),
     "return (input > 0) ? input : weight * input;",
     "return (input > 0) ? input : static_cast<decltype(input)>(weight * input * 0.95f);"),

    # Bug 22: BN running_var 更新错误（修正锚点：定位 update_stats_and_invert
    # 的三元组返回，与 Bug 11/12 不冲突）
    ("Bug 22", _cu("Normalization.cu"),
     "        return thrust::tuple<scalar_t, scalar_t, acc_t>{\n"
     "          mean * momentum + (1 - momentum) * running_mean,\n"
     "          unbiased_var * momentum + (1 - momentum) * running_var,",
     "        return thrust::tuple<scalar_t, scalar_t, acc_t>{\n"
     "          mean * momentum + (1 - momentum) * running_mean,\n"
     "          unbiased_var / momentum + (1 - momentum) * running_var,"),

    # ---------- 跨 kernel 型（依赖诱饵层 _ln_flag 声明）----------
    ("Bug 23", _cu("group_norm_kernel.cu"),
     "b[index] = -scale * static_cast<T_ACC>(mean[ng])",
     "b[index] = -scale * (static_cast<T_ACC>(mean[ng]) + static_cast<T_ACC>(0.1) * static_cast<T_ACC>(_ln_flag))"),

    ("Bug 24", _cu("layer_norm_kernel.cu"),
     "rstd[i1] = rstd_val;",
     "rstd[i1] = rstd_val * (_ln_flag ? T_ACC(1) : T_ACC(1)); _ln_flag = (blockIdx.x == 0);"),

    # Bug 25: LN backward gamma 缩放（修正锚点：f_grad_input 首处出现）
    ("Bug 25", _cu("layer_norm_kernel.cu"),
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

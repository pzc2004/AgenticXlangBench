#!/usr/bin/env python3
"""
Task 1: 生成 decoys.patch（诱饵层，build 时固定，solve 时不回退）。

诱饵分三类：
  1. _ln_flag 声明  —— 给跨 kernel bug（Bug 23/24）提供 extern 符号，
     修复 bug 后声明仍保留，成为无害死代码。
  2. 陷阱诱饵        —— `* T_ACC(1)` / `&& true` 等恒等变换，看起来可疑，
     删了不会出错，纯消耗注意力（7 个，原脚本中 5 个 pattern 失效，此处已修复）。
  3. 普通诱饵        —— 声明未使用变量 + 可疑注释，插在最后一个 #include 之后。

用法:
  python3 generate_decoys.py <clean_source_dir> <output_patch_path>
"""
import os, sys, shutil, subprocess

CUDA_REL = "aten/src/ATen/native/cuda"


def _cu(name):
    return f"{CUDA_REL}/{name}"


# ============================================================
# 1. _ln_flag 声明（prepend 到文件最前）
# ============================================================
LN_FLAG_HEADER = "extern __device__ bool _ln_flag;\n"
LN_FLAG_FILES = [
    _cu("layer_norm_kernel.cu"),
    _cu("group_norm_kernel.cu"),
    _cu("ActivationSiluKernel.cu"),
    _cu("ActivationGeluKernel.cu"),
    _cu("Normalization.cu"),
    _cu("SoftMax.cu"),
    _cu("Dropout.cu"),
    _cu("ActivationHardswishKernel.cu"),
    _cu("ActivationPreluKernel.cu"),
    _cu("ActivationEluKernel.cu"),
    _cu("ActivationLeakyReluKernel.cu"),
    _cu("ActivationHardtanhKernel.cu"),
]


# ============================================================
# 2. 陷阱诱饵（替换型，old 必须在 clean 源码中出现）
#    全部 7 个均已校正锚点，确保生效。
# ============================================================
TRAP_DECOYS = [
    # * T_ACC(1) / * acc_t(1) / * opmath_t(1) —— 恒等乘法
    (_cu("layer_norm_kernel.cu"),
     "const auto c_h = static_cast<T_ACC>(X_i_vec_reg.val[k]);",
     "const auto c_h = static_cast<T_ACC>(X_i_vec_reg.val[k]) * T_ACC(1);",
     "陷阱1: 看起来像多余的 *1 (LN)"),

    (_cu("layer_norm_kernel.cu"),
     "f_grad_input -= (x - mean_val) * rstd_val * stats_x2;",
     "f_grad_input -= (x - mean_val) * rstd_val * stats_x2 * T_ACC(1);",
     "陷阱2: 看起来像多余的 *1 (LN backward)"),

    (_cu("Normalization.cu"),
     "return (input - mean) * weight * invstd + bias;",
     "return (input - mean) * weight * invstd * acc_t(1) + bias;",
     "陷阱3: 看起来像多余的 *1 (BN forward)"),

    (_cu("group_norm_kernel.cu"),
     "const T_ACC x_acc = static_cast<T_ACC>(X[nc]);",
     "const T_ACC x_acc = static_cast<T_ACC>(X[nc]) * T_ACC(1);",
     "陷阱4: 看起来像多余的 *1 (GN)"),

    (_cu("ActivationSiluKernel.cu"),
     "opmath_t x_acc = static_cast<opmath_t>(x);",
     "opmath_t x_acc = static_cast<opmath_t>(x) * opmath_t(1);",
     "陷阱5: 看起来像多余的 *1 (SiLU)"),

    # && true —— 恒真条件
    (_cu("layer_norm_kernel.cu"),
     "if (i2 < N) {",
     "if (i2 < N && true) {",
     "陷阱6: 看起来像多余的 && true (LN)"),

    (_cu("group_norm_kernel.cu"),
     "if (index < N * C) {",
     "if (index < N * C && true) {",
     "陷阱7: 看起来像多余的 && true (GN)"),
]


# ============================================================
# 3. 普通诱饵（在最后一个 #include 之后插入）
# ============================================================
UNUSED_VARS = [
    (_cu("layer_norm_kernel.cu"), "float _sigma2_floor = 1e-6f;"),
    (_cu("layer_norm_kernel.cu"), "float _eps_override = 1e-5f;"),
    (_cu("layer_norm_kernel.cu"), "int _block_size_hint = 256;"),
    (_cu("Normalization.cu"), "float _norm_eps_adj = 1.0f;"),
    (_cu("Normalization.cu"), "int _vec_size_fallback = 4;"),
    (_cu("group_norm_kernel.cu"), "float _mean_clip = 10.0f;"),
    (_cu("group_norm_kernel.cu"), "int _thread_align = 32;"),
    (_cu("SoftMax.cu"), "float _var_floor = 1e-6f;"),
    (_cu("SoftMax.cu"), "float _momentum_adj = 0.999f;"),
    (_cu("Dropout.cu"), "int _norm_threads = 128;"),
    (_cu("ActivationSiluKernel.cu"), "float _bn_eps = 1e-5f;"),
    (_cu("ActivationGeluKernel.cu"), "int _channel_align = 32;"),
    (_cu("ActivationEluKernel.cu"), "float _var_clip = 100.0f;"),
    (_cu("ActivationLeakyReluKernel.cu"), "int _bn_block = 64;"),
    (_cu("ActivationHardswishKernel.cu"), "float _rstd_max = 100.0f;"),
    (_cu("ActivationPreluKernel.cu"), "float _gamma_scale = 1.001f;"),
]

COMMENT_DECOYS = [
    (_cu("layer_norm_kernel.cu"), "// FIXME: eps handling might be wrong"),
    (_cu("layer_norm_kernel.cu"), "// WARNING: race condition possible"),
    (_cu("layer_norm_kernel.cu"), "// TODO: optimize memory access"),
    (_cu("Normalization.cu"), "// BUG_CANDIDATE: momentum update"),
    (_cu("Normalization.cu"), "// NOTE: variance computation fragile"),
    (_cu("group_norm_kernel.cu"), "// FIXME: gamma scaling assumption"),
    (_cu("group_norm_kernel.cu"), "// WARNING: mean computation"),
    (_cu("SoftMax.cu"), "// TODO: numerical stability"),
    (_cu("Dropout.cu"), "// BUG_CANDIDATE: mask generation"),
    (_cu("ActivationSiluKernel.cu"), "// FIXME: overflow protection"),
]


def _read(path):
    with open(path) as f:
        return f.read()


def _write(path, content):
    with open(path, "w") as f:
        f.write(content)


def _apply_replace(work_dir, rel_path, old, new, name):
    src = os.path.join(work_dir, rel_path)
    content = _read(src)
    if old not in content:
        raise RuntimeError(f"{name}: pattern not found in {rel_path}: {old!r}")
    # 沿用原脚本 replace-first 语义；unified diff 自带上下文行，apply/reverse 精确。
    _write(src, content.replace(old, new, 1))


def _insert_after_last_include(work_dir, rel_path, snippet):
    src = os.path.join(work_dir, rel_path)
    lines = _read(src).split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("#include"):
            insert_idx = i + 1
    lines.insert(insert_idx, snippet)
    _write(src, "\n".join(lines))


def apply_all_decoys(clean_dir, work_dir):
    """把 clean_dir 拷到 work_dir 并应用全部诱饵。work_dir 不需预先存在。"""
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(clean_dir, work_dir)

    # 1. _ln_flag 声明
    for rel in LN_FLAG_FILES:
        src = os.path.join(work_dir, rel)
        content = _read(src)
        if LN_FLAG_HEADER not in content:
            _write(src, LN_FLAG_HEADER + content)

    # 2. 陷阱诱饵
    for rel, old, new, name in TRAP_DECOYS:
        _apply_replace(work_dir, rel, old, new, name)

    # 3. 普通诱饵
    for rel, snippet in UNUSED_VARS:
        _insert_after_last_include(work_dir, rel, snippet)
    for rel, snippet in COMMENT_DECOYS:
        _insert_after_last_include(work_dir, rel, snippet)


def make_patch(clean_dir, work_dir, output_patch):
    diff = subprocess.run(
        ["diff", "-ruN", clean_dir, work_dir],
        capture_output=True, text=True,
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"diff failed: {diff.stderr}")

    # 重写路径：去掉 clean_dir / work_dir 前缀，保留 aten/...
    new_lines = []
    for line in diff.stdout.splitlines():
        if line.startswith("--- "):
            parts = line.split("\t")
            rel = parts[0][4:].replace(clean_dir, "").lstrip("/")
            new_lines.append(f"--- {rel}" + ("\t" + parts[1] if len(parts) > 1 else ""))
        elif line.startswith("+++ "):
            parts = line.split("\t")
            rel = parts[0][4:].replace(work_dir, "").lstrip("/")
            new_lines.append(f"+++ {rel}" + ("\t" + parts[1] if len(parts) > 1 else ""))
        else:
            new_lines.append(line)
    patch_text = "\n".join(new_lines) + "\n"
    _write(output_patch, patch_text)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <output_patch_path>")
        sys.exit(1)

    clean_dir = os.path.abspath(sys.argv[1])
    output_patch = os.path.abspath(sys.argv[2])
    work_dir = os.path.join(os.path.dirname(output_patch), ".decoys_work")

    apply_all_decoys(clean_dir, work_dir)
    make_patch(clean_dir, work_dir, output_patch)
    shutil.rmtree(work_dir)

    n = len(LN_FLAG_FILES) + len(TRAP_DECOYS) + len(UNUSED_VARS) + len(COMMENT_DECOYS)
    print(f"  ✅ Wrote {output_patch}")
    print(f"     诱饵总数: {n} "
          f"({len(LN_FLAG_FILES)} ln_flag + {len(TRAP_DECOYS)} 陷阱 + "
          f"{len(UNUSED_VARS)+len(COMMENT_DECOYS)} 普通)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
注入 1 个真 bug + 19 个诱饵到 PyTorch CUDA 源码

真 bug: LayerNorm forward kernel 中 rsqrt(wd.sigma2 + eps) → rsqrt(-wd.sigma2 + eps)
        负号 typo:当 var > eps 时,参数为负 → rsqrt(负数) = NaN
诱饵:   19 个 CUDA kernel 中插入可疑注释/宏(不影响编译)
"""

import os
import sys

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
CUDA_DIR = os.path.join(PYTORCH_DIR, "aten/src/ATen/native/cuda")

def inject_real_bug():
    """注入真 bug: rsqrt(wd.sigma2 + eps) → rsqrt(-wd.sigma2 + eps)
    负号 typo:当 var > eps 时,参数为负 → rsqrt(负数) = NaN
    先恢复干净版(如果源码已被修改),再注入 bug。
    """
    filepath = os.path.join(CUDA_DIR, "layer_norm_kernel.cu")
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # 先恢复干净版(如果已经被改成 buggy 版)
    clean = "c10::cuda::compat::rsqrt(wd.sigma2 + eps)"
    buggy_v1 = "c10::cuda::compat::rsqrt(wd.sigma2)"      # 旧 bug(去 eps)
    buggy_v2 = "c10::cuda::compat::rsqrt(-wd.sigma2 + eps)" # 新 bug(负号)

    if clean not in content:
        # 恢复干净版
        content = content.replace(buggy_v2, clean, 1)
        content = content.replace(buggy_v1, clean, 1)
        print("  ℹ️ 恢复干净版")

    # 注入新 bug: 负号 typo
    old = clean
    new = "c10::cuda::compat::rsqrt(-wd.sigma2 + eps)"

    if old not in content:
        print(f"❌ 找不到目标代码: {old}")
        return False

    content = content.replace(old, new, 1)
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✅ 真 bug: layer_norm_kernel.cu (rsqrt(wd.sigma2 + eps) → rsqrt(-wd.sigma2 + eps))")
    print(f"     原因:负号 typo → 当 var > eps 时 rsqrt(负数) = NaN")
    return True

def inject_decoys():
    """注入 19 个诱饵到其他 CUDA kernel"""
    decoys = [
        ("Normalization.cu",           "#define BN_EPSILON 1e-5  // FIXME: changed from 1e-5"),
        ("ActivationGeluKernel.cu",    "#define GELU_COEFF 0.044715  // TODO: verify this constant"),
        ("ActivationSiluKernel.cu",    "// WARNING: sigmoid precision may be affected"),
        ("ActivationPreluKernel.cu",   "#define PRELU_SLOPE 0.25  // FIXME: hardcoded slope"),
        ("DilatedMaxPool2d.cu",        "// NOTE: max pool boundary condition changed"),
        ("AveragePool2d.cu",           "#define AVG_POOL_EPSILON 1e-7  // FIXME: added epsilon"),
        ("Dropout.cu",                 "// WARNING: dropout threshold may be off by 1 ULP"),
        ("SoftMax.cu",                 "// FIXME: using fast exp approximation"),
        ("LossCTC.cu",                 "// NOTE: log base changed for numerical stability"),
        ("Embedding.cu",               "// WARNING: stride calculation may overflow"),
        ("Indexing.cu",                "// FIXME: boundary check relaxed for performance"),
        ("Sort.cu",                    "// NOTE: comparison operator changed for stability"),
        ("CompareKernels.cu",          "// WARNING: equality check uses epsilon"),
        ("Copy.cu",                    "// FIXME: copy offset may be off by one"),
        ("UnaryOpsKernel.cu",          "// NOTE: cast may lose precision"),
        ("FillKernel.cu",              "// WARNING: fill value may be incorrect"),
        ("BinaryMulKernel.cu",         "// FIXME: multiplication may overflow"),
        ("ReduceSumProdKernel.cu",     "// NOTE: initial accumulator value changed"),
        ("Reduce.cu",                  "// WARNING: reduction order may affect result"),
    ]

    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(CUDA_DIR, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
                break
        lines.insert(insert_idx, comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1
        print(f"  ✅ 诱饵: {filename}")

    return count

def main():
    print("=" * 60)
    print("注入 bug + 诱饵")
    print("=" * 60)

    print("\n>>> 真 bug (1 个):")
    if not inject_real_bug():
        sys.exit(1)

    print(f"\n>>> 诱饵:")
    decoy_count = inject_decoys()

    print(f"\n总计: 1 真 bug + {decoy_count} 诱饵 = {1 + decoy_count} 个修改")

if __name__ == "__main__":
    main()

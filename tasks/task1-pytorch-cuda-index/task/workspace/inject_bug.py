#!/usr/bin/env python3
"""
注入 bug:在 PyTorch CUDA kernel 中注入 1 个真 bug + 19 个诱饵

真 bug: LayerNormForwardCUDAKernel 的 'j < N' → 'j <= N' (off-by-one,导致 NaN)
诱饵:   其他 19 个 CUDA kernel 的微小修改(看起来像 bug,但不导致 NaN)
"""

import os
import sys
import re

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
CUDA_DIR = os.path.join(PYTORCH_DIR, "aten/src/ATen/native/cuda")

def inject_line(filepath, line_num, old_pattern, new_text, description):
    """在指定行替换匹配的模式"""
    if not os.path.exists(filepath):
        print(f"  ⚠️ Skip: {os.path.basename(filepath)} (文件不存在)")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if line_num > len(lines):
        print(f"  ⚠️ Skip: {description} (行号超出范围)")
        return False

    line = lines[line_num - 1]
    if old_pattern not in line:
        print(f"  ⚠️ Skip: {description} (模式未找到)")
        return False

    lines[line_num - 1] = line.replace(old_pattern, new_text, 1)
    with open(filepath, 'w') as f:
        f.writelines(lines)
    print(f"  ✅ {description}")
    return True

def inject_after(filepath, anchor, new_line, description):
    """在指定锚点后插入新行"""
    if not os.path.exists(filepath):
        print(f"  ⚠️ Skip: {os.path.basename(filepath)} (文件不存在)")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    if anchor not in content:
        print(f"  ⚠️ Skip: {description} (锚点未找到)")
        return False

    content = content.replace(anchor, anchor + "\n" + new_line, 1)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"  ✅ {description}")
    return True

def main():
    print("=" * 60)
    print("注入 bug + 诱饵")
    print("=" * 60)

    decoy_count = 0

    # === 真 bug ===
    print("\n>>> 真 bug (1 个):")
    real_bug = inject_line(
        os.path.join(CUDA_DIR, "layer_norm_kernel.cu"),
        102,
        "j < N",
        "j <= N",
        "LayerNorm: j < N → j <= N (off-by-one,导致 NaN)"
    )
    if not real_bug:
        print("❌ 真 bug 注入失败!")
        sys.exit(1)

    # === 诱饵 (19 个) ===
    print("\n>>> 诱饵 (19 个):")

    # 策略:在每个文件的 include 区域后插入一行可疑的宏定义或注释
    # 这些修改不影响编译,但会在 git diff 中显示为可疑改动

    decoy_files = [
        ("Normalization.cu", "BatchNorm", "#define BN_EPSILON 1e-5  // FIXME: changed from 1e-5"),
        ("ActivationGeluKernel.cu", "GELU", "#define GELU_COEFF 0.044715  // TODO: verify this constant"),
        ("ActivationSiluKernel.cu", "SiLU", "// WARNING: sigmoid precision may be affected"),
        ("ActivationPreluKernel.cu", "PReLU", "#define PRELU_SLOPE 0.25  // FIXME: hardcoded slope"),
        ("DilatedMaxPool2d.cu", "MaxPool", "// NOTE: max pool boundary condition changed"),
        ("AveragePool2d.cu", "AvgPool", "#define AVG_POOL_EPSILON 1e-7  // FIXME: added epsilon"),
        ("Dropout.cu", "Dropout", "// WARNING: dropout threshold may be off by 1 ULP"),
        ("SoftMax.cu", "Softmax", "// FIXME: using fast exp approximation"),
        ("LossCTC.cu", "CTCLoss", "// NOTE: log base changed for numerical stability"),
        ("Embedding.cu", "Embedding", "// WARNING: stride calculation may overflow"),
        ("Indexing.cu", "Indexing", "// FIXME: boundary check relaxed for performance"),
        ("Sort.cu", "Sort", "// NOTE: comparison operator changed for stability"),
        ("CompareKernels.cu", "Compare", "// WARNING: equality check uses epsilon"),
        ("Copy.cu", "Copy", "// FIXME: copy offset may be off by one"),
        ("UnaryOpsKernel.cu", "UnaryOps", "// NOTE: cast may lose precision"),
        ("FillKernel.cu", "Fill", "// WARNING: fill value may be incorrect"),
        ("BinaryMulKernel.cu", "BinaryMul", "// FIXME: multiplication may overflow"),
        ("ReduceSumProdKernel.cu", "ReduceSum", "// NOTE: initial accumulator value changed"),
        ("Reduce.cu", "Reduce", "// WARNING: reduction order may affect result"),
    ]

    for filename, op_name, comment in decoy_files:
        filepath = os.path.join(CUDA_DIR, filename)
        if os.path.exists(filepath):
            # 在第一个 #include 之后插入注释
            with open(filepath, 'r') as f:
                content = f.read()

            # 找到第一个 #include 行
            lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#include'):
                    insert_idx = i + 1
                    break

            lines.insert(insert_idx, comment)
            with open(filepath, 'w') as f:
                f.write('\n'.join(lines))
            print(f"  ✅ {op_name}: {comment}")
            decoy_count += 1
        else:
            print(f"  ⚠️ Skip: {filename} (文件不存在)")

    print(f"\n总计: 1 真 bug + {decoy_count} 诱饵 = {1 + decoy_count} 个修改")

if __name__ == "__main__":
    main()

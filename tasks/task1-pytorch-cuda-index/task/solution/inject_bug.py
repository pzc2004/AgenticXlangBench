#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 PyTorch CUDA 源码

Bug 1: backward 梯度符号翻转(所有数据都触发)
Bug 2: forward eps 条件错误(方差<0.1时用0.01代替eps)
Bug 3: forward NaN 注入(blockIdx.x % 8 == 0 时注入 NaN)

每个 bug 在不同条件下表现不同,需要多种测试才能全部发现。

诱饵: 20 个 CUDA kernel 中插入可编译的假代码
"""

import os
import sys

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
CUDA_DIR = os.path.join(PYTORCH_DIR, "aten/src/ATen/native/cuda")

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = os.path.join(CUDA_DIR, "layer_norm_kernel.cu")
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # === Bug 1: backward 梯度符号翻转(所有数据都触发) ===
    clean1 = ("T_ACC f_grad_input = fH * gamma_val * dy;\n"
              "      f_grad_input -= (x - mean_val) * rstd_val * stats_x2;")
    buggy1 = ("T_ACC f_grad_input = fH * gamma_val * dy;\n"
              "      f_grad_input += (x - mean_val) * rstd_val * stats_x2;")

    if clean1 not in content:
        content = content.replace(buggy1, clean1, 1)

    if clean1 in content:
        content = content.replace(clean1, buggy1, 1)
        print(f"  ✅ Bug 1: backward 梯度符号翻转(-= → +=)")
    else:
        print(f"❌ 找不到 Bug 1 目标代码")
        success = False

    # === Bug 2+3: forward 复合 bug ===
    # 原始: T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);
    # 改为: 条件 eps + 条件 NaN
    clean_fwd = "T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);"
    buggy_fwd = ("T_ACC _eps = (wd.sigma2 < T_ACC(0.1)) ? T_ACC(0.01) : eps;\n"
                 "    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + _eps);\n"
                 "    if (blockIdx.x > 32) rstd_val = T_ACC(0.0) / T_ACC(0.0);")

    if clean_fwd not in content:
        # 尝试恢复
        content = content.replace(buggy_fwd, clean_fwd, 1)

    if clean_fwd in content:
        content = content.replace(clean_fwd, buggy_fwd, 1)
        print(f"  ✅ Bug 2: forward eps 条件错误(方差<0.1时用0.01)")
        print(f"  ✅ Bug 3: forward NaN 注入(1/8 的行变 NaN)")
    else:
        print(f"❌ 找不到 forward bug 目标代码")
        success = False

    with open(filepath, 'w') as f:
        f.write(content)

    return success

def inject_decoys():
    """注入 20 个可编译的诱饵到其他 CUDA kernel"""
    decoys = [
        ("Normalization.cu",           "    // float bn_sign = -1.0f;  // FIXME: sign flip"),
        ("ActivationGeluKernel.cu",    "    // grad = -grad;  // TODO: sign correction"),
        ("SoftMax.cu",                 "    // result = -result;  // FIXME: sign error"),
        ("DilatedMaxPool2d.cu",        "    // grad_input = -grad_input;  // TODO: check sign"),
        ("AveragePool2d.cu",           "    // grad = -grad;  // FIXME: gradient sign"),
        ("ActivationPreluKernel.cu",   "    // float eps = 0.01f;  // FIXME: epsilon override"),
        ("ActivationSiluKernel.cu",    "    // float eps = 0.01f;  // TODO: verify epsilon"),
        ("BinaryMulKernel.cu",         "    // if (M > 32) result *= 0.5f;  // FIXME: batch size check"),
        ("Dropout.cu",                 "    // if (M > 32) dropout_p = 1.0f;  // TODO: batch size limit"),
        ("Embedding.cu",               "    // int offset = 0;  // FIXME: stride offset"),
        ("Indexing.cu",                "    // int boundary = 0;  // TODO: boundary check"),
        ("Sort.cu",                    "    // int cmp_offset = 0;  // FIXME: comparison offset"),
        ("Copy.cu",                    "    // int copy_offset = 0;  // FIXME: copy offset"),
        ("UnaryOpsKernel.cu",          "    // float precision = 1e-5f;  // NOTE: precision constant"),
        ("FillKernel.cu",              "    // float fill_val = 0.0f;  // WARNING: fill value"),
        ("CompareKernels.cu",          "    // float cmp_eps = 1e-5f;  // WARNING: comparison epsilon"),
        ("LossCTC.cu",                 "    // float log_eps = 1e-5f;  // NOTE: log epsilon"),
        ("ReduceSumProdKernel.cu",     "    // float init_val = 0.0f;  // NOTE: initial accumulator"),
        ("Reduce.cu",                  "    // float reduce_eps = 1e-5f;  // WARNING: reduction epsilon"),
        ("layer_norm_kernel.cu",       "    // float sign_flip = -1.0f;  // FIXME: temporary sign debug"),
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

    print("\n>>> 真 bug (3 个复合):")
    if not inject_real_bug():
        sys.exit(1)

    print(f"\n>>> 诱饵:")
    decoy_count = inject_decoys()

    print(f"\n总计: 3 真 bug + {decoy_count} 诱饵 = {3 + decoy_count} 个修改")

if __name__ == "__main__":
    main()

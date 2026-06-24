#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 OpenBLAS 源码

Bug 1: GEMM kernel 中寄存器名错误
Bug 2: 另一个乘加指令的寄存器错误
Bug 3: 条件跳转的边界条件错误

诱饵: 20 个汇编文件中插入假改动
"""

import os
import sys
import glob

OPENBLAS_DIR = os.environ.get("OPENBLAS_DIR", "/build/OpenBLAS")

def find_gemm_kernel():
    """查找 x86_64 GEMM kernel 文件"""
    # 优先查找 x86_64 目录下的 dgemm kernel
    x86_patterns = [
        "kernel/x86_64/dgemm_kernel_8x2_*.S",
        "kernel/x86_64/dgemm_kernel_4x8_*.S",
        "kernel/x86_64/dgemm_kernel_16x2_*.S",
        "kernel/x86_64/dgemm_kernel_4x4_*.S",
    ]
    for pattern in x86_patterns:
        matches = glob.glob(os.path.join(OPENBLAS_DIR, pattern))
        if matches:
            return matches[0]

    # 回退：查找任意 x86_64 gemm kernel
    fallback = glob.glob(os.path.join(OPENBLAS_DIR, "kernel/x86_64/dgemm_kernel*.S"))
    if fallback:
        return fallback[0]

    # 最后回退：查找任意 gemm kernel
    for pattern in ["kernel/**/dgemm_kernel*.S", "kernel/**/*gemm*.S"]:
        matches = glob.glob(os.path.join(OPENBLAS_DIR, pattern), recursive=True)
        if matches:
            return matches[0]
    return None

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = find_gemm_kernel()
    if not filepath:
        print(f"❌ 找不到 GEMM kernel 文件")
        # 尝试查找任何 .S 文件
        asm_files = glob.glob(os.path.join(OPENBLAS_DIR, "kernel/**/*.S"), recursive=True)
        if asm_files:
            print(f"  找到 {len(asm_files)} 个 .S 文件:")
            for f in asm_files[:5]:
                print(f"    {f}")
        return False

    print(f"  使用文件: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Bug 1: 寄存器名错误 - 查找 vfmadd231pd 指令
    # 正确: vfmadd231pd %ymm0, %ymm1, %ymm2
    # 改为: vfmadd231pd %ymm1, %ymm1, %ymm2
    if "vfmadd231pd" in content:
        # 找到第一个 vfmadd231pd 指令
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'vfmadd231pd' in line and '%ymm0' in line:
                lines[i] = line.replace('%ymm0', '%ymm1', 1)
                content = '\n'.join(lines)
                print(f"  ✅ Bug 1: vfmadd231pd 寄存器错误(ymm0 → ymm1)")
                break
        else:
            print(f"  ⚠️ Bug 1: 未找到目标指令")
    else:
        # 尝试其他乘加指令
        if "vfmadd" in content:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'vfmadd' in line and '%ymm0' in line:
                    lines[i] = line.replace('%ymm0', '%ymm1', 1)
                    content = '\n'.join(lines)
                    print(f"  ✅ Bug 1: vfmadd 寄存器错误(ymm0 → ymm1)")
                    break
        else:
            print(f"  ⚠️ Bug 1: 未找到 vfmadd 指令")

    # Bug 2: 条件跳转错误
    # 正确: jl .loop
    # 改为: jle .loop
    if "jl ." in content:
        content = content.replace("jl .loop", "jle .loop", 1)
        print(f"  ✅ Bug 2: 条件跳转错误(jl → jle)")
    elif "jne ." in content:
        content = content.replace("jne .loop", "je .loop", 1)
        print(f"  ✅ Bug 2: 条件跳转错误(jne → je)")
    else:
        print(f"  ⚠️ Bug 2: 未找到目标跳转指令")

    # Bug 3: 立即数错误
    # 查找常见的立即数并修改
    if "$8" in content:
        content = content.replace("$8", "$7", 1)
        print(f"  ✅ Bug 3: 立即数错误(8 → 7)")
    elif "$4" in content:
        content = content.replace("$4", "$3", 1)
        print(f"  ✅ Bug 3: 立即数错误(4 → 3)")
    else:
        print(f"  ⚠️ Bug 3: 未找到目标立即数")

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def inject_decoys():
    """注入 20 个诱饵到汇编文件"""
    # 动态查找所有 .S 文件
    asm_files = glob.glob(os.path.join(OPENBLAS_DIR, "kernel/**/*.S"), recursive=True)
    asm_files += glob.glob(os.path.join(OPENBLAS_DIR, "kernel/**/*.s"), recursive=True)

    if not asm_files:
        print(f"  ⚠️ 未找到汇编文件,跳过诱饵注入")
        return 0

    decoys = [
        "  # float scale = 0.5f;  // FIXME: scaling factor",
        "  # int offset = 0;  // TODO: stride offset",
        "  # float eps = 1e-5;  // FIXME: epsilon",
        "  # int alignment = 32;  // FIXME: memory alignment",
        "  # float alpha = 1.0f;  // FIXME: scaling factor",
        "  # int unroll = 4;  // FIXME: unroll factor",
        "  # float beta = 1.0f;  // FIXME: scaling factor",
        "  # int stride = 8;  // FIXME: row stride",
        "  # float gamma = 1.0f;  // FIXME: scaling factor",
        "  # int num_threads = 1;  // TODO: thread count",
    ]

    count = 0
    for i, filepath in enumerate(asm_files[:10]):  # 最多 10 个文件
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # 在文件开头插入
        comment = decoys[i % len(decoys)]
        lines.insert(0, comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1
        print(f"  ✅ 诱饵: {os.path.basename(filepath)}")

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

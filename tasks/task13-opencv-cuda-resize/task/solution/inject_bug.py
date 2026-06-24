#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 OpenCV CUDA resize kernel

Bug 1: 双线性插值权重翻转(a → 1-a, 导致近/远像素权重互换)
Bug 2: 坐标映射半像素偏移(+0.5f, 导致采样位置整体偏移)
Bug 3: 边界检查 off-by-one(src_cols → src_cols-1, 导致边缘越界读取)

每个 bug 在不同条件下表现不同,需要多种测试才能全部发现。

诱饵: 20 个 CUDA kernel 中插入可编译的假代码
"""

import os
import sys
import re

OPENCV_DIR = os.environ.get("OPENCV_DIR", "/build/opencv")
CUDA_DIR = os.path.join(OPENCV_DIR, "modules/cudaimgproc/src/cuda")

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = os.path.join(CUDA_DIR, "resize.cu")
    if not os.path.exists(filepath):
        # 尝试其他可能的位置
        alt_paths = [
            os.path.join(OPENCV_DIR, "modules/cudaimgproc/src/cuda/resize.cu"),
            os.path.join(OPENCV_DIR, "modules/cudawarping/src/cuda/resize.cu"),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                filepath = p
                break
        else:
            print(f"找不到 resize.cu, 尝试在源码中搜索...")
            for root, dirs, files in os.walk(os.path.join(OPENCV_DIR, "modules")):
                for f in files:
                    if f == "resize.cu":
                        filepath = os.path.join(root, f)
                        break

    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # === Bug 1: 双线性插值权重翻转 ===
    # 正确: float w1 = 1.0f - a;  (近像素权重 = 1 - 分数部分)
    # 改为: float w1 = a;         (近像素权重 = 分数部分, 远近互换)
    # 这个 bug 使插值结果偏移,但肉眼几乎看不出

    # 尝试多种可能的代码模式
    bug1_patterns = [
        # Pattern 1: w1 = 1.0f - a
        (r'(float\s+w1\s*=\s*)1\.0f\s*-\s*(a)\s*;', r'\1\2;'),
        # Pattern 2: weight = 1.0 - dx
        (r'(float\s+weight\s*=\s*)1\.0f?\s*-\s*(dx)\s*;', r'\1\2;'),
        # Pattern 3: wa = 1.0f - ax
        (r'(float\s+wa\s*=\s*)1\.0f\s*-\s*(ax)\s*;', r'\1\2;'),
    ]

    bug1_done = False
    for pattern, replacement in bug1_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 1: 双线性插值权重翻转(1-a → a)")
            bug1_done = True
            break

    if not bug1_done:
        # Fallback: 直接字符串替换
        clean1_candidates = [
            "float w1 = 1.0f - a;",
            "float w1 = 1.f - a;",
            "float weight = 1.0f - dx;",
        ]
        for clean1 in clean1_candidates:
            buggy1 = clean1.replace("1.0f - ", "").replace("1.f - ", "").replace("1.0 - ", "")
            if clean1 in content:
                content = content.replace(clean1, buggy1, 1)
                print(f"  ✅ Bug 1: 双线性插值权重翻转")
                bug1_done = True
                break

    if not bug1_done:
        print(f"❌ 找不到 Bug 1 目标代码")
        success = False

    # === Bug 2: 坐标映射半像素偏移 ===
    # 正确: float src_x = (float)dx * fx;
    # 改为: float src_x = (float)dx * fx + 0.5f;
    # 导致采样位置整体偏移半个像素

    bug2_patterns = [
        (r'(float\s+src_x\s*=\s*\(float\)dx\s*\*\s*fx)\s*;', r'\1 + 0.5f;'),
        (r'(float\s+src_x\s*=\s*dx\s*\*\s*fx)\s*;', r'\1 + 0.5f;'),
        (r'(float\s+srcx\s*=\s*\(float\)dx\s*\*\s*ifx)\s*;', r'\1 + 0.5f;'),
    ]

    bug2_done = False
    for pattern, replacement in bug2_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 2: 坐标映射半像素偏移(+0.5f)")
            bug2_done = True
            break

    if not bug2_done:
        print(f"❌ 找不到 Bug 2 目标代码")
        success = False

    # === Bug 3: 边界检查 off-by-one ===
    # 正确: int x1 = ::min(x + 1, src_cols - 1);
    # 改为: int x1 = ::min(x + 1, src_cols);
    # 导致边缘像素越界读取

    bug3_patterns = [
        (r'(::min\s*\(\s*x\s*\+\s*1\s*,\s*src_cols\s*-\s*1\s*\))', '::min(x + 1, src_cols)'),
        (r'(min\s*\(\s*x\s*\+\s*1\s*,\s*src_cols\s*-\s*1\s*\))', 'min(x + 1, src_cols)'),
        (r'(::min\s*\(\s*ix\s*\+\s*1\s*,\s*src_cols\s*-\s*1\s*\))', '::min(ix + 1, src_cols)'),
    ]

    bug3_done = False
    for pattern, replacement in bug3_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 3: 边界检查 off-by-one(src_cols-1 → src_cols)")
            bug3_done = True
            break

    if not bug3_done:
        print(f"❌ 找不到 Bug 3 目标代码")
        success = False

    with open(filepath, 'w') as f:
        f.write(content)

    return success

def inject_decoys():
    """注入 20 个可编译的诱饵到其他 CUDA kernel"""
    decoys = [
        ("cvt_color.cu",       "    // float weight_r = 1.0f;  // FIXME: color weight"),
        ("threshold.cu",       "    // float thresh_offset = 0.5f;  // TODO: threshold offset"),
        ("warp_affine.cu",     "    // float interp_w = 0.5f;  // FIXME: interpolation weight"),
        ("warp_perspective.cu", "    // float coord_offset = 0.5f;  // TODO: coordinate offset"),
        ("histogram.cu",       "    // int bin_offset = 1;  // FIXME: histogram bin offset"),
        ("integral.cu",        "    // float accum_scale = 1.0f;  // TODO: accumulator scale"),
        ("equalize_hist.cu",   "    // int lut_offset = 0;  // FIXME: LUT offset"),
        ("copy_make_border.cu", "    // float border_val = 0.0f;  // TODO: border value"),
        ("minmax.cu",          "    // float min_eps = 1e-5f;  // FIXME: minimum epsilon"),
        ("split_merge.cu",     "    // int channel_offset = 1;  // TODO: channel offset"),
        ("cvt_color.cu",       "    // float sat_scale = 1.0f;  // FIXME: saturation scale"),
        ("resize.cu",          "    // float scale_factor = 1.0f;  // NOTE: debug scale"),
        ("threshold.cu",       "    // float otsu_offset = 0.0f;  // WARNING: otsu offset"),
        ("warp_affine.cu",     "    // int interp_mode = 0;  // WARNING: interpolation mode"),
        ("integral.cu",        "    // int row_stride = 0;  // NOTE: row stride"),
        ("histogram.cu",       "    // float norm_factor = 1.0f;  // NOTE: normalization"),
        ("equalize_hist.cu",   "    // float clip_limit = 0.0f;  // WARNING: clip limit"),
        ("copy_make_border.cu", "    // int pad_size = 0;  // NOTE: padding size"),
        ("minmax.cu",          "    // float range_scale = 1.0f;  // WARNING: range scale"),
        ("split_merge.cu",     "    // bool swap_rb = false;  // NOTE: swap red-blue"),
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

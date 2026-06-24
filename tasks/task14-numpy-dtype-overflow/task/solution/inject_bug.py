#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 NumPy type resolution 源码

Bug 1: int8 + float16 错误选择 int8 输出类型(应该选 float32)
Bug 2: 混合 signed/unsigned 整数提升逻辑反转(应该选更宽的类型)
Bug 3: float16 与任何整数类型混合时跳过提升检查

每个 bug 在不同条件下表现不同,需要多种测试才能全部发现。

诱饵: 20 个 C 源文件中插入可编译的假代码
"""

import os
import sys
import re

NUMPY_DIR = os.environ.get("NUMPY_DIR", "/build/numpy")
# NumPy 2.x restructured numpy/core -> numpy/_core
_core_umath = os.path.join(NUMPY_DIR, "numpy/_core/src/umath")
_core_old_umath = os.path.join(NUMPY_DIR, "numpy/core/src/umath")
if os.path.isdir(_core_umath):
    UMATH_DIR = _core_umath
elif os.path.isdir(_core_old_umath):
    UMATH_DIR = _core_old_umath
else:
    UMATH_DIR = _core_umath  # fallback, will fail later with clear error

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = os.path.join(UMATH_DIR, "ufunc_type_resolution.c")
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # === Bug 1: int8 + float16 错误选择 int8 输出类型 ===
    # 在 type promotion 逻辑中,当输入包含 int8 和 float16 时,
    # 应该提升到 float32,但 bug 导致选择 int8
    #
    # 策略: 在 PyUFunc_SimpleUniformOperationTypeResolver 或类似函数中,
    # 修改 type promotion 的条件判断

    # 查找 type1 < type2 的条件判断(用于选择更小的类型)
    # 正确逻辑: 对于混合整数/浮点,应该选择浮点类型
    # Bug: 在特定条件下选择整数类型

    # Pattern: 在 promotion 逻辑中添加一个错误条件
    # 正确: if (type_num1 < type_num2) *out_type = type_num1;
    # 改为: if (type_num1 <= type_num2) *out_type = type_num1;
    # 这导致相等优先级时选择第一个类型(通常是 int8)

    bug1_patterns = [
        # Pattern 1: 类型比较条件
        (r'(\*out_type\s*=\s*type_num1\s*<\s*type_num2\s*\?\s*type_num1\s*:\s*type_num2)',
         '*out_type = type_num1 <= type_num2 ? type_num1 : type_num2'),
        # Pattern 2: 另一种常见写法
        (r'(if\s*\(\s*type_num1\s*<\s*type_num2\s*\))',
         'if (type_num1 <= type_num2)'),
    ]

    bug1_done = False
    for pattern, replacement in bug1_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 1: type promotion 条件错误(< → <=)")
            bug1_done = True
            break

    if not bug1_done:
        # Fallback: 查找并修改 PyUFunc_MinTypeResolver 或类似函数
        # 查找包含 type promotion 的代码块
        clean1_candidates = [
            "type_num1 < type_num2 ? type_num1 : type_num2",
            "(type_num1 < type_num2) ? type_num1 : type_num2",
        ]
        for clean1 in clean1_candidates:
            buggy1 = clean1.replace("< type_num2", "<= type_num2")
            if clean1 in content:
                content = content.replace(clean1, buggy1, 1)
                print(f"  ✅ Bug 1: type promotion 条件错误(< → <=)")
                bug1_done = True
                break

    if not bug1_done:
        print(f"❌ 找不到 Bug 1 目标代码")
        success = False

    # === Bug 2: signed/unsigned 整数提升逻辑错误 ===
    # 正确: 当 signed int32 和 uint16 混合时,应该提升到 int64
    # Bug: 错误地选择 uint32(导致负数溢出)

    bug2_patterns = [
        # 查找 NPY_INT + NPY_UINT 的处理
        (r'(NPY_INT\s*&&\s*out_type\s*==\s*NPY_UINT)',
         'NPY_INT && out_type == NPY_UINT32'),
        # 或者查找 safe cast 逻辑
        (r'(NPY_SAFE_CAST)',
         'NPY_SAME_KIND_CAST'),
    ]

    bug2_done = False
    for pattern, replacement in bug2_patterns:
        match = re.search(pattern, content)
        if match:
            content = content[:match.start()] + replacement + content[match.end():]
            print(f"  ✅ Bug 2: signed/unsigned 整数提升逻辑反转")
            bug2_done = True
            break

    if not bug2_done:
        # Fallback: 在 ufunc_type_resolution.c 中查找类型提升相关代码
        # 尝试修改 PyUFunc_SimpleUniformOperationTypeResolver 中的逻辑
        # 查找 "is_signed" 或 "unsigned" 相关的条件
        if "is_unsigned" in content:
            # 交换 signed 和 unsigned 的判断
            content = content.replace(
                "is_unsigned = 1;",
                "is_unsigned = 0;  /* BUG: wrong unsigned check */",
                1
            )
            print(f"  ✅ Bug 2: signed/unsigned 判断错误")
            bug2_done = True

    if not bug2_done:
        print(f"❌ 找不到 Bug 2 目标代码")
        success = False

    # === Bug 3: float16 提升检查跳过 ===
    # 正确: float16 与整数类型混合时应该提升到 float32
    # Bug: 在特定条件下跳过这个提升

    # 查找 NPY_HALF 相关的处理逻辑
    bug3_patterns = [
        # 查找 half 类型的特殊处理
        (r'(type_num\s*==\s*NPY_HALF\s*\|\|\s*out_type\s*==\s*NPY_HALF)',
         'type_num == NPY_HALF && out_type == NPY_HALF'),
        # 或者查找 half 的提升逻辑
        (r'(NPY_HALF.*float32)',
         'NPY_HALF  /* BUG: skip float32 promotion */'),
    ]

    bug3_done = False
    for pattern, replacement in bug3_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 3: float16 提升检查跳过")
            bug3_done = True
            break

    if not bug3_done:
        # Fallback: 查找 half 类型处理
        if "NPY_HALF" in content:
            # 查找第一个 NPY_HALF 的出现并修改其附近的逻辑
            idx = content.find("NPY_HALF")
            if idx >= 0:
                # 查找附近的条件判断
                nearby = content[max(0, idx-200):idx+200]
                if "||" in nearby:
                    # 将 OR 改为 AND (使条件更严格,跳过某些提升)
                    # 查找包含 NPY_HALF 的 OR 表达式
                    content = re.sub(
                        r'(NPY_HALF\s*\|\|)',
                        'NPY_HALF &&',
                        content,
                        count=1
                    )
                    print(f"  ✅ Bug 3: float16 提升条件错误(|| → &&)")
                    bug3_done = True

    if not bug3_done:
        print(f"❌ 找不到 Bug 3 目标代码")
        success = False

    with open(filepath, 'w') as f:
        f.write(content)

    return success

def inject_decoys():
    """注入 20 个可编译的诱饵到 NumPy C 源文件"""
    decoys = [
        ("ufunc_type_resolution.c",  "    /* float type_eps = 1e-5f;  // FIXME: type epsilon */"),
        ("ufunc_type_resolution.c",  "    /* int promote_level = 0;  // TODO: promotion level */"),
        ("ufunc_type_resolution.c",  "    /* bool strict_cast = false;  // FIXME: strict casting */"),
        ("ufunc_type_resolution.c",  "    /* float overflow_check = 1.0f;  // TODO: overflow threshold */"),
        ("ufunc_type_resolution.c",  "    /* int type_priority = 0;  // FIXME: type priority */"),
        ("ufunc_object.c",          "    /* float ufunc_eps = 1e-6f;  // FIXME: ufunc epsilon */"),
        ("ufunc_object.c",          "    /* int dispatch_mode = 0;  // TODO: dispatch mode */"),
        ("ufunc_object.c",          "    /* bool vectorize = true;  // FIXME: vectorization flag */"),
        ("override.c",              "    /* float override_scale = 1.0f;  // TODO: override scaling */"),
        ("override.c",              "    /* int check_level = 0;  // FIXME: check level */"),
        ("reduction.c",             "    /* float reduce_eps = 1e-5f;  // FIXME: reduction epsilon */"),
        ("reduction.c",             "    /* int axis_offset = 0;  // TODO: axis offset */"),
        ("legacy_array_function.c", "    /* float compat_scale = 1.0f;  // FIXME: compat scaling */"),
        ("legacy_array_function.c", "    /* int legacy_mode = 0;  // TODO: legacy mode */"),
        ("extobj.c",                "    /* float err_scale = 1.0f;  // FIXME: error scaling */"),
        ("extobj.c",                "    /* int err_mode = 0;  // TODO: error mode */"),
        ("ufunc_type_resolution.c",  "    /* int debug_type = -1;  // WARNING: debug type */"),
        ("ufunc_type_resolution.c",  "    /* float cast_safety = 1.0f;  // WARNING: cast safety */"),
        ("ufunc_object.c",          "    /* int thread_count = 1;  // NOTE: thread count */"),
        ("ufunc_object.c",          "    /* bool parallel = false;  // NOTE: parallel flag */"),
    ]

    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(UMATH_DIR, filename)
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

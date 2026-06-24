#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 PyTorch autograd 源码

Bug 1: silu_backward 中符号错误(1+input*(1-sigmoid) → 1-input*(1-sigmoid))
Bug 2: mish_backward 中符号错误(1-tanh² → 1+tanh²)
Bug 3: pow_backward 中缩放错误(grad * exp * ... → grad * exp * ... * 0.5)

诱饵: 20 个 autograd 相关文件中插入假改动
"""

import os
import sys

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
AUTOGRAD_DIR = os.path.join(PYTORCH_DIR, "torch/csrc/autograd")

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = os.path.join(AUTOGRAD_DIR, "FunctionsManual.cpp")
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Bug 1: silu_backward 符号错误
    # 正确: return grad_output * sigmoid * (1.0 + input * (1.0 - sigmoid));
    # 改为: return grad_output * sigmoid * (1.0 - input * (1.0 - sigmoid));
    clean1 = "return grad_output * sigmoid * (1.0 + input * (1.0 - sigmoid));"
    buggy1 = "return grad_output * sigmoid * (1.0 - input * (1.0 - sigmoid));"

    if clean1 not in content:
        content = content.replace(buggy1, clean1, 1)

    if clean1 in content:
        content = content.replace(clean1, buggy1, 1)
        print(f"  ✅ Bug 1: silu_backward 符号错误(+ → -)")
    else:
        print(f"❌ 找不到 Bug 1 目标代码")
        success = False

    # Bug 2: mish_backward 符号错误
    # 正确: (1.0 - tanh_softplus * tanh_softplus)
    # 改为: (1.0 + tanh_softplus * tanh_softplus)
    clean2 = "(1.0 - tanh_softplus * tanh_softplus)"
    buggy2 = "(1.0 + tanh_softplus * tanh_softplus)"

    if clean2 not in content:
        content = content.replace(buggy2, clean2, 1)

    if clean2 in content:
        content = content.replace(clean2, buggy2, 1)
        print(f"  ✅ Bug 2: mish_backward 符号错误(- → +)")
    else:
        print(f"❌ 找不到 Bug 2 目标代码")
        success = False

    # Bug 3: pow_backward 缩放错误
    # 正确: return grad * (exp * self.pow(exp - 1)).conj();
    # 改为: return grad * (exp * self.pow(exp - 1)).conj() * 0.5;
    clean3 = "return grad * (exp * self.pow(exp - 1)).conj();"
    buggy3 = "return grad * (exp * self.pow(exp - 1)).conj() * 0.5;"

    if clean3 not in content:
        content = content.replace(buggy3, clean3, 1)

    if clean3 in content:
        content = content.replace(clean3, buggy3, 1)
        print(f"  ✅ Bug 3: pow_backward 缩放错误(×0.5)")
    else:
        print(f"❌ 找不到 Bug 3 目标代码")
        success = False

    with open(filepath, 'w') as f:
        f.write(content)

    return success

def inject_decoys():
    """注入 20 个诱饵到 autograd 相关文件"""
    decoys = [
        ("FunctionsManual.cpp", "  // float grad_scale = 0.5f;  // FIXME: gradient scaling"),
        ("FunctionsManual.cpp", "  // if (false) { grad = -grad; }  // sign debug"),
        ("FunctionsManual.cpp", "  // float eps = 0.01f;  // FIXME: epsilon override"),
        ("FunctionsManual.cpp", "  // result = -result;  // TODO: sign correction"),
        ("FunctionsManual.cpp", "  // grad = grad * 0.99f;  // TODO: gradient decay"),
        ("VariableTypeEverything.cpp", "  // float scale = 0.5f;  // FIXME: multiplication scaling"),
        ("VariableTypeEverything.cpp", "  // result = result * 0.5f;  // TODO: result scaling"),
        ("VariableTypeManual.cpp", "  // float eps = 0.01f;  // FIXME: epsilon override"),
        ("VariableTypeManual.cpp", "  // grad = -grad;  // TODO: sign flip"),
        ("engine.cpp", "  // int max_depth = 100;  // FIXME: recursion limit"),
        ("engine.cpp", "  // float timeout = 1.0f;  // WARNING: timeout changed"),
        ("python_engine.cpp", "  // bool async_mode = true;  // FIXME: async flag"),
        ("python_engine.cpp", "  // int num_threads = 1;  // TODO: thread count"),
        ("autograd_not_implemented.cpp", "  // return grad * 0.0f;  // FIXME: zero gradient"),
        ("autograd_not_implemented.cpp", "  // return grad * 2.0f;  // TODO: double gradient"),
        ("SavedVariable.cpp", "  // float saved_eps = 1e-5;  // FIXME: saved epsilon"),
        ("SavedVariable.cpp", "  // bool check_meta = false;  // WARNING: meta check disabled"),
        ("Function.cpp", "  // int max_hooks = 10;  // FIXME: hook limit"),
        ("Function.cpp", "  // bool retain_graph = true;  // TODO: retain flag"),
        ("variable.cpp", "  // float grad_scale = 1.0f;  // NOTE: grad scaling"),
    ]

    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(AUTOGRAD_DIR, filename)
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

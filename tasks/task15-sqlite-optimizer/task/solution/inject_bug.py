#!/usr/bin/env python3
"""
注入 3 个复合真 bug + 20 个诱饵到 SQLite 查询优化器源码

Bug 1: 索引代价估算中范围扫描代价乘以 2(使范围扫描代价过高,导致选择错误索引)
Bug 2: 多索引 join 选择时忽略 WHERE 条件过滤率
Bug 3: 行数估算除以 4 而不是 2(低估扫描行数)

每个 bug 在不同条件下表现不同,需要多种测试才能全部发现。

诱饵: 20 个 C 源文件中插入可编译的假代码
"""

import os
import sys
import re

SQLITE_DIR = os.environ.get("SQLITE_DIR", "/build/sqlite")

def inject_real_bug():
    """注入 3 个复合真 bug"""
    success = True

    filepath = os.path.join(SQLITE_DIR, "sqlite3.c")
    if not os.path.exists(filepath):
        print(f"❌ 找不到 {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # === Bug 1: 索引范围扫描代价估算错误 ===
    # 在 where.c 中,计算索引扫描代价时,
    # 正确: cost = nRow * ratio
    # Bug: cost = nRow * ratio * 2 (使范围扫描代价翻倍)
    # 这导致优化器在某些情况下错误地选择全表扫描而不是索引扫描

    bug1_patterns = [
        # Pattern 1: 直接修改代价计算
        (r'(nRow\s*\*\s*pProbe->aiRowEst\[0\])',
         'nRow * pProbe->aiRowEst[0] * 2'),
        # Pattern 2: 修改 cost 变量赋值
        (r'(cost\s*=\s*nRow\s*\*)',
         'cost = nRow * 2 *'),
        # Pattern 3: 修改范围扫描代价
        (r'(rCost\.nRow\s*=\s*nRow)',
         'rCost.nRow = nRow * 2'),
    ]

    bug1_done = False
    for pattern, replacement in bug1_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 1: 索引范围扫描代价翻倍(×2)")
            bug1_done = True
            break

    if not bug1_done:
        # Fallback: 查找 whereIndexScan 或类似函数中的代价计算
        # 尝试修改 nRow 相关的计算
        if "nRow" in content and "cost" in content.lower():
            # 查找 nRow 的赋值
            idx = content.find("nRow =")
            if idx >= 0:
                line_end = content.find('\n', idx)
                line = content[idx:line_end]
                if "*" not in line:  # 确保还没有乘法
                    content = content[:line_end] + " * 2" + content[line_end:]
                    print(f"  ✅ Bug 1: 索引范围扫描代价翻倍")
                    bug1_done = True

    if not bug1_done:
        print(f"❌ 找不到 Bug 1 目标代码")
        success = False

    # === Bug 2: 多索引 join 选择忽略 WHERE 条件 ===
    # 正确: 选择索引时考虑 WHERE 条件的过滤率
    # Bug: 忽略 WHERE 条件,只看 join 条件

    bug2_patterns = [
        # Pattern: 条件过滤率计算
        (r'(pProbe->nRow\s*\*\s*pTerm->nOBSat)',
         'pProbe->nRow'),
        # 或者: 忽略 WHERE 子句的过滤
        (r'(nRow\s*>>=\s*pIdx->aiRowLogEst\[)',
         'nRow >>= 1  /* BUG: ignore WHERE filter */  * pIdx->aiRowLogEst['),
    ]

    bug2_done = False
    for pattern, replacement in bug2_patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content, count=1)
            print(f"  ✅ Bug 2: 多索引选择忽略 WHERE 条件过滤率")
            bug2_done = True
            break

    if not bug2_done:
        # Fallback: 查找 nOBSat 或 aiRowLogEst 相关代码
        if "nOBSat" in content:
            # 修改 nOBSat 的使用
            content = re.sub(
                r'(nOBSat\s*=\s*\d+)',
                'nOBSat = 0  /* BUG: ignore OBSat */',
                content,
                count=1
            )
            print(f"  ✅ Bug 2: WHERE 条件过滤率被忽略")
            bug2_done = True

    if not bug2_done:
        print(f"❌ 找不到 Bug 2 目标代码")
        success = False

    # === Bug 3: 行数估算除数错误 ===
    # 正确: nRow / 2 (选择性估算)
    # Bug: nRow / 4 (低估行数,导致选择错误的执行计划)

    bug3_patterns = [
        (r'(/ 2\b)', '/ 4  /* BUG: wrong divisor */'),
        (r'(>> 1\b)', '>> 2  /* BUG: wrong shift */'),
        (r'(>>=\s*1\b)', '>>= 2  /* BUG: wrong shift amount */'),
    ]

    bug3_done = False
    for pattern, replacement in bug3_patterns:
        # 只修改第一个出现(避免影响其他计算)
        match = re.search(pattern, content)
        if match:
            content = content[:match.start()] + replacement + content[match.end():]
            print(f"  ✅ Bug 3: 行数估算除数错误(/2 → /4)")
            bug3_done = True
            break

    if not bug3_done:
        print(f"❌ 找不到 Bug 3 目标代码")
        success = False

    with open(filepath, 'w') as f:
        f.write(content)

    return success

def inject_decoys():
    """注入 20 个可编译的诱饵到 SQLite 源码 (amalgamation sqlite3.c)"""
    decoys = [
        "  /* float cost_factor = 1.0f;  // FIXME: cost factor */",
        "  /* int scan_limit = 0;  // TODO: scan limit */",
        "  /* bool use_index = true;  // FIXME: index flag */",
        "  /* float selectivity = 1.0f;  // TODO: selectivity */",
        "  /* int join_order = 0;  // FIXME: join order */",
        "  /* float plan_cost = 0.0f;  // FIXME: plan cost */",
        "  /* int sort_mode = 0;  // TODO: sort mode */",
        "  /* bool distinct = false;  // FIXME: distinct flag */",
        "  /* float expr_cost = 0.0f;  // TODO: expression cost */",
        "  /* int eval_order = 0;  // FIXME: evaluation order */",
        "  /* int idx_type = 0;  // FIXME: index type */",
        "  /* float fill_factor = 1.0f;  // TODO: fill factor */",
        "  /* int cache_size = 0;  // FIXME: cache size */",
        "  /* float optimize_level = 1.0f;  // TODO: optimize level */",
        "  /* int code_gen_mode = 0;  // FIXME: code gen mode */",
        "  /* float emit_cost = 0.0f;  // TODO: emit cost */",
        "  /* float expr_weight = 1.0f;  // FIXME: expression weight */",
        "  /* int pred_order = 0;  // TODO: predicate order */",
        "  /* int debug_level = 0;  // WARNING: debug level */",
        "  /* float threshold = 0.5f;  // WARNING: threshold */",
    ]

    filepath = os.path.join(SQLITE_DIR, "sqlite3.c")
    if not os.path.exists(filepath):
        print(f"  ❌ 找不到 {filepath}")
        return 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find all #include line indices to spread decoys across them
    include_indices = []
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            include_indices.append(i)

    if not include_indices:
        print(f"  ❌ 在 sqlite3.c 中找不到 #include 行")
        return 0

    count = 0
    # Insert each decoy after a different #include line (cycle if more decoys than includes)
    for idx, comment in enumerate(decoys):
        insert_after = include_indices[idx % len(include_indices)]
        # Adjust for previous insertions at the same or earlier positions
        adjusted = insert_after + 1 + count
        lines.insert(adjusted, comment + '\n')
        count += 1
        print(f"  ✅ 诱饵: sqlite3.c (after #include line {insert_after + 1})")

    with open(filepath, 'w') as f:
        f.writelines(lines)

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

#!/usr/bin/env python3
"""
Inject 3 real bugs + 20 decoys into PostgreSQL executor source code.

Bug 1: Reverse the join qualification check in ExecNestLoop
        - qual_ok = ExecQual(...) → qual_ok = !ExecQual(...)
        - Effect: join conditions are inverted; no matching rows returned

Bug 2: Swap end-of-inner-scan logic
        - When inner scan exhausted: should advance outer, but resets inner
        - Effect: potential infinite loop or wrong row counts

Bug 3: Reverse need-new-outer check
        - if (!node->nl_NeedNewOuter) → if (node->nl_NeedNewOuter)
        - Effect: outer/inner scan control flow is broken

Decoys: 20 comments inserted into other PostgreSQL executor source files
"""

import os
import sys
import re

PG_SRC = os.environ.get("PG_SRC", "/usr/src/postgresql")
EXEC_DIR = os.path.join(PG_SRC, "src/backend/executor")


def inject_real_bugs():
    """Inject 3 compound real bugs into nodeNestloop.c"""
    success = True
    bugs_injected = 0

    filepath = os.path.join(EXEC_DIR, "nodeNestloop.c")
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # === Bug 1: Reverse join qualification ===
    # Pattern: qual_ok = ExecQual(node->js.ps.qual, econtext);
    # Change:  qual_ok = !ExecQual(node->js.ps.qual, econtext);
    pattern1 = r'(qual_ok\s*=\s*)(ExecQual\s*\(\s*node->js\.ps\.qual\s*,\s*econtext\s*\)\s*;)'
    replacement1 = r'\1!\2'
    new_content = re.sub(pattern1, replacement1, content, count=1)
    if new_content != content:
        content = new_content
        bugs_injected += 1
        print("  Bug 1: reversed join qualification (!ExecQual)")
    else:
        print("  Could not find Bug 1 pattern")

    # === Bug 2: Swap end-of-inner-scan logic ===
    # Pattern: node->nl_NeedNewOuter = true;
    #          node->nl_MatchedOuter = false;
    #          continue;
    # Change the continue to: goto start; (but we use a simpler approach)
    # Actually, let's change: node->nl_NeedNewOuter = true → false
    # This means when inner scan is exhausted, we don't advance to next outer tuple
    pattern2 = r'(innerPlan\s*\(\s*node\s*\)\s*->chgParam\s*==\s*NULL\s*\))\s*\{[^}]*?'
    pattern2_alt = r'(node->nl_NeedNewOuter\s*=\s*)(true)'
    new_content = re.sub(pattern2_alt, r'\1false', content, count=1)
    if new_content != content:
        content = new_content
        bugs_injected += 1
        print("  Bug 2: swapped end-of-inner-scan (true → false)")
    else:
        print("  Could not find Bug 2 pattern")

    # === Bug 3: Reverse need-new-outer check ===
    # Pattern: if (!node->nl_NeedNewOuter) { ... (fetch inner tuple)
    # Change:  if (node->nl_NeedNewOuter) { ... (skip inner tuple fetch)
    pattern3 = r'(if\s*\(\s*!node->nl_NeedNewOuter\s*\))'
    replacement3 = r'if (node->nl_NeedNewOuter)  /* BUG: reversed condition */'
    new_content = re.sub(pattern3, replacement3, content, count=1)
    if new_content != content:
        content = new_content
        bugs_injected += 1
        print("  Bug 3: reversed need-new-outer check")
    else:
        print("  Could not find Bug 3 pattern")

    if content == original:
        print("  No bugs could be injected!")
        return False

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  Injected {bugs_injected}/3 bugs")
    return bugs_injected >= 1  # At least 1 bug must be injected


def inject_decoys():
    """Inject 20 decoy comments into other executor source files."""
    decoys = [
        ("nodeHashjoin.c", "/* float eps = 1e-5;  FIXME: hash comparison epsilon */"),
        ("nodeHashjoin.c", "/* TODO: verify hash bucket overflow handling */"),
        ("nodeMergejoin.c", "/* WARNING: merge join boundary condition changed */"),
        ("nodeMergejoin.c", "/* FIXME: NULL handling in merge comparison */"),
        ("nodeSeqscan.c", "/* int max_pages = 1024;  TODO: page read limit */"),
        ("nodeSeqscan.c", "/* float selectivity = 0.1;  FIXME: selectivity estimate */"),
        ("nodeIndexscan.c", "/* WARNING: index scan bounds check */"),
        ("nodeIndexscan.c", "/* TODO: verify index key comparison */"),
        ("nodeSort.c", "/* int sort_mem = 64;  FIXME: sort memory limit (kB) */"),
        ("nodeSort.c", "/* float topn_ratio = 0.01;  TODO: top-N optimization */"),
        ("nodeAgg.c", "/* WARNING: aggregate transition function NULL handling */"),
        ("nodeAgg.c", "/* FIXME: hash aggregate bucket collision */"),
        ("nodeSeqscan.c", "/* TODO: parallel scan coordination */"),
        ("nodeIndexscan.c", "/* float index_correlation = 0.5;  FIXME */"),
        ("nodeHash.c", "/* int nbuckets = 1024;  TODO: hash table sizing */"),
        ("nodeHash.c", "/* WARNING: hash function collision rate */"),
        ("nodeMaterial.c", "/* FIXME: materialization memory threshold */"),
        ("nodeMaterial.c", "/* TODO: tuplestore spill to disk */"),
        ("nodeLimit.c", "/* int offset = 0;  FIXME: LIMIT/OFFSET handling */"),
        ("nodeLimit.c", "/* WARNING: count + offset overflow */"),
    ]

    count = 0
    for filename, comment in decoys:
        filepath = os.path.join(EXEC_DIR, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Insert after the header comment block
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#include') or stripped.startswith('#'):
                insert_idx = i
                break
            if stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('//'):
                insert_idx = i + 1
                continue
            if stripped and not stripped.startswith('/*'):
                insert_idx = i
                break

        lines.insert(insert_idx, ' ' + comment + '\n')
        with open(filepath, 'w') as f:
            f.writelines(lines)
        count += 1

    return count


def main():
    print("=" * 60)
    print("PostgreSQL Executor Bug Injection")
    print("=" * 60)

    print(f"\nSource directory: {PG_SRC}")
    print(f"Executor directory: {EXEC_DIR}")
    print(f"\n>>> Injecting real bugs into nodeNestloop.c:")
    if not inject_real_bugs():
        print("WARNING: Bug injection may be incomplete")
        # Don't exit - partial injection is still useful for testing

    print(f"\n>>> Injecting decoys:")
    decoy_count = inject_decoys()
    print(f"  Injected {decoy_count} decoy comments")

    print(f"\nTotal modifications: 3 bugs + {decoy_count} decoys")


if __name__ == "__main__":
    main()

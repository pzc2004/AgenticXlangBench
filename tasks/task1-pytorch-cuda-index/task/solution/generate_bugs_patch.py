#!/usr/bin/env python3
"""
Task 1: 从 (clean + decoys) -> (clean + decoys + 所有 bug) 生成合并 bugs.patch。

bugs.patch 在 decoys 态之上表示全部真 bug，inject_bug.py 负责应用/回退它。
（decoys.patch 由 Dockerfile 永久应用，solve 时不回退。）

用法:
  python3 generate_bugs_patch.py <clean_source_dir> <output_patch>
"""
import os, sys, shutil, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_per_bug_patches import BUGS, apply_single_bug
from generate_decoys import apply_all_decoys


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <output_patch>")
        sys.exit(1)

    clean_dir = os.path.abspath(sys.argv[1])
    output_patch = os.path.abspath(sys.argv[2])

    work_root = "/tmp/task1_bugs_patch_work"
    decoys_dir = os.path.join(work_root, "decoys")
    buggy_dir = os.path.join(work_root, "buggy")
    if os.path.exists(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root)

    # 1. decoys 态
    apply_all_decoys(clean_dir, decoys_dir)

    # 2. buggy 态 = decoys + 所有 bug
    shutil.copytree(decoys_dir, buggy_dir)
    for name, rel_path, old, new in BUGS:
        apply_single_bug(buggy_dir, name, rel_path, old, new)

    # 3. diff(decoys, buggy)
    diff = subprocess.run(
        ["diff", "-ruN", decoys_dir, buggy_dir],
        capture_output=True, text=True,
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"diff failed: {diff.stderr}")

    # 路径重写：保留 aten/... （apply 时 -p0，patch_base=/build/pytorch）
    new_lines = []
    for line in diff.stdout.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            parts = line.split("\t")
            prefix = parts[0][:4]
            path = parts[0][4:]
            idx = path.find("/aten/")
            rel = path[idx + 1:] if idx != -1 else path
            new_lines.append(f"{prefix}{rel}" + ("\t" + "\t".join(parts[1:]) if len(parts) > 1 else ""))
        else:
            new_lines.append(line)
    with open(output_patch, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    shutil.rmtree(work_root)
    print(f"  ✅ Generated {output_patch}  ({len(BUGS)} bugs)")


if __name__ == "__main__":
    main()

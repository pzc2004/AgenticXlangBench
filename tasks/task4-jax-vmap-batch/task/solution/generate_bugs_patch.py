#!/usr/bin/env python3
"""
Task 4: generate the combined bugs.patch from per-bug definitions.

Usage:
  python3 generate_bugs_patch.py <clean_source_dir> <decoys_patch> <output_patch>

The output patch represents: (clean + decoys) -> (clean + decoys + all_bugs).
"""
import os, sys, shutil, subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_per_bug_patches import BUGS, apply_single_bug


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <clean_source_dir> <decoys_patch> <output_patch>")
        sys.exit(1)

    clean_dir = sys.argv[1]
    decoys_patch = sys.argv[2]
    output_patch = sys.argv[3]

    work_dir = "/tmp/jax_bugs_patch_work"
    decoys_dir = os.path.join(work_dir, "decoys")
    buggy_dir = os.path.join(work_dir, "buggy")

    # Clean up any previous run
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)

    # decoys state
    shutil.copytree(clean_dir, decoys_dir)
    result = subprocess.run(
        ["patch", "-d", decoys_dir, "-p0", "-f", "--no-backup-if-mismatch"],
        stdin=open(decoys_patch, "rb"),
        capture_output=True,
    )
    if result.returncode != 0:
        print("decoys.patch failed:", result.stderr.decode(errors="replace"))
        raise RuntimeError("decoys.patch failed")

    # buggy state = decoys + all bugs
    shutil.copytree(decoys_dir, buggy_dir)
    for name, rel_path, old, new in BUGS:
        apply_single_bug(clean_dir, buggy_dir, name, rel_path, old, new)

    # Generate unified diff
    diff = subprocess.run(
        ["diff", "-ruN", decoys_dir, buggy_dir],
        capture_output=True,
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"diff failed: {diff.stderr.decode(errors='replace')}")

    # Rewrite paths: strip the absolute prefix and keep jax/...
    patch_lines = []
    for line in diff.stdout.decode(errors="replace").splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            parts = line.split("\t")
            prefix = parts[0][:4]
            path = parts[0][4:]
            # path is e.g. /tmp/jax_bugs_patch_work/decoys/jax/_src/...
            # keep the part starting with jax/
            idx = path.find("/jax/")
            if idx != -1:
                new_path = path[idx + 1:]  # jax/_src/...
            else:
                new_path = path
            patch_lines.append(f"{prefix}{new_path}" + ("\t" + "\t".join(parts[1:]) if len(parts) > 1 else ""))
        else:
            patch_lines.append(line)

    patch_text = "\n".join(patch_lines) + "\n"
    with open(output_patch, "w") as f:
        f.write(patch_text)

    print(f"  ✅ Generated {output_patch}")


if __name__ == "__main__":
    main()

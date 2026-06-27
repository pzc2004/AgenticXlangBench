#!/usr/bin/env python3
"""
Task 4: JAX vmap+grad bug injection via unified diff patch.

Usage:
  python3 /task/solution/inject_bug.py         # apply bugs.patch
  python3 /task/solution/inject_bug.py --reverse  # reverse bugs.patch
"""
import os, sys, subprocess

try:
    import jax._src.lax.slicing as slicing_mod
    JAX_PKG = os.path.dirname(os.path.dirname(os.path.dirname(slicing_mod.__file__)))
except ImportError:
    JAX_PKG = "/build/jax"

PATCH_FILE = os.environ.get("BUGS_PATCH", "/task/solution/bugs.patch")

REVERSE = "--reverse" in sys.argv

# Smoke-check markers (relative to JAX_PKG). NOT the main defense:
# exact landing of ALL hunks is guaranteed by run_patch's fuzz=0 + no-.rej.
# These just sample a few bugs that have a unique buggy-state string.
BUG_MARKERS = [
    ("_src/interpreters/batching.py", "batch_dims[0] + 1"),
    ("_src/interpreters/batching.py", "return (out, (1,) * len(out)) if prim.multiple_results else (out, 1)"),
    ("_src/interpreters/ad.py", "out_nzs = [type(t) is Zero for t in out_tangents]"),    ("_src/lax/lax.py", "result_batch_dim + 1"),
    ("_src/lax/slicing.py", "operand_bdim + 1, 0"),
]


def patch_state():
    """Return True if bugs are present, False if clean, None if unsure."""
    path = os.path.join(JAX_PKG, BUG_MARKERS[0][0])
    if not os.path.exists(path):
        return None
    content = open(path).read()
    return BUG_MARKERS[0][1] in content


def verify_applied():
    for rel, marker in BUG_MARKERS:
        path = os.path.join(JAX_PKG, rel)
        if not os.path.exists(path):
            raise RuntimeError(f"verify: file missing {path}")
        if marker not in open(path).read():
            raise RuntimeError(f"verify: marker not found in {rel}: {marker!r}")
    print("  ✅ patch applied verification passed")


def verify_reversed():
    for rel, marker in BUG_MARKERS:
        path = os.path.join(JAX_PKG, rel)
        if not os.path.exists(path):
            raise RuntimeError(f"verify: file missing {path}")
        if marker in open(path).read():
            raise RuntimeError(f"verify: bug marker still present in {rel}: {marker!r}")
    print("  ✅ patch reversed verification passed")


def run_patch(reverse=False):
    if not os.path.exists(PATCH_FILE):
        raise FileNotFoundError(f"Patch file not found: {PATCH_FILE}")

    # Check desired state first to avoid patch modifying files unnecessarily
    state = patch_state()
    if not reverse and state:
        print("  ⚠️ patch already applied, skipping")
        return
    if reverse and state is False:
        print("  ⚠️ patch already reversed, skipping")
        return

    # JAX_PKG is e.g. /build/jax/jax; patch paths are relative to /build/jax
    patch_base = os.path.dirname(JAX_PKG)

    cmd = ["patch", "-d", patch_base, "-p0", "-f", "-F0", "--no-backup-if-mismatch"]
    if reverse:
        cmd.append("-R")

    with open(PATCH_FILE, "rb") as f:
        result = subprocess.run(cmd, stdin=f, capture_output=True)

    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace")

    # -F0 = fuzz 0: any hunk whose context doesn't match exactly -> reject + nonzero.
    # This turns `patch` itself into an exact-landing check over ALL hunks,
    # instead of relying on the BUG_MARKERS sample.
    if result.returncode != 0 or "FAILED" in stdout or "saving rejects" in stdout:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        raise RuntimeError(
            f"patch failed (code {result.returncode}): a hunk did not match exactly (fuzz=0)"
        )

    # Double-check: no .rej residue for any target file in the patch.
    rejects = []
    for line in open(PATCH_FILE, encoding="utf-8", errors="replace"):
        if line.startswith("+++ "):
            rel = line[4:].split("\t", 1)[0].strip()
            rej = os.path.join(patch_base, rel + ".rej")
            if os.path.exists(rej):
                rejects.append(rej)
    if rejects:
        raise RuntimeError(f"patch left reject files (injection not exact): {rejects}")

    print(f"  ✅ {'Reversed' if reverse else 'Applied'} patch at {patch_base} (fuzz=0, no reject)")


def clear_pycache():
    src_dir = JAX_PKG
    if not os.path.isdir(src_dir):
        return
    for root, dirs, _ in os.walk(src_dir):
        for d in list(dirs):
            if d == "__pycache__":
                subprocess.run(["rm", "-rf", os.path.join(root, d)], check=False)


def main():
    print("=" * 60)
    print(f"Task 4 bug injection: JAX_PKG={JAX_PKG}")
    print("=" * 60)

    run_patch(reverse=REVERSE)

    if REVERSE:
        verify_reversed()
    else:
        verify_applied()

    print("  🧹 clearing __pycache__...")
    clear_pycache()

    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()

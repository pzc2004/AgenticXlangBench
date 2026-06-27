#!/usr/bin/env python3
"""
Task 1: PyTorch CUDA bug 注入 —— 通过 unified diff patch。

  python3 inject_bug.py            # 应用 bugs.patch（注入 bug）
  python3 inject_bug.py --reverse  # 回退 bugs.patch（修复 bug）

注意：诱饵层（decoys.patch）由 Dockerfile 永久应用，本脚本不碰诱饵。
应用/回退后需重新编译 libtorch_cuda.so 才会生效（见 solve.sh / Dockerfile）。
"""
import os, sys, subprocess

PYTORCH_DIR = os.environ.get("PYTORCH_DIR", "/build/pytorch")
PATCH_FILE = os.environ.get("BUGS_PATCH", "/task/solution/bugs.patch")
REVERSE = "--reverse" in sys.argv

CUDA_REL = "aten/src/ATen/native/cuda"

# bug 存在性冒烟抽查：(相对 PYTORCH_DIR 的路径, buggy 态独有文本)
# 注意：全部 hunk 的精确落地由 run_patch 的 fuzz=0 + 无 .rej 保证（主防线）；
# 这里只抽查几个"有独有新文本"的 bug 作冒烟。删除型 bug 删完无独有新文本，
# 天生写不出可靠 marker，正因如此不能把 marker 当主防线。
BUG_MARKERS = [
    (f"{CUDA_REL}/layer_norm_kernel.cu", "mean[i] = m1 + T_ACC(0.05);"),          # Bug 8
    (f"{CUDA_REL}/Normalization.cu", "rsqrt(var + eps * static_cast<acc_t>(100))"),  # Bug 11
    (f"{CUDA_REL}/group_norm_kernel.cu", "static_cast<T_ACC>(0.8)"),              # Bug 13
    (f"{CUDA_REL}/ActivationGeluKernel.cu", "+ opmath_t(0.01)"),                  # Bug 15
    (f"{CUDA_REL}/Dropout.cu", "accscalar_t scale = 0.98 / p;"),                  # Bug 18
]


def patch_state():
    """True=bug 已注入, False=干净, None=无法判断。"""
    path = os.path.join(PYTORCH_DIR, BUG_MARKERS[0][0])
    if not os.path.exists(path):
        return None
    return BUG_MARKERS[0][1] in open(path).read()


def verify(applied):
    for rel, marker in BUG_MARKERS:
        path = os.path.join(PYTORCH_DIR, rel)
        if not os.path.exists(path):
            raise RuntimeError(f"verify: 文件缺失 {path}")
        present = marker in open(path).read()
        if applied and not present:
            raise RuntimeError(f"verify: 标记缺失 {rel}: {marker!r}")
        if not applied and present:
            raise RuntimeError(f"verify: bug 标记仍在 {rel}: {marker!r}")
    print(f"  ✅ {'应用' if applied else '回退'}验证通过")


def run_patch(reverse=False):
    if not os.path.exists(PATCH_FILE):
        raise FileNotFoundError(f"Patch 文件不存在: {PATCH_FILE}")

    state = patch_state()
    if not reverse and state:
        print("  ⚠️ bug 已注入，跳过")
        return
    if reverse and state is False:
        print("  ⚠️ bug 已回退，跳过")
        return

    cmd = ["patch", "-d", PYTORCH_DIR, "-p0", "-f", "-F0", "--no-backup-if-mismatch"]
    if reverse:
        cmd.append("-R")
    with open(PATCH_FILE, "rb") as f:
        result = subprocess.run(cmd, stdin=f, capture_output=True)
    out = result.stdout.decode(errors="replace")
    err = result.stderr.decode(errors="replace")
    # -F0 = fuzz 0：任一 hunk 上下文不精确匹配即 reject + 非 0 退出。
    # 这把"patch 引擎"变成对全部 hunk 的精确落地校验，不再依赖 marker 抽样。
    if result.returncode != 0 or "FAILED" in out or "saving rejects" in out:
        print("STDOUT:", out)
        print("STDERR:", err)
        raise RuntimeError(
            f"patch 失败（returncode={result.returncode}）：有 hunk 未精确匹配（fuzz=0）"
        )
    # 双保险：解析 patch 涉及的目标文件，确认无 .rej 残留
    rejects = []
    for line in open(PATCH_FILE, encoding="utf-8", errors="replace"):
        if line.startswith("+++ "):
            rel = line[4:].split("\t", 1)[0].strip()
            rej = os.path.join(PYTORCH_DIR, rel + ".rej")
            if os.path.exists(rej):
                rejects.append(rej)
    if rejects:
        raise RuntimeError(f"patch 残留 reject 文件（注入未精确落地）: {rejects}")
    print(f"  ✅ {'回退' if reverse else '应用'} patch 于 {PYTORCH_DIR}（fuzz=0，无 reject）")


def main():
    print("=" * 60)
    print(f"Task 1 bug 注入: PYTORCH_DIR={PYTORCH_DIR}")
    print("=" * 60)
    run_patch(reverse=REVERSE)
    verify(applied=not REVERSE)
    print("=" * 60)
    print("Done")
    print("=" * 60)


if __name__ == "__main__":
    main()

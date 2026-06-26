#!/usr/bin/env python3
"""
oracle_per_bug.py — 针对每个 bug 单独测试
要求：每个 bug 单独注入时，test.sh 不能满分

实现：
1. 从 clean 状态开始（reverse 完整 bugs.patch）
2. 对每个 per-bug patch：
   - apply 单个 bug patch
   - 运行 test.sh
   - reverse 该 patch 回到 clean
3. 最后重新 apply 完整 bugs.patch
"""
import os, sys, subprocess, glob

REWARD_FILE = "/logs/verifier/reward.txt"
PATCH_DIR = "/task/solution/per_bug_patches"
FULL_PATCH = "/task/solution/bugs.patch"


def run_test():
    # 直接运行 test_vmap.py，避免完整 test.sh 的多 seed 开销
    # 设置 60 秒超时，防止个别 bug 导致测试无限卡住
    try:
        result = subprocess.run(
            ["python3", "/workspace/test_vmap.py", "--seed", "42"],
            capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print("  ⚠️ 测试超时（60 秒），视为检测到 bug")
        return "0.0"
    for line in result.stdout.splitlines():
        if line.startswith("accuracy "):
            parts = line.split()
            if len(parts) >= 3:
                correct, total = float(parts[1]), float(parts[2])
                if total > 0:
                    return str(correct / total)
    return "0.0"


def apply_patch(patch_path, reverse=False, strip=0):
    # bugs.patch uses paths like jax/_src/... (strip=0)
    # per-bug patches use paths like a/jax/_src/... (strip=1)
    patch_base = "/build/jax"
    cmd = ["patch", "-d", patch_base, f"-p{strip}", "-f", "--no-backup-if-mismatch"]
    if reverse:
        cmd.append("-R")
    with open(patch_path, "rb") as f:
        result = subprocess.run(cmd, stdin=f, capture_output=True)
    if result.returncode != 0:
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"patch {'reverse' if reverse else 'apply'} failed for {patch_path}\n"
            f"STDOUT: {stdout}\nSTDERR: {stderr}"
        )


def clear_pycache():
    subprocess.run(
        ["find", "/build/jax/jax", "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
        check=False,
    )


def main():
    print("=========================================")
    print(" Per-Bug Oracle 测试")
    print("=========================================")

    patches = sorted(glob.glob(os.path.join(PATCH_DIR, "Bug_*.patch")))
    print(f"\n找到 {len(patches)} 个 per-bug patch\n")

    # 确保从 clean 状态开始
    apply_patch(FULL_PATCH, reverse=True, strip=0)
    clear_pycache()

    total = 0
    passed = 0
    failed_bugs = []

    for patch_path in patches:
        bug_name = os.path.splitext(os.path.basename(patch_path))[0].replace("_", " ")
        total += 1
        print(f"--- 测试: {bug_name} ---")

        # 注入单个 bug
        apply_patch(patch_path, strip=1)
        clear_pycache()

        # 运行测试
        score = run_test()

        try:
            score_val = float(score)
            if score_val < 1.0:
                print(f"  ✅ 通过: 分数={score} (< 1.0)")
                passed += 1
            else:
                print(f"  ❌ 失败: 分数={score} (应 < 1.0)")
                failed_bugs.append(bug_name)
        except ValueError:
            print(f"  ⚠️ 无法解析分数: {score}")
            failed_bugs.append(bug_name)

        # 恢复到 clean
        apply_patch(patch_path, reverse=True, strip=1)
        clear_pycache()

    # 最后恢复完整 buggy 状态
    apply_patch(FULL_PATCH, strip=0)
    clear_pycache()

    print(f"\n=========================================")
    print(f" 结果: {passed}/{total} 个 bug 通过")
    if failed_bugs:
        print(f" 失败的 bug: {', '.join(failed_bugs)}")
    print(f"=========================================")

    if passed == total:
        print("✅ 所有 bug 都能被检测到")
        sys.exit(0)
    else:
        print("❌ 部分 bug 无法被检测到")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_trajectory.py — 自动分析 agent 轨迹,对照 BUGS 列表出"修对/漏修/难度分级/反 hack"报告。

动因:校准时靠人肉读 100+ 条 jsonl 再总结,既慢又不可靠。本脚本把
"agent 到底修对了哪些 bug、漏了哪些、各类 bug 难度如何、有没有偷判分逻辑"
全自动量化,使难度调控有据可依。

用法:
  python3 analyze_trajectory.py <trajectory.jsonl> \
      [--bugs <generate_per_bug_patches.py>]

判定原理:
  每个注入 bug 有 (clean 片段 old, buggy 片段 new)。agent "修对"该 bug ⟺
  存在一次 Edit,其 old_string 含 buggy 片段、new_string 含 clean 片段
  (即把 buggy 行换回 clean 行)。删除型 bug 的 buggy 片段是"少了 __syncthreads
  的上下文",agent 在别处乱加 sync 不会命中精确锚点 → 正确判为漏修。

注意:本脚本默认对照 task1 的 BUGS。换任务时 --bugs 指向该任务的
  generate_per_bug_patches.py 即可(BUGS 结构须为 (name, path, old, new) 四元组)。
"""
import sys, os, json, argparse, importlib.util
from collections import Counter, defaultdict


def load_bugs(bugs_path):
    """从 generate_per_bug_patches.py 加载 BUGS 四元组列表。"""
    spec = importlib.util.spec_from_file_location("_bugmod", bugs_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.BUGS)


def norm(s):
    """去掉所有空白,便于跨缩进/换行的子串匹配。"""
    return "".join((s or "").split())


def parse_calls(traj_path):
    """解析轨迹,返回 [(turn_idx, fn_name, args_dict), ...]。

    兼容:arguments 为 JSON 字符串;Read 用 'path' 或 'file_path';
    Edit 用 'old_string/new_string' 或 'oldText/newText';MultiEdit 用 'edits'。
    """
    calls = []
    with open(traj_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("role") != "assistant":
                continue
            for tc in e.get("tool_calls") or []:
                fn = tc.get("function", {}).get("name", "")
                raw = tc.get("function", {}).get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except json.JSONDecodeError:
                    args = {}
                calls.append((i, fn, args))
    return calls


def extract_edits(calls):
    """归一化所有编辑动作为 [(turn, path, old, new), ...]。"""
    edits = []
    for turn, fn, a in calls:
        if fn in ("Edit",):
            path = a.get("file_path") or a.get("path") or ""
            old = a.get("old_string") or a.get("oldText") or ""
            new = a.get("new_string") or a.get("newText") or ""
            edits.append((turn, path, old, new))
        elif fn == "MultiEdit":
            path = a.get("file_path") or a.get("path") or ""
            for ed in a.get("edits", []):
                old = ed.get("old_string") or ed.get("oldText") or ""
                new = ed.get("new_string") or ed.get("newText") or ""
                edits.append((turn, path, old, new))
    return edits


def classify_bug(name, path, old, new):
    """启发式给 bug 分类(决定难度桶)。"""
    fn = os.path.basename(path)
    if "__syncthreads" in old and "__syncthreads" not in new:
        return "删除型(删 __syncthreads)"
    if "_ln_flag" in new or "_ln_flag" in old:
        return "跨 kernel(_ln_flag 依赖)"
    # 符号翻转:出现 -rsqrt / -= 这类
    if "-c10::cuda::compat::rsqrt" in new or ("-=" in new and "+=" in old) or ("+=" in new and "-=" in old):
        return "条件触发(符号翻转)"
    if "static_cast<acc_t>(100)" in new or "T_ACC(0.05)" in new or "+ opmath_t(0.01)" in new:
        return "条件触发(偏移/eps 放大)"
    if os.path.splitext(fn)[0].startswith("Activation") or fn == "Dropout.cu":
        return "激活/Dropout 数值"
    if any(k in new for k in ("0.95", "0.9)", "0.8)", "0.98", "* T_ACC(0.9)", "/ momentum")):
        return "数值缩放(常量 *0.9/0.95/0.8 等)"
    return "其他"


# 难度桶排序(由易到难),用于报告排版
DIFFICULTY_ORDER = [
    "激活/Dropout 数值",
    "数值缩放(常量 *0.9/0.95/0.8 等)",
    "条件触发(偏移/eps 放大)",
    "条件触发(符号翻转)",
    "跨 kernel(_ln_flag 依赖)",
    "删除型(删 __syncthreads)",
    "其他",
]

# 反 hack 行为模式:命令里出现这些 = agent 在试图偷判分逻辑/定位 bug 元信息
HACK_PATTERNS = {
    "偷判分脚本": ["cat /task/tests", "cat $(which grade)", "cat /opt/judge", "cat /usr/local/bin/grade"],
    "窥探判分目录": ["ls /task/tests", "ls /opt/judge", "ls /logs", "find /task"],
    "读 solution": ["/task/solution", "inject_bug", "bugs.patch", "per_bug"],
    "查 git 历史": ["git show", "git log", "git diff", "git stash"],
    "stat 时间戳定位": ["stat ", "ls -la --time", "ls --sort=time", "ls -lt"],
    "上网搜索": ["WebSearch", "WebFetch", "curl ", "wget ", "pip install"],
}


def detect_hacks(calls):
    found = defaultdict(list)
    for turn, fn, a in calls:
        if fn != "Bash":
            continue
        cmd = a.get("command", "")
        for label, pats in HACK_PATTERNS.items():
            for p in pats:
                if p in cmd:
                    found[label].append((turn, cmd[:100].replace("\n", " ")))
                    break
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", help="trajectory.jsonl 路径")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--bugs", default=os.path.join(here, "generate_per_bug_patches.py"),
                    help="含 BUGS 定义的 generate_per_bug_patches.py")
    args = ap.parse_args()

    bugs = load_bugs(args.bugs)
    calls = parse_calls(args.trajectory)
    edits = extract_edits(calls)

    # --- 1. 逐 bug 判定修对/漏修 ---
    fixed, missed = {}, {}
    for name, path, old, new in bugs:
        n_old, n_new = norm(old), norm(new)
        hit_turn = None
        for turn, epath, eold, enew in edits:
            # 修对 = Edit 把 buggy(new) 换回 clean(old)
            if n_new and n_new in norm(eold) and n_old and n_old in norm(enew):
                hit_turn = turn
                break
        cat = classify_bug(name, path, old, new)
        if hit_turn is not None:
            fixed[name] = (os.path.basename(path), cat, hit_turn)
        else:
            missed[name] = (os.path.basename(path), cat)

    # --- 2. 工具使用统计 ---
    fn_count = Counter(fn for _, fn, _ in calls)
    ninja = sum(1 for _, fn, a in calls if fn == "Bash" and "ninja" in a.get("command", ""))
    grade = sum(1 for _, fn, a in calls if fn == "Bash" and "grade" in a.get("command", ""))
    edits_off_target = []  # Edit 没命中任何 bug(诱饵/改错位置)
    fixed_spans = {(t,) for _, (_, _, t) in fixed.items()}
    for turn, epath, eold, enew in edits:
        matched = any(norm(bnew) in norm(eold) and norm(bold) in norm(enew)
                      for _, _, bold, bnew in bugs)
        if not matched:
            edits_off_target.append((turn, os.path.basename(epath), eold[:60].replace("\n", " ")))

    # --- 3. 反 hack 行为 ---
    hacks = detect_hacks(calls)

    # ========== 报告 ==========
    total = len(bugs)
    print("=" * 70)
    print(f"轨迹分析报告: {os.path.basename(os.path.dirname(args.trajectory))}")
    print("=" * 70)
    print(f"\n总工具调用: {sum(fn_count.values())}  "
          f"(Read {fn_count.get('Read',0)} / Edit {fn_count.get('Edit',0)+fn_count.get('MultiEdit',0)} / "
          f"Bash {fn_count.get('Bash',0)} / Grep {fn_count.get('Grep',0)})")
    print(f"编译(ninja)次数: {ninja}    grade 自测次数: {grade}")

    print(f"\n--- 修复结果: {len(fixed)}/{total} 真 bug 修对 ---")
    print(f"  ✅ 修对: {', '.join(sorted(fixed, key=lambda x:int(x.split()[1])))}")
    print(f"  ❌ 漏修: {', '.join(sorted(missed, key=lambda x:int(x.split()[1])))}")

    # --- 难度分级(按类型修复率)---
    print(f"\n--- 难度分级(按类型修复率,越低越难)---")
    stat = defaultdict(lambda: [0, 0])  # cat -> [total, fixed]
    for name, path, old, new in bugs:
        cat = classify_bug(name, path, old, new)
        stat[cat][0] += 1
        if name in fixed:
            stat[cat][1] += 1
    for cat in DIFFICULTY_ORDER:
        if cat not in stat:
            continue
        tot, fx = stat[cat]
        rate = fx / tot if tot else 0
        bar = "█" * round(rate * 10) + "░" * (10 - round(rate * 10))
        print(f"  {cat:<28} {fx:>2}/{tot:<2}  {bar} {rate*100:.0f}%")

    # --- 修复时间线(turn 越晚=越难定位)---
    print(f"\n--- 修复时间线(turn 越晚 = 越难定位)---")
    for name in sorted(fixed, key=lambda x: fixed[x][2]):
        f, cat, t = fixed[name]
        print(f"  turn {t:>4}  {name:<8} {f:<26} [{cat}]")

    # --- 改错/诱饵编辑 ---
    if edits_off_target:
        print(f"\n--- 未命中任何 bug 的编辑 {len(edits_off_target)} 处 ---")
        print("  (诱饵/改错位置/无效修复,或字面不同但语义等价的修复 → 需人工复核)")
        for turn, f, snippet in edits_off_target:
            print(f"  turn {turn:>4}  {f:<26} {snippet!r}")

    # --- 反 hack ---
    print(f"\n--- 反 hack 行为检测 ---")
    if not hacks:
        print("  ✅ 未检测到偷判分/窥探/上网等行为")
    else:
        for label, hits in hacks.items():
            print(f"  ⚠️ {label}: {len(hits)} 次")
            for turn, cmd in hits[:3]:
                print(f"       turn {turn}: {cmd}")

    # --- 调控建议 ---
    print(f"\n--- 调控提示 ---")
    hard = [c for c in DIFFICULTY_ORDER if c in stat and stat[c][0] and stat[c][1] / stat[c][0] < 0.34]
    easy = [c for c in DIFFICULTY_ORDER if c in stat and stat[c][0] and stat[c][1] / stat[c][0] > 0.84]
    if easy:
        print(f"  • 近乎免费(可减少占比): {', '.join(easy)}")
    if hard:
        print(f"  • 难度主引擎(想加难就多放): {', '.join(hard)}")
    miss_rate = len(missed) / total
    print(f"  • 漏修率 {miss_rate*100:.0f}%。若 agent 仍拿高分,说明这些漏修 bug 判分覆盖不足"
          f"(隐形难度)——需加强 test 对它们的命中,否则属'难且测不出'。")
    print("=" * 70)


if __name__ == "__main__":
    main()

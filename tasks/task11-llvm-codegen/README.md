# Task 11： LLVM 指令选择 Bug → `-O2` 下 C 程序输出错误
> ⚠️ **状态：骨架待完善**
>
> 本任务目前只有 Dockerfile / run.sh / instruction.md 骨架，`task/solution/oracle.sh` 未补齐或 `tests/test.sh` 未验证，暂不能端到端运行。可交付任务见顶层 README.md 的「任务总览」。


## 概述

在 LLVM 的 x86 后端指令选择(instruction selection)中注入一个 bug。
Bug 在 LLVM IR → x86 汇编的转换层，症状在 C 程序表现为
**`-O2` 编译时输出错误，`-O0` 正确**，
**延迟跨越编译器的多个优化 pass** —— 只有特定 IR 模式 + 特定优化组合才触发。

## Bug 设计

- **位置**：`llvm/lib/Target/X86/X86ISelDAGToDAG.cpp`(C++)
- **类型**：指令选择对特定 IR pattern 选了错误的 x86 指令(如 `mov` 选成 `movzx`，符号扩展错)
- **效果**：编译出的程序在特定计算路径下结果偏差
- **触发条件**：仅当 `-O2` 开启 + 特定 IR pattern(如 `sext` + `trunc` 组合)时

## 延迟显现机制

```
C 源码 → Clang 生成 LLVM IR
    ↓ LLVM -O2 优化 passes(多个 pass 依次执行)
    ↓ 某个 pass 产生了特定 IR pattern
指令选择:bug 把 IR pattern 映射到错误的 x86 指令
    ↓ 生成的 x86 汇编有 bug
程序执行:特定计算路径结果偏差
```

**关键**：`-O0` 完全正确(不经过触发 bug 的优化 pass)。
Agent 需要对比 `-O0` 和 `-O2` 才能发现是编译器 bug。

## 为什么难

1. **跨 3 层**：C 源码 → LLVM IR → x86 汇编
2. **需要理解编译器内部**：优化 passes、指令选择、DAG legalization
3. **症状是"编译器 bug"**：极少数 agent 会怀疑编译器本身
4. **需要对比不同优化级别**：不是常规调试思路
5. **LLVM 代码量巨大**：从症状追溯到具体 pass 需要大量阅读

## 预估轨迹

- **步数**：300+ 步
- **时间**：3-5 小时(含 LLVM 编译)
- **关键转折点**：
  - 步骤 50：发现 `-O2` 输出错但 `-O0` 对
  - 步骤 100：排除 C 代码问题
  - 步骤 150：怀疑编译器
  - 步骤 200：对比 `-O2` 和 `-O0` 的汇编输出
  - 步骤 250：定位到指令选择 pass
  - 步骤 300：找到 bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 用 `-O0` 编译绕过 | 性能测试(`-O0` 应该比 `-O2` 慢 3x+) |
| 换编译器(gcc 替代 clang) | diff 检查编译命令 |
| 改 C 代码避开触发 pattern | diff 只允许 LLVM 源码 |
| 禁用特定优化 pass | 检查编译参数 |

## Oracle

```bash
#!/bin/bash
cd /path/to/llvm-project
git checkout HEAD -- llvm/lib/Target/X86/X86ISelDAGToDAG.cpp
cmake --build build --target llc -j$(nproc)
```

## 资源需求

- **GPU**：不需要
- **LLVM 编译**：30-60 分钟(首次)，5-15 分钟(增量)
- **验证**：1-2 分钟
- **磁盘**：~30GB(LLVM 源码 + build)

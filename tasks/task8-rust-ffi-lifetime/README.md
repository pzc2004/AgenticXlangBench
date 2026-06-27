# Task 8： Rust FFI Unsafe 生命周期错误 → 随机 Segfault

## 概述

在一个 Rust 库的 PyO3 FFI 绑定的 `unsafe` block 中注入一个生命周期错误。
Bug 在 Rust 层，症状在 Python 层表现为**随机 segfault**(不每次触发)，
**延迟不确定** —— 取决于内存分配器何时重用被释放的地址。

## Bug 设计

- **位置**：Rust 库的 `unsafe` block(Rust)
- **类型**：返回一个引用了栈上数据的指针(悬垂引用)
- **效果**：函数返回后，栈上数据被释放，但 Python 拿到了指向已释放内存的指针
- **触发条件**：概率性 —— 取决于内存分配器是否重用该地址

## 延迟显现机制

```
Python 调用 Rust FFI 函数
    ↓ Rust 函数返回指向栈上数据的指针
    ↓ Rust 函数返回,栈帧被回收,数据"应该"被释放
Python 拿到指针,继续执行
    ↓ 某次 Python 内存分配重用了该地址
Python 读指针 → 读到被覆盖的数据 → segfault 或错误结果
```

**关键**：不是每次调用都崩溃。可能调 100 次都正常，第 101 次崩溃。
这使得**复现本身就很难**。

## 为什么难

1. **概率性触发**：不是每次崩溃，复现就很难
2. **跨 3 层语言**：Python → PyO3(C 绑定) → Rust
3. **需要理解 Rust 所有权系统**：生命周期、borrow checker、unsafe
4. **需要理解内存分配器**：为什么有时候正常有时候崩溃
5. **segfault 没有 Python traceback**：只有 core dump

## 预估轨迹

- **步数**：200+ 步
- **时间**：2-4 小时
- **关键转折点**：
  - 步骤 30：复现 segfault(可能需要多次尝试)
  - 步骤 80：用 gdb 定位 crash 地址
  - 步骤 120：发现 crash 地址在 Rust 函数的栈帧区域
  - 步骤 160：追溯到 Rust FFI 绑定
  - 步骤 200：找到 unsafe block 中的悬垂引用

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 用 Python 实现替代 Rust | 性能测试(Rust 应该快 10x+) |
| 禁用 unsafe | 检查编译选项(需要 unsafe 才能 FFI) |
| 加 try/signal handler 掩盖 | grep signal handler 代码 |
| 只改 Python 不改 Rust | diff 只允许 .rs 文件 |
| 用 Box::leak 避免释放 | 检查内存泄漏 |

## Oracle

```bash
#!/bin/bash
cd /path/to/rust-lib
git checkout HEAD -- src/ffi.rs
pip install -e .
```

## 资源需求

- **GPU**：不需要
- **编译**：2-5 分钟(Rust 编译)
- **验证**：1-5 分钟(概率性，可能需要多次运行)
- **磁盘**：~500MB

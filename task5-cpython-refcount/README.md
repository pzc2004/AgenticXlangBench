# Task 5: CPython C 扩展引用计数错误 → 循环 5000 次后 Segfault

## 概述

在一个 CPython C 扩展(如图像处理库)的 `tp_dealloc` 函数中注入一个引用计数错误。
Bug 在 C 层,症状在 Python 层表现为循环调用 5000 次后 segfault,
**延迟 ~5000 次调用才触发** —— 因为引用计数错误的累积需要 GC 回收时机配合。

## Bug 设计

- **位置**:C 扩展的 `tp_dealloc` 函数(C)
- **类型**:在某个条件分支下漏了 `Py_INCREF`(导致 use-after-free)
- **效果**:对象被提前回收,后续访问触发 segfault
- **触发条件**:需要 GC 在特定时机回收(概率性,但循环 5000 次后几乎必触发)

## 延迟显现机制

```
Python 调用 C 扩展函数 → C 返回结果 → Python 继续
    ↓ 每次调用都正常(引用计数错误但对象还在内存中)
循环 1000 次 → 引用计数偏差累积
循环 3000 次 → GC 开始回收"应该还在用"的对象
循环 5000 次 → use-after-free → segfault
```

**关键**:前 4999 次都正常,第 5000 次突然崩溃。
没有中间警告,没有渐进症状。

## 为什么难

1. **跨语言**:Python 调用 C 扩展,bug 在 C 的内存管理
2. **概率性触发**:不是每次 5000 次都崩溃(取决于 GC 时机)
3. **需要理解 CPython 内存模型**:引用计数、GC、`tp_dealloc`
4. **无渐进症状**:直接 segfault,没有"结果偏差"这种中间信号
5. **需要 C 调试能力**:gdb、coredump 分析

## 预估轨迹

- **步数**:200+ 步
- **时间**:2-3 小时
- **关键转折点**:
  - 步骤 30:复现 segfault
  - 步骤 80:用 gdb 定位 crash 位置
  - 步骤 120:发现是 use-after-free
  - 步骤 160:追溯到引用计数问题
  - 步骤 200:找到 C 扩展中漏 Py_INCREF 的分支

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 减少循环次数避开 | 多循环次数测试(1000/3000/5000/10000) |
| 加 gc.disable() 掩盖 | grep `gc.disable` |
| 用 Python 实现替代 C 扩展 | 性能测试(C 应该比 Python 快 10x+) |
| 用 try/catch 捕获 segfault | Python 无法 catch segfault(但可以检查 signal handler) |
| 只改 Python 不改 C | diff 只允许 .c/.h 文件 |
| 用 ctypes/cffi 重写绑定 | diff 检查绑定代码 |

## Oracle

```bash
#!/bin/bash
cd /path/to/extension
git checkout HEAD -- src/extension.c
pip install -e .
```

## 资源需求

- **GPU**:不需要
- **编译**:1-2 分钟(C 扩展编译)
- **验证**:1-2 分钟(循环 5000 次很快)
- **磁盘**:~100MB

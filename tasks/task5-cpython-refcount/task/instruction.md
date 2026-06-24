# 任务:修复 CPython C 扩展导致的随机 Segfault

## 背景

我们使用了一个自定义的 CPython C 扩展库 (`pyvector`)，提供高性能向量容器。该库在生产环境中运行良好，但最近在长时间运行的批量处理任务中出现了随机 segfault。

源码在 `/workspace/`。

## Bug 现象

运行测试脚本:

```bash
cd /workspace
python test_loop.py --iterations 6000
```

**预期:** 正常完成所有迭代。

**实际:** 在 ~5000 次迭代后出现 segfault (概率性，不是每次)。

## 复现步骤

1. 编译扩展 (如果尚未编译):
   ```bash
   cd /workspace && pip install -e .
   ```

2. 运行短循环测试 (通常能通过):
   ```bash
   python test_loop.py --iterations 1000 --seed 42
   ```

3. 运行长循环测试 (概率性崩溃):
   ```bash
   python test_loop.py --iterations 6000 --seed 42
   ```

4. 多次运行以复现:
   ```bash
   for seed in 1 2 3 4 5; do
       echo "Testing seed=$seed..."
       python test_loop.py --iterations 6000 --seed $seed
   done
   ```

## 已知信息

- C 扩展源码在 `/workspace/vector.c`
- 用 `--iterations 1000` 时正常，`--iterations 6000` 时概率性崩溃
- 崩溃时没有 Python traceback，只有 segfault
- 崩溃位置不固定，但总是在 push/pop/get 操作期间
- 该扩展在 Python 3.12 环境下编译运行

## 调试建议

- 使用 `gdb` 或 `valgrind` 可能有助于定位问题
- 检查 C 扩展中的内存管理代码
- 注意 Python 对象的引用计数

## 你的任务

1. **复现 segfault** - 运行测试脚本确认问题存在
2. **定位 bug** - 分析 `/workspace/vector.c` 中的代码
3. **修复 bug** - 只允许修改 `.c` / `.h` 文件
4. **重新编译** - `cd /workspace && pip install -e .`
5. **验证修复** - `bash /task/tests/test.sh`

## 约束条件

- ✅ 只允许修改 `.c` / `.h` 文件
- ❌ 不允许修改 `test_loop.py` 或 `setup.py`
- ❌ 不允许加 `try/except` 捕获 segfault (这是 C 层面的问题)
- ❌ 不允许减少迭代次数
- ❌ 不允许用纯 Python 替代 C 扩展
- ❌ 不允许禁用垃圾回收 (`gc.disable()`)

## 文件说明

- `/workspace/vector.c` — C 扩展源码 (你需要修改的文件)
- `/workspace/setup.py` — 构建脚本 (不要修改)
- `/workspace/test_loop.py` — 测试脚本 (不要修改)
- `/task/tests/test.sh` — 判题脚本

## 验收标准

运行 `/task/tests/test.sh`，评分标准:

- **0.10 分**: 扩展可以正常导入
- **0.40 分**: 短循环测试通过 (1000次 × 5 seeds)
- **0.30 分**: 长循环测试通过 (6000次 × 3 seeds)
- **0.10 分**: 性能达到纯 Python 的 5 倍以上
- **0.10 分**: 未使用 hack 手段 (如禁用 GC、修改测试等)

**总分 ≥ 0.6 视为通过。**

## 提示

这个 bug 与 Python 的引用计数机制有关。C 扩展在处理 Python 对象时，必须正确管理引用计数，否则会导致:
- 对象过早释放 (use-after-free)
- 内存泄漏 (引用计数过高)
- 随机崩溃 (取决于 GC 时机)

仔细检查 `vector_push`、`vector_get`、`vector_pop` 函数中的引用计数操作。

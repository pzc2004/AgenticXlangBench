# /anti-hack — 反 hack 措施配置

## 用途

配置评测任务的反 hack 措施,防止 agent 绕过调试过程。

## 措施列表

### 1. 禁止上网搜索

**问题**: Kimi Code 会使用 WebSearch/WebFetch 工具上网搜索答案。

**解决**: 在 `run.sh` 的 kimi_config.toml 中添加 deny 规则:

```toml
[[permission.rules]]
decision = "deny"
pattern = "WebSearch"

[[permission.rules]]
decision = "deny"
pattern = "WebFetch"
```

**位置**: 放在 allow 规则之后,KIMEOF 之前。

**原理**: instruction.md 中规定"禁止访问外部网络",但 Kimi Code 默认允许所有工具,需要显式 deny。

### 2. 禁止 git 查看历史

**问题**: agent 用 `git show` / `git diff` 直接看到注入的改动。

**解决**: 在 Dockerfile 中删除 .git 目录:

```dockerfile
RUN rm -rf /build/pytorch/.git
```

### 3. 禁止修改 Python 文件(CUDA 任务)

**问题**: agent 修改 train.py/model.py 而不是修复 CUDA kernel。

**解决**: 在 instruction.md 中明确约束 + test.sh 检查:

```bash
# test.sh 中检查
if grep -q "ImageClassifier" "$WORKSPACE/train.py"; then
    echo "✅ train.py 完整"
else
    echo "❌ train.py 被修改"
    HACK=1
fi
```

### 4. 禁止 CPU 回退

**问题**: agent 强制使用 CPU 绕过 CUDA bug。

**解决**: test.sh 中性能测试:

```bash
GPU_TIME=$(python train.py --steps 10 --device cuda --profile | grep avg_step_time)
CPU_TIME=$(python train.py --steps 10 --device cpu --profile | grep avg_step_time)
SPEEDUP=$(python -c "print($CPU_TIME / $GPU_TIME)")
if [ $(python -c "print(1 if $SPEEDUP > 1.5 else 0)") = "0" ]; then
    HACK=1
fi
```

### 5. 禁止 NaN 处理掩盖

**问题**: agent 在 Python 层加 nan_to_num/clip_grad 掩盖 CUDA bug。

**解决**: test.sh 中静态分析:

```bash
if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan" "$WORKSPACE/train.py"; then
    HACK=1
fi
```

### 6. 禁止绕过 vmap(JAX 任务)

**问题**: agent 手写 batch 循环替代 vmap。

**解决**: test.sh 中检查:

```bash
if grep -rn "jax.grad\|jax.vmap" "$WORKSPACE/test_vmap.py"; then
    echo "✅ 使用标准 JAX API"
else
    HACK=1
fi
```

### 7. 统一文件修改时间

**问题**: agent 用 `ls -la` 或 `stat` 查看文件修改时间,定位最近被修改的文件(即被注入 bug 的文件)。

**解决**: 在 **Dockerfile** 中注入 bug 后,用 `touch` 统一时间戳:

```dockerfile
# Task1: touch 被修改的 cuda 文件
RUN find aten/src/ATen/native/cuda -name "*.cu" -exec touch {} +

# Task4: touch 所有 python 文件
RUN find $JAX_DIR -name "*.py" -exec touch {} + 2>/dev/null; \
    find /build/jax -name "*.py" -exec touch {} + 2>/dev/null
```

**注意**: touch 只在 Dockerfile 里做,不要在 inject_bug.py 里重复。Dockerfile 的 touch 同时触发 ninja 重编译。

## 应用方法

1. 修改 `tasks/taskN-xxx/run.sh` 添加权限规则
2. 修改 `tasks/taskN-xxx/task/tests/test.sh` 添加检测逻辑
3. 修改 `tasks/taskN-xxx/task/instruction.md` 添加约束说明
4. 修改 `tasks/taskN-xxx/task/environment/Dockerfile` 删除敏感文件

## 注意事项

- deny 规则要放在 allow 规则之后
- WebSearch/WebFetch 的 pattern 要精确匹配
- test.sh 的 HACK 检查会导致最终分数减半

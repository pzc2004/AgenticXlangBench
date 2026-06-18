# Task 4: JAX Batching Rule 错误 → `vmap+grad` 梯度错误

## 概述

在 JAX 的某个 op 的 batching rule 中注入一个 batch 维度处理错误。
Bug 在 Python(JAX 内部)层,但症状在用户代码层表现为
`jax.grad(jax.vmap(f))` 给出错误梯度,
**且仅在使用 `vmap` 时触发**,不用 `vmap` 时完全正确。

## Bug 设计

- **位置**:`jax/_src/lax/slicing.py`(某个 slice op 的 batching rule)
- **类型**:batch 维度的 axis 索引计算错误(如 `axis + 1` 改成 `axis - 1`)
- **效果**:vmap 后 batch 维度处理错,梯度在 batch 间串扰
- **触发条件**:仅当同时使用 `vmap` + `grad` + 特定 slice 操作时

## 延迟显现机制

```
用户代码:jax.grad(jax.vmap(f))(x)
    ↓ JAX tracing:先 vmap 变换,再 grad 变换
batching rule 错误 → batch 维度的 slice 索引算错
    ↓ XLA lowering:错误被编码进 HLO
HLO 执行:结果"看起来对"(只有梯度错,前向传播可能没问题)
    ↓ 用户检查梯度:跟期望值不一致
```

**关键**:前向传播可能正确,只有梯度错误。Agent 需要专门测试梯度才能发现。

## 为什么难

1. **JAX 抽象层多**:Python → JAX tracing → batching transform → XLA lowering → HLO → 执行(6 层)
2. **症状只在 vmap+grad 组合下出现**:单独用 vmap 或 grad 都正确
3. **需要理解 JAX transform 系统**:tracing、batching、jvp/vjp 的交互
4. **需要数值验证**:梯度错误需要 finite-difference 检查
5. **JAX 内部代码复杂**:batching rule 是框架内部实现,不是用户 API

## 预估轨迹

- **步数**:250+ 步
- **时间**:2-4 小时
- **关键转折点**:
  - 步骤 40:发现 vmap+grad 梯度不对
  - 步骤 100:确认不用 vmap 时梯度正确
  - 步骤 150:追溯到 JAX 的 batching transform
  - 步骤 200:找到具体的 batching rule
  - 步骤 250:定位 axis 索引错误

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| 不用 vmap,手写 batch 循环 | 性能测试(vmap 应该比循环快) |
| 用有限差分替代 autodiff | 检查代码是否禁用了 grad |
| 只改用户代码不改 JAX | diff 只允许 JAX 源码 |
| 用 PyTorch 替代 JAX | 检查依赖没有变化 |
| 后处理梯度(clamp/normalize) | grep 梯度处理代码 |

## Oracle

```bash
#!/bin/bash
cd /path/to/jax
git checkout HEAD -- jax/_src/lax/slicing.py
# JAX 不需要编译(Python),直接生效
```

## 资源需求

- **GPU**:可选(JAX CPU 模式也能验证)
- **编译**:不需要(纯 Python 修改)
- **验证**:1-2 分钟
- **磁盘**:~500MB

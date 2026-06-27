# /anti-hack — 反 hack 措施配置

## 用途

配置评测任务的反 hack 措施，防止 agent 绕过调试过程。

## 措施列表

### 1. 禁止上网搜索

**问题**： Kimi Code 会使用 WebSearch/WebFetch 工具上网搜索答案。

**解决**： 在 `run.sh` 的 kimi_config.toml 中添加 deny 规则：

```toml
[[permission.rules]]
decision = "deny"
pattern = "WebSearch"

[[permission.rules]]
decision = "deny"
pattern = "WebFetch"
```

**位置**： 放在 allow 规则之后，KIMEOF 之前。

**原理**： instruction.md 中规定"禁止访问外部网络"，但 Kimi Code 默认允许所有工具，需要显式 deny。

### 2. 禁止 git 查看历史

**问题**： agent 用 `git show` / `git diff` 直接看到注入的改动。

**解决**： 在 Dockerfile 中删除 .git 目录：

```dockerfile
RUN rm -rf /build/pytorch/.git
```

### 3. 禁止修改 Python 文件(CUDA 任务)

**问题**： agent 修改 train.py/model.py 而不是修复 CUDA kernel。

**解决**： 在 instruction.md 中明确约束 + test.sh 检查：

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

**问题**： agent 强制使用 CPU 绕过 CUDA bug。

**解决**： test.sh 中性能测试：

```bash
GPU_TIME=$(python train.py --steps 10 --device cuda --profile | grep avg_step_time)
CPU_TIME=$(python train.py --steps 10 --device cpu --profile | grep avg_step_time)
SPEEDUP=$(python -c "print($CPU_TIME / $GPU_TIME)")
if [ $(python -c "print(1 if $SPEEDUP > 1.5 else 0)") = "0" ]; then
    HACK=1
fi
```

### 5. 禁止 NaN 处理掩盖

**问题**： agent 在 Python 层加 nan_to_num/clip_grad 掩盖 CUDA bug。

**解决**： test.sh 中静态分析：

```bash
if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan" "$WORKSPACE/train.py"; then
    HACK=1
fi
```

### 6. 禁止绕过 vmap(JAX 任务)

**问题**： agent 手写 batch 循环替代 vmap。

**解决**： test.sh 中检查：

```bash
if grep -rn "jax.grad\|jax.vmap" "$WORKSPACE/test_vmap.py"; then
    echo "✅ 使用标准 JAX API"
else
    HACK=1
fi
```

### 7. 统一文件修改时间

**问题**： agent 用 `ls -la` 或 `stat` 查看文件修改时间，定位最近被修改的文件(即被注入 bug 的文件)。

**解决**： 在 **Dockerfile** 中注入 bug 后，用 `touch` 统一时间戳：

```dockerfile
# Task1: touch 被修改的 cuda 文件
RUN find aten/src/ATen/native/cuda -name "*.cu" -exec touch {} +

# Task4: touch 所有 python 文件
RUN find $JAX_DIR -name "*.py" -exec touch {} + 2>/dev/null; \
    find /build/jax -name "*.py" -exec touch {} + 2>/dev/null
```

**注意**： touch 只在 Dockerfile 里做，不要在 inject_bug.py 里重复。Dockerfile 的 touch 同时触发 ninja 重编译。

### 8. 判分逻辑防读（setuid + 非 root agent）

**问题**： 判分脚本 test.sh 以只读挂载给 agent，且 instruction 引导它跑 test.sh。
agent 直接 `cat test.sh` 把判分清单当"答案地图"，照着测试路径反向定位 bug
(实测： kimi 读到带电检查的 Python 代码，立刻针对性修复了所有被测算子)。

**解决**： agent 以非 root 运行，真 test.sh 锁进 root-only 目录，只通过 setuid
程序回显**总分**——agent 能用最终测试自测，但读不到判分逻辑。无 smoke/final gap
(grade 跑的就是最终 test 同一份 → 过 grade ⟺ 过最终)。

**Dockerfile** 关键层：
```dockerfile
COPY .../tests/test.sh /opt/judge/test.sh
COPY .../environment/grade.c /tmp/grade.c
RUN useradd -m -u 1500 agent && \
    cp -r /root/.kimi-code /opt/kimi-code && chmod -R a+rX /opt/kimi-code && \
    mkdir -p /home/agent/.kimi-code && chown -R agent:agent /home/agent && \
    chown -R root:root /opt/judge && chmod 700 /opt/judge && \
    gcc -O2 -o /usr/local/bin/grade /tmp/grade.c && chmod 4755 /usr/local/bin/grade && \
    mkdir -p /logs/verifier && chown -R root:root /logs && chmod 700 /logs/verifier && \
    chown -R agent:agent /build/pytorch /usr/local/lib/python3.12/dist-packages/torch/lib
ENV PATH="/opt/kimi-code/bin:$PATH"
```

**grade.c** (setuid-root，只回显总分，丢弃 test.sh 输出防泄漏分项)：
```c
setgid(0); setuid(0);                       // 提权
// fork 子进程: dup2 /dev/null 到 stdout/stderr, execle 固定 PATH 跑 /opt/judge/test.sh
// 读 /logs/verifier/reward.txt, 只 printf("score=%s\n", ...)
```
编译： `gcc -O2 -o /usr/local/bin/grade grade.c && chmod 4755 /usr/local/bin/grade`

**run.sh** 配套改动：
- agent 运行环境： `docker run --user 1500 -e HOME=/home/agent`，**去掉 tests 挂载**，
  config 挂到 `/home/agent/.kimi-code/config.toml`
- **最终评分镜像必须显式 `--user 0`**(否则 `docker commit` 继承 agent 的 USER=1500，
  非 root 读不到 /opt/judge/test.sh → reward 恒为 0.0，这是个易踩的坑)
- **max-history 计分**：test.sh 把每次分数追加到 `/logs/verifier/history.log`(与 reward.txt
  同在 root:700 目录，agent 无法伪造)；run.sh 末取 `sort -g history.log | tail -1` 当 reward，
  保证 agent 中途到过的最佳状态不被末态打断/改坏拖累。详见 generate-task.md "max-history 计分"。

**instruction.md**： 把 `bash /task/tests/test.sh` 全改为 `grade`，并说明"只返回总分，
需自己定位"。**注意别在 instruction 里泄露判分方法**(如"逐算子对比 CPU/CUDA""归约用 >1024 维
触发多 warp""部分 bug 是竞态需重复跑")——这些等于把最难那簇的答案给 agent；只留"冲满分"动机。

**原理 / 安全要点**：
- bash 脚本无法"可执行不可读"(解释器要读文件)，所以用 setuid 程序代跑
- setuid 进程 dumpable=0，同 uid 的 agent 无法 ptrace 或读其 fd
- grade.c 用 execle 固定 PATH/HOME，杜绝环境变量劫持 bash/python
- model.py/train.py 仍 workspace **只读挂载** + agent 非 root → 物理改不了，
  "禁改 model"自动强制，判分用原文件即可、无需副本，也堵死"借判分进程提权读答案"
- 局限： 对强模型只提高成本不质变(kimi 仍靠端到端对比逼近满分)，需配合 bug 本身
  的不可逆/需深推理才能真正拉开难度

## 应用方法

1. 修改 `tasks/taskN-xxx/run.sh` 添加权限规则
2. 修改 `tasks/taskN-xxx/task/tests/test.sh` 添加检测逻辑
3. 修改 `tasks/taskN-xxx/task/instruction.md` 添加约束说明
4. 修改 `tasks/taskN-xxx/task/environment/Dockerfile` 删除敏感文件

## 注意事项

- deny 规则要放在 allow 规则之后
- WebSearch/WebFetch 的 pattern 要精确匹配
- test.sh 的 HACK 检查会导致最终分数减半

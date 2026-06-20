# /generate-task — 生成一道跨语言 bug-fix 评测题

## 用途

生成一道面向 agentic RL 后训练的跨语言 bug-fix 评测题。
目标:前沿模型成功率 ~50%,轨迹步数 >200。

## 输入

```
/generate-task --cve CVE-2024-xxxx [--framework pytorch|tensorflow|numpy] [--bug-type rsqrt|off-by-one|overflow]
```

或手动指定:
```
/generate-task --bug-file path/to/kernel.cu --bug-line 246 --bug-desc "rsqrt missing eps"
```

## 执行流程

### Phase 1: 分析 Bug

1. **定位 bug 文件和行号**
   - 如果提供 CVE:读 CVE 描述 → 找到对应源码文件 → 确定 bug 位置
   - 如果提供 bug-file:直接使用

2. **分析 bug 类型**
   - 数值稳定性(rsqrt/eps/NaN)
   - 越界访问(off-by-one/index error)
   - 竞态条件(race condition/GIL)
   - 类型错误(dtype/overflow)
   - 逻辑错误(wrong comparison/wrong formula)

3. **确定触发条件**
   - 随机数据就能触发(最理想)
   - 需要特定输入才能触发(需要设计数据注入)
   - 需要特定环境才能触发(需要设计环境)

### Phase 2: 设计模型结构

**核心原则**:bug 所在的 op 不能是模型的第一层,也不能在最后一层。藏在中间。

```python
# 推荐结构(10+ 种 op):
Conv2d → BatchNorm → ReLU → Conv2d → GELU → MaxPool → Flatten
    → Linear → [buggy op] → ReLU
    → Linear → [same op] → Dropout
    → Linear → 输出
```

**关键**:用自定义封装类隐藏真实 op 名称。例如:
```python
class FeatureNorm(nn.Module):  # 不直接暴露 nn.LayerNorm
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
```

**模型文件**:放在 `model.py`,训练脚本只 import。

### Phase 3: 设计训练脚本

**核心原则**:不暴露任何 bug 细节。

```python
# train.py 示例结构
from model import ImageClassifier  # 不直接 import nn.LayerNorm

def make_batch(batch_size, device):
    return torch.randn(batch_size, 3, 32, 32, device=device), torch.randint(0, 10, (batch_size,), device=device)

# 没有 hook、没有特殊注入、纯随机数据
# bug 由 CUDA kernel 自身触发
```

**禁止**:
- ❌ 不要在 train.py 里提到 bug 所在的 op 名称
- ❌ 不要加 hook / 注入 / 特殊处理
- ❌ 不要加 NaN 检查 / 梯度裁剪 / try-catch

### Phase 4: 设计诱饵

**目标**:20 个 CUDA 文件被修改,只有 1 个是真 bug。

**诱饵类型**(优先用真实代码改动):
1. 改常量(如 eps 精度、阈值)
2. 改比较符(< → <=, >= → >)
3. 改循环边界(off-by-one 但不触发)
4. 改数值精度(cast 类型)
5. 添加死代码(if false && ...)

**诱饵要求**:
- ✅ 编译通过
- ✅ 运行不 NaN
- ✅ 看起来像真 bug
- ❌ 不能是纯注释(太容易排除)

**注入脚本**:写成 `inject_bug.py`,包含:
- 先恢复干净版(防缓存问题)
- 注入 1 个真 bug
- 注入 19 个诱饵
- 打印注入结果

### Phase 5: 设计 instruction.md

**核心原则**:不暴露任何 bug 细节。

```
# 任务:修复 PyTorch 训练 NaN 问题

## 背景
我们使用了一个从源码编译的 PyTorch。训练时 loss 变成 NaN。

## 你的任务
1. 理解 bug
2. 定位 bug(在 /build/pytorch/ 的 CUDA 源码中)
3. 修复 bug(只允许改 .cu/.cpp/.h)
4. 重新编译(ninja -j32 lib/libtorch_cuda.so && cp ...)
5. 验证(bash /task/tests/test.sh)
6. 检查分数(≥ 0.6 才通过,< 0.6 继续排查)

## 约束条件
- 只允许改 C++/CUDA 文件
- 不允许改 Python 文件
- 不允许加 NaN 检查 / 梯度裁剪 / try-catch
```

**禁止在 instruction.md 里提到**:
- ❌ bug 所在的具体文件名
- ❌ bug 的具体类型(rsqrt / off-by-one / ...)
- ❌ bug 的修复方法
- ❌ 任何 hook / 注入机制

### Phase 6: 设计 test.sh

**核心原则**:只检查结果,不暴露 bug 细节。

评分标准(满分 1.0):
```
0.10  基础:框架可导入
0.05  基础:GPU 可用
0.25  核心:多 seed 无 NaN
0.15  多 batch_size 通过
0.20  性能:GPU 比 CPU 快 2x+
0.25  Anti-hack:hook 存在 + 无掩盖 + 文件未被修改
```

**禁止在 test.sh 里暴露**:
- ❌ bug 所在的文件名
- ❌ bug 的具体模式(rsqrt / off-by-one)
- ❌ 修复方法

**Anti-hack 检查**:
- NaN 掩盖(nan_to_num / clip_grad / clamp)
- CPU 回退
- 关键文件是否被修改(train.py / model.py)

### Phase 7: 设计 Dockerfile

**两阶段构建**:

```dockerfile
# Dockerfile.base (一次性,1-2 小时):
# - 从源码编译框架
# - 保留 .git / third_party / build 目录
# - 支持 ninja 增量编译

# Dockerfile (增量,5-15 分钟):
FROM <base-image>
COPY inject_bug.py /tmp/
RUN python /tmp/inject_bug.py && rm /tmp/inject_bug.py
RUN git commit --no-verify  # 隐藏改动
RUN ninja -j32 lib/<target>.so && cp ...
CMD ["/bin/bash"]
```

**关键**:
- 用 `--mount=type=secret` 挂载 SSH key(不烤进镜像)
- 用 `--build-arg CACHE_BUST=$(date +%s)` 强制重编
- API key 运行时注入(`docker run -e` 或 `-v`)

### Phase 8: 验证 + 校准

1. **验证 NaN**: `docker run --rm --gpus all task1 python /workspace/train.py --steps 100`
2. **验证 CPU 不 NaN**: `docker run --rm task1 python /workspace/train.py --steps 100 --device cpu`
3. **验证 git diff 为空**: `docker run --rm task1 bash -c 'cd /build/pytorch && git diff --stat HEAD'`
4. **跑 Kimi 校准**: `./calibrate.sh kimi-code/kimi-for-coding 10 3`
5. **检查 reward**: 目标 40-60%(太简单 > 60%,太难 < 40%)

### Phase 9: 输出

最终输出:
```
tasks/taskN-<name>/
├── README.md
├── run.sh / calibrate.sh
└── task/
    ├── task.toml / instruction.md
    ├── environment/Dockerfile / Dockerfile.base
    ├── workspace/train.py / model.py
    ├── solution/inject_bug.py / solve.sh
    └── tests/test.sh
```

## 参考实现

完整的 Task 1 实现见: `tasks/task1-pytorch-cuda-index/README.md`

## 关键经验(踩坑记录)

1. **Bug 必须始终触发**(随机数据下),不能依赖 hook 注入特殊数据
2. **model.py 用封装类隐藏真实 op 名称**(FeatureNorm 而非 LayerNorm)
3. **train.py 不暴露任何 bug 细节**(无 hook、无特殊注入)
4. **test.sh 只检查结果,不暴露 bug 细节**
5. **诱饵用真实代码改动**,不只是注释
6. **inject_bug.py 先恢复干净版再注入**(防缓存问题)
7. **git commit 隐藏改动**(防 agent 用 git diff)
8. **Dockerfile 用 --mount=type=secret**(key 不烤进镜像)
9. **API key 运行时注入**(不烤进镜像)
10. **用 docker commit 保留 agent 修复**(同一个容器里跑 test.sh)

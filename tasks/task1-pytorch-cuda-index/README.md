# Task 1: PyTorch LayerNorm CUDA Off-by-One → 训练 NaN

## 概述

在 PyTorch 源码的 LayerNorm CUDA forward kernel 中注入一个 off-by-one 错误,然后从源码编译。
Bug 在 CUDA 层触发,但症状在 Python 层表现为训练 loss 变为 NaN,
且**延迟 ~90 个 iteration 才显现** —— 因为越界写入的累积效应需要多次迭代才破坏到关键内存。

## Bug 设计

- **位置**:`/build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu` → `LayerNormForwardCUDAKernel`
- **类型**:off-by-one(`for (j = threadIdx.x; j < N; ...)` 改成 `j <= N`)
- **注入方式**:Dockerfile 中用 `sed` 命令注入
- **效果**:越界读写 1 个元素,破坏相邻内存
- **触发条件**:每次 LayerNorm forward 都触发

## 延迟显现机制

```
LayerNorm CUDA kernel 越界读写 → 破坏相邻 tensor 的 1 个元素
    ↓ 该 tensor 参与后续计算(Linear + ReLU)
误差在后续层传播 → 梯度计算受影响
    ↓ 累积 ~90 个 iteration
梯度爆炸 → loss NaN
```

**关键**:前 89 个 iteration 都"看起来正常",第 90 个才 NaN。
Agent 不能一眼看到 bug。

## 为什么难

1. **跨语言**:症状在 Python,bug 在 CUDA,需要读两种语言
2. **症状模糊**:NaN 可能是 100 种原因(learning rate / 数据 / 模型 / bug)
3. **无 stack trace**:NaN 不指向 CUDA kernel,指向 Python 层的 loss 计算
4. **需要理解 CUDA 编程**:线程索引、越界访问、内存布局
5. **延迟显现**:bug 执行后跑了大量操作才出现症状
6. **代码量大**:PyTorch 源码有几百万行,需要在其中定位 bug

## 预估轨迹

- **步数**:150-300 步
- **时间**:2-4 小时(含 PyTorch 重新编译)
- **关键转折点**:
  - 步骤 30:发现 NaN,开始排查
  - 步骤 80:定位到 LayerNorm 相关
  - 步骤 150:在 PyTorch 源码中找到 layer_norm_kernel.cu
  - 步骤 200:读懂 CUDA 内核逻辑
  - 步骤 250:找到 off-by-one bug

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `isnan` / `nan_to_num` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 改训练参数 | diff 检查 |
| 注释掉越界代码 | 检查源码修改 |

## Oracle

```bash
#!/bin/bash
# 撤销 off-by-one: j <= N → j < N
sed -i '/__global__ void LayerNormForwardCUDAKernel/,/^}/s/for (int64_t j = threadIdx.x; j <= N;/for (int64_t j = threadIdx.x; j < N;/' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
# 重新编译 PyTorch
cd /build/pytorch && python setup.py develop
```

## 资源需求

- **GPU**:1× RTX A6000(4GB+)
- **编译**:30-60 分钟(首次 Docker build)
- **增量编译**:5-15 分钟(修复后重新编译)
- **训练验证**:1-2 分钟(100 iteration)
- **磁盘**:~30GB(PyTorch 源码 + 编译产物)

## 关于使用 SSH

Dockerfile 中所有 git 操作(clone / submodule)均使用 SSH(`git@github.com:...`)而非 HTTPS。
原因:HTTPS 在 Docker build 过程中**经常超时或连接中断**,尤其在 clone 大型仓库(PyTorch ~2GB)和初始化子模块时。
SSH 连接更稳定,适合长时间的 Docker build。

构建命令:
```bash
eval $(ssh-agent) && ssh-add ~/.ssh/id_rsa
DOCKER_BUILDKIT=1 docker build --ssh default -t task1 -f Dockerfile .
```

---

## 开发踩坑记录

### 坑 1:自定义 CUDA 扩展 vs PyTorch 源码编译

**问题**:最初用自定义 CUDA 扩展(`torch.utils.cpp_extension.load()`)实现 buggy LayerNorm。
但这导致 agent 一眼就能看到 bug 所在(只需要看 `layernorm_cuda/` 目录),难度大幅下降。

**解决**:改为在 PyTorch 源码中注入 bug,然后从源码编译。这样 agent 需要在 PyTorch 的几百万行代码中定位 bug。

**结论**:对于跨语言 bug-fix 任务,**bug 应该在框架源码中**,而不是在独立的自定义扩展中。

### 坑 2:PyTorch 编译在开发机上失败

**问题**:在开发机上直接编译 PyTorch 时,遇到子模块初始化失败、CMake 配置错误等问题。

**解决**:PyTorch 编译应该在 **Docker 容器**中进行。Dockerfile 负责:
1. 使用 PyTorch 官方 CUDA 镜像(有完整编译环境)
2. Clone 源码并初始化子模块
3. 注入 bug
4. 编译 PyTorch

**结论**:后续题目如果需要从源码编译大型项目,**必须用 Dockerfile**,不要在开发机上硬编。

### 坑 3:子模块初始化需要选择性

**问题**:`git submodule update --init --recursive` 会拉取所有子模块(包括 ideep、kineto 等),经常失败。

**解决**:只初始化必要的子模块,跳过不需要的:
```dockerfile
RUN git submodule update --init \
    third_party/pybind11 \
    third_party/cpuinfo \
    ...
    || true
```

### 坑 4:编译选项需要最小化

**问题**:PyTorch 默认编译所有组件(caffe2、kineto、fbgemm 等),编译时间长且容易出错。

**解决**:用环境变量禁用不需要的组件:
```dockerfile
ENV USE_IDEEP=0
ENV USE_MKLDNN=0
ENV USE_KINETO=0
ENV BUILD_CAFFE2=0
...
```

### 通用经验总结

| 经验 | 说明 |
|---|---|
| **bug 在框架源码中,不在自定义扩展中** | agent 需要在海量代码中定位,难度更高 |
| **编译在 Docker 中进行** | 开发机不需要编译,只需要写 Dockerfile |
| **子模块选择性初始化** | 只拉必要的,跳过 ideep/kineto 等 |
| **编译选项最小化** | 禁用不需要的组件,减少编译时间 |
| **sed 注入 bug** | 简单、可复现、易于 oracle 撤销 |
| **git 使用 SSH** | 比 HTTPS 更稳定,避免 Docker build 时超时 |
| **inject_bug.py 放在 solution/** | 不放在 workspace/,防止 agent 直接读到答案 |

# Task 1: PyTorch LayerNorm CUDA rsqrt Bug → 训练 NaN

## 概述

在 PyTorch 源码的 LayerNorm CUDA forward kernel 中注入一个 bug:去掉 `rsqrt` 的 `eps` 保护。
当方差 = 0 时(`rsqrt(0)` = Inf → `0 * Inf` = NaN),训练 loss 变 NaN。
每 20 步注入零方差 batch 触发,约第 90 步 NaN 出现。

## Bug 设计

- **位置**: `aten/src/ATen/native/cuda/layer_norm_kernel.cu` 第 246 行
- **类型**: `rsqrt(wd.sigma2 + eps)` → `rsqrt(wd.sigma2)` (去掉 eps 保护)
- **注入方式**: `solution/inject_bug.py` 注入 1 个真 bug + 19 个诱饵
- **效果**: 当方差 = 0 时,`rsqrt(0)` = Inf,`(x - mean) * Inf` = NaN
- **触发条件**: 零方差输入(batch 内所有特征相同)

## 延迟显现机制

```
每 20 步:注入零方差 batch(所有特征 = 相同值)
    ↓ LayerNorm 看到方差 = 0
rsqrt(0) = Inf → (x - mean) * Inf = NaN
    ↓ NaN 传播到 loss → NaN 梯度 → 模型权重变 NaN
    ↓ 后续所有步骤都是 NaN
第 ~90 步:loss 检测到 NaN
```

## 为什么难

1. **跨语言**: 症状在 Python(loss NaN),bug 在 CUDA(rsqrt)
2. **症状模糊**: NaN 可能是 100 种原因(learning rate / 数据 / 模型 / bug)
3. **无 stack trace**: NaN 不指向 CUDA kernel
4. **19 个诱饵**: 20 个 CUDA 文件被修改,只有 1 个是真 bug
5. **需要理解 CUDA 编程**: LayerNorm 内部实现、Welford 算法、rsqrt 数值稳定性
6. **代码量大**: PyTorch 源码几百万行

## 预估轨迹

- **步数**: 200-400 步
- **时间**: 2-4 小时
- **关键转折点**:
  - 步骤 30: 发现 NaN,开始排查
  - 步骤 80: 定位到 LayerNorm 相关
  - 步骤 150: 在 PyTorch 源码中找到 layer_norm_kernel.cu
  - 步骤 200: 读懂 CUDA 内核逻辑(Welford 算法)
  - 步骤 250: 找到 rsqrt 缺少 eps 的 bug
  - 步骤 300: 修复 + 增量重编 + 验证

## Anti-hack 措施

| Hack 路径 | 检测方法 |
|---|---|
| Python 层加 NaN 检查 | grep `isnan` / `nan_to_num` |
| 梯度裁剪掩盖 | grep `clip_grad_norm` |
| CPU 回退 | 性能测试(GPU 应比 CPU 快 2x+) |
| 换用原生 LayerNorm | 检查 train.py 是否修改 |
| 改训练参数 | diff 检查 |
| 用 try/catch 吞错误 | grep `try:` |
| monkey-patch | grep `setattr` |

## Oracle

`solution/solve.sh`:
```bash
# 恢复 eps 保护
sed -i 's/rsqrt(wd\.sigma2)/rsqrt(wd.sigma2 + eps)/' \
    /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu
# 增量重编
cd /build/pytorch/build && ninja -j32 lib/libtorch_cuda.so
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

## 资源需求

- **GPU**: 1× RTX A6000(4GB+)
- **fat base 构建**: 1-2 小时(一次性)
- **task1 增量构建**: 5-15 分钟
- **训练验证**: 1-2 分钟
- **磁盘**: ~30GB(fat base) + ~1GB(task1)

## 架构

```
pytorch-2.5.0-fat-base (一次性构建):
  PyTorch 2.5.0 源码 + .git + third_party + build 目录
  支持 ninja 增量编译

task1 (增量构建,5-15 分钟):
  FROM fat-base
  → inject_bug.py 注入 1 真 bug + 19 诱饵
  → ninja 增量编译(只重编 layer_norm_kernel.cu.o + 链接)
  → cp .so 到 site-packages
  → 复制 train.py
```

---

## 开发踩坑记录

### 坑 1: 自定义 CUDA 扩展 vs PyTorch 源码编译

**问题**: 最初用自定义 CUDA 扩展(`torch.utils.cpp_extension.load()`)实现 buggy LayerNorm。agent 一眼就能看到 bug 所在(只需看 `layernorm_cuda/` 目录),难度太低。

**解决**: 改为在 PyTorch 源码中注入 bug。agent 需要在几百万行代码中定位。

**结论**: bug 应该在框架源码中,不在独立扩展中。

### 坑 2: PyTorch 从源码编译极其复杂

**问题**: `git submodule update --init --recursive` 经常失败,CMake 配置错误,编译 1-2 小时。

**解决**: 两阶段构建:
1. `Dockerfile.base`: 完整编译一次(1-2 小时),保留 .git/third_party/build
2. `Dockerfile`: 基于 fat base,只做增量编译(5-15 分钟)

**结论**: 大型项目编译必须分"base 构建"和"增量构建"两层。

### 坑 3: `setup.py install` 不支持真正的增量编译

**问题**: 改了 .cu 源码后,`setup.py install` 检测到已安装就跳过编译。

**解决**: 直接用 `ninja` 命令:
```bash
cd /build/pytorch/build
ninja -j32 lib/libtorch_cuda.so  # 只重编改过的 .o + 重新链接
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

**结论**: `setup.py install` 不可靠,`ninja` 直接调用更可控。

### 坑 4: `rsqrt` 去掉 eps 在随机数据下不触发 NaN

**问题**: `rsqrt(wd.sigma2)` 和 `rsqrt(wd.sigma2 + eps)` 在随机数据下几乎没区别(方差从不精确等于 0)。

**解决**: 在 train.py 中每 20 步注入零方差 batch(所有特征设为相同值),强制触发 `rsqrt(0)`。

**结论**: 数值稳定性 bug 需要专门构造边界数据才能触发,不能依赖随机数据。

### 坑 5: BatchNorm 破坏零方差

**问题**: 最初模型有 BatchNorm,它会把零方差输入"修"成非零方差,导致 LayerNorm 看不到零方差。

**解决**: 去掉 BatchNorm,让 LayerNorm 成为第一层,零方差输入直达 LayerNorm。

**结论**: 模型结构设计需要考虑 bug 触发路径,中间层可能破坏边界条件。

### 坑 6: `rsqrt(m2)` 改错位置

**问题**: 最初 sed 替换 `rsqrt(m2 + eps)` → `rsqrt(m2)`,但改的是 `RowwiseMomentsCUDAKernel`(计算方差的 kernel),不是 `LayerNormForwardCUDAKernel`(使用方差的 kernel)。

**解决**: 精确匹配目标: `rsqrt(wd.sigma2 + eps)` → `rsqrt(wd.sigma2)`(forward kernel 的变量名是 `wd.sigma2`,不是 `m2`)。

**结论**: 同一文件可能有多个相似代码,sed 模式必须精确匹配目标位置。

### 坑 7: ninja 目标路径

**问题**: `ninja torch/lib/libtorch_cuda.so` 报错"unknown target"。

**解决**: 正确目标是 `lib/libtorch_cuda.so`(在 build 目录下,不是 torch 子目录)。用 `ninja -t targets all | grep libtorch_cuda` 查找。

**结论**: ninja 目标名跟文件路径不完全一致,需要查 `ninja -t targets`。

### 坑 8: `.so` 文件需要手动复制到 site-packages

**问题**: ninja 编译的 .so 在 `build/lib/`,但 Python 用的是 `/usr/local/lib/python3.12/dist-packages/torch/lib/` 里的。两者不同步。

**解决**: 编译后手动 `cp`:
```bash
cp build/lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/
```

**结论**: 增量编译后必须同步 .so 到 Python site-packages,否则 Python 用的还是旧版。

### 坑 9: SSH key 格式兼容性

**问题**: ed25519 key 在 Docker 容器的 OpenSSL 下报 `error in libcrypto`。

**解决**: 改用 RSA key(`ssh-keygen -t rsa -b 4096`),兼容所有 OpenSSL 版本。

**结论**: SSH key 优先用 RSA,ed25519 在某些环境下不兼容。

### 坑 10: `--mount=type=secret` 不能 chmod

**问题**: BuildKit secret mount 是只读的,`chmod 600` 报"Read-only file system"。

**解决**: 去掉 chmod,git 不严格要求 key 文件权限(在容器内)。

**结论**: Docker BuildKit secret 是只读的,不能修改挂载的文件。

### 通用经验总结

| 经验 | 说明 |
|---|---|
| **bug 在框架源码中** | agent 需要在海量代码中定位 |
| **两阶段构建** | fat base(1-2h) + 增量(5-15min) |
| **ninja 直接编译** | 不依赖 setup.py |
| **手动 cp .so 到 site-packages** | 编译产物和 Python 安装路径不同 |
| **inject_bug.py 统一注入** | 1 真 bug + 19 诱饵,集中管理 |
| **构造边界数据触发 bug** | 数值稳定性 bug 需要专门的输入 |
| **模型结构适配 bug 路径** | 去掉可能破坏边界条件的中间层 |
| **RSA SSH key** | 兼容性最好 |
| **BuildKit secret 只读** | 不能 chmod |

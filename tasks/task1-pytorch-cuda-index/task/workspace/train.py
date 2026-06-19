#!/usr/bin/env python3
"""
训练脚本 — 验证 PyTorch CUDA bug 是否导致 NaN
用法: python train.py [--steps N] [--seed S] [--device D] [--profile]
"""

import argparse
import time
import os
import torch
import torch.nn as nn

# 加载自定义 CUDA LayerNorm 扩展
def load_layernorm_cuda():
    from torch.utils.cpp_extension import load
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layernorm_cuda')
    return load(
        name='layernorm_cuda',
        sources=[
            os.path.join(src_dir, 'layernorm_cuda.cpp'),
            os.path.join(src_dir, 'layernorm_cuda_kernel.cu'),
        ],
        verbose=False
    )

_layernorm_module = None

class CudaLayerNorm(nn.Module):
    """使用自定义 CUDA kernel 的 LayerNorm(有 bug)"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        global _layernorm_module
        if _layernorm_module is None:
            _layernorm_module = load_layernorm_cuda()
        return _layernorm_module.forward(x, self.weight, self.bias, self.eps)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="训练步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--hidden", type=int, default=512, help="隐藏层大小")
    parser.add_argument("--profile", action="store_true", help="输出性能统计")
    return parser.parse_args()

def make_model(hidden, device):
    """创建一个使用 CudaLayerNorm + ReLU 的简单模型"""
    return nn.Sequential(
        nn.Linear(256, hidden),
        CudaLayerNorm(hidden),  # ← 这个 op 的 CUDA kernel 有 bug
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        CudaLayerNorm(hidden),  # ← 多调几次,增加触发概率
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        CudaLayerNorm(hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    ).to(device)

def make_batch(batch_size, device):
    """生成随机训练数据"""
    x = torch.randn(batch_size, 256, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = make_model(args.hidden, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    nan_detected = False
    losses = []
    times = []

    for step in range(args.steps):
        t0 = time.time()
        x, y = make_batch(args.batch_size, device)

        output = model(x)
        loss = loss_fn(output, y)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_detected = True
            print(f"[step {step}] NaN/Inf detected! loss={loss.item()}")
            # 继续跑几步看传播
            if step > args.steps - 5:
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1 = time.time()
        losses.append(loss.item())
        times.append(t1 - t0)

        if step % 20 == 0 or nan_detected:
            print(f"step={step:4d}  loss={loss.item():.6f}  time={t1-t0:.4f}s")

    # 最终报告
    final_loss = losses[-1] if losses else float('nan')
    print(f"\nfinal_loss {final_loss}")
    print(f"nan_detected {nan_detected}")

    if args.profile:
        avg_time = sum(times) / len(times) if times else 0
        print(f"avg_step_time {avg_time:.4f}")
        print(f"total_time {sum(times):.4f}")

if __name__ == "__main__":
    main()

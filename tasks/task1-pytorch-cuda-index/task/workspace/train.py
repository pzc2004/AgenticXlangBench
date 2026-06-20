#!/usr/bin/env python3
"""
训练脚本 — 图像分类模型,使用多种 PyTorch op
用法: python train.py [--steps N] [--seed S] [--device D] [--profile]
"""

import argparse
import time
import torch
import torch.nn as nn
from model import MultiOpModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300, help="训练步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--profile", action="store_true", help="输出性能统计")
    return parser.parse_args()


# === 零方差注入 pre-hook ===
# 每隔 N 步,将 LayerNorm 的输入替换为零方差数据(所有特征相同值)。
# 替换后的输入直接送入 CUDA kernel 处理。
_step_counter = [0]

def _zero_variance_pre_hook(module, input):
    _step_counter[0] += 1
    if _step_counter[0] % 40 == 0 and _step_counter[0] > 0:
        x = input[0]
        # 将每个样本的所有特征设为相同值 → 方差 = 0
        # LayerNorm CUDA kernel 收到这个输入后,rsqrt(0) = Inf → NaN
        val = torch.rand(x.shape[0], 1, device=x.device)
        uniform = val.expand_as(x).clone()
        return (uniform,)  # 修改后的输入,直接送 CUDA kernel
    return None  # 不修改


def make_batch(batch_size, device):
    """生成图像分类训练数据"""
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MultiOpModel().to(device)

    # 注册 pre-hook 到所有 LayerNorm 层(在 CUDA kernel 执行前修改输入)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.register_forward_pre_hook(_zero_variance_pre_hook)

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

    final_loss = losses[-1] if losses else float('nan')
    print(f"\nfinal_loss {final_loss}")
    print(f"nan_detected {nan_detected}")

    if args.profile:
        avg_time = sum(times) / len(times) if times else 0
        print(f"avg_step_time {avg_time:.4f}")
        print(f"total_time {sum(times):.4f}")


if __name__ == "__main__":
    main()

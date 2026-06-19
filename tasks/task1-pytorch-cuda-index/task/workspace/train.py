#!/usr/bin/env python3
"""
训练脚本 — 验证 PyTorch CUDA bug 是否导致 NaN
用法: python train.py [--steps N] [--seed S] [--device D] [--profile]
"""

import argparse
import time
import torch
import torch.nn as nn

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
    """创建一个使用 LayerNorm + ReLU 的简单模型"""
    return nn.Sequential(
        nn.Linear(256, hidden),
        nn.LayerNorm(hidden),  # ← 这个 op 的 CUDA forward kernel 有 bug
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),  # ← 多调几次,增加触发概率
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.LayerNorm(hidden),
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

#!/usr/bin/env python3
"""
训练脚本 — 简单图像分类模型
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
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--profile", action="store_true", help="输出性能统计")
    return parser.parse_args()

class SimpleModel(nn.Module):
    """简单分类模型"""
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 512)
        self.relu1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(self.ln1(x)))
        x = self.relu2(self.fc2(self.ln2(x)))
        x = self.fc3(x)
        return x

def make_batch(batch_size, device, step):
    """生成训练数据"""
    x = torch.randn(batch_size, 256, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    nan_detected = False
    losses = []
    times = []

    for step in range(args.steps):
        t0 = time.time()
        x, y = make_batch(args.batch_size, device, step)

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

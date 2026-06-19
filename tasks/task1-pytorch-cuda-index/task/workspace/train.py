#!/usr/bin/env python3
"""
训练脚本 — 使用多种 PyTorch op,验证 CUDA bug 是否导致 NaN
用法: python train.py [--steps N] [--seed S] [--device D] [--profile]

模型使用 10+ 种不同 op,agent 需要判断哪个 op 的 CUDA kernel 有 bug。
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

class MultiOpModel(nn.Module):
    """使用多种 op 的模型,用于测试不同 CUDA kernel"""
    def __init__(self):
        super().__init__()
        # Conv2d + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Conv2d + GELU
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gelu = nn.GELU()

        # MaxPool
        self.pool = nn.MaxPool2d(2)

        # Conv2d + SiLU
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.silu = nn.SiLU()

        # LayerNorm ← 真正的 bug 在这里
        self.flatten = nn.Flatten()
        self.ln1 = nn.LayerNorm(128 * 16 * 16)
        self.ln2 = nn.LayerNorm(512)

        # Linear + Dropout
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Conv2d → BatchNorm → ReLU
        x = self.relu1(self.bn1(self.conv1(x)))
        # Conv2d → GELU
        x = self.gelu(self.conv2(x))
        # MaxPool
        x = self.pool(x)
        # Conv2d → SiLU
        x = self.silu(self.conv3(x))
        # Flatten → LayerNorm
        x = self.flatten(x)
        x = self.ln1(x)  # ← bug 触发点
        # Linear → LayerNorm → Dropout
        x = self.fc1(x)
        x = self.ln2(x)  # ← 多调几次
        x = self.dropout(x)
        # Linear
        x = self.fc2(x)
        return x

def make_batch(batch_size, device):
    """生成随机训练数据(图像分类)"""
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

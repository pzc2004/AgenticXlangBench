#!/usr/bin/env python3
"""
图像分类训练脚本(用于 autograd 测试)
用法: python train.py [--epochs N] [--seed S] [--device D] [--eval_fixed_data]
"""

import argparse
import time
import torch
import torch.nn as nn
from model import SimpleClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="每轮步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--eval_fixed_data", action="store_true",
                        help="用固定数据评估(使 CPU/CUDA 结果可比较)")
    return parser.parse_args()


def make_batch(batch_size, device):
    """生成图像分类训练数据"""
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def make_fixed_dataset(num_samples, batch_size, device, seed=12345):
    """生成固定数据集"""
    torch.manual_seed(seed)
    batches = []
    for _ in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - _)
        x = torch.randn(bs, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        batches.append((x, y))
    return batches


def compute_accuracy(model, batches, device):
    """计算 accuracy"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct, total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    nan_detected = False
    losses = []

    if args.eval_fixed_data:
        num_train = args.batch_size * 10
        train_batches = make_fixed_dataset(num_train, args.batch_size, device, seed=args.seed)
        eval_batches = make_fixed_dataset(num_train, args.batch_size, device, seed=args.seed)
    else:
        train_batches = None
        eval_batches = None

    total_steps = args.epochs * args.steps_per_epoch
    for step in range(total_steps):
        if train_batches is not None:
            x, y = train_batches[step % len(train_batches)]
            x, y = x.to(device), y.to(device)
        else:
            x, y = make_batch(args.batch_size, device)

        output = model(x)
        loss = loss_fn(output, y)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_detected = True
            print(f"[step {step}] NaN/Inf detected! loss={loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        epoch = step // args.steps_per_epoch
        if step % args.steps_per_epoch == 0:
            avg_loss = sum(losses[-args.steps_per_epoch:]) / args.steps_per_epoch
            print(f"epoch={epoch:3d}  avg_loss={avg_loss:.6f}")

    final_loss = sum(losses[-args.steps_per_epoch:]) / args.steps_per_epoch
    print(f"\nfinal_loss {final_loss}")
    print(f"nan_detected {nan_detected}")

    # 计算最终 accuracy
    if eval_batches is not None:
        correct, total = compute_accuracy(model, eval_batches, device)
    else:
        eval_data = make_fixed_dataset(320, args.batch_size, device, seed=99999)
        correct, total = compute_accuracy(model, eval_data, device)

    print(f"accuracy {correct} {total}")
    print(f"final_accuracy {correct/total*100:.1f}%")


if __name__ == "__main__":
    main()

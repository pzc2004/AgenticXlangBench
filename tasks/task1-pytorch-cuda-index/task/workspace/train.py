#!/usr/bin/env python3
"""
图像分类训练脚本
用法: python train.py [--steps N] [--seed S] [--device D] [--profile] [--eval_fixed_data]
"""

import argparse
import time
import torch
import torch.nn as nn
from model import ImageClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="训练步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--profile", action="store_true", help="输出性能统计")
    parser.add_argument("--eval_fixed_data", action="store_true",
                        help="用固定数据训练和评估(使 CPU/CUDA 结果可比较)")
    return parser.parse_args()


def make_batch(batch_size, device):
    """生成图像分类训练数据"""
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    return x, y


def make_fixed_dataset(num_samples, batch_size, device, seed=12345):
    """生成固定数据集(用于 CPU/CUDA 可比较的评估)"""
    torch.manual_seed(seed)
    batches = []
    for _ in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - _)
        x = torch.randn(bs, 3, 32, 32, device=device)
        y = torch.randint(0, 10, (bs,), device=device)
        batches.append((x, y))
    return batches


def compute_accuracy(model, batches, device):
    """在固定数据集上计算 accuracy"""
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
    model = ImageClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    nan_detected = False
    losses = []
    times = []

    if args.eval_fixed_data:
        # 固定数据模式:用固定 seed 生成数据集(使 CPU/CUDA 可比较)
        # 数据量适中,正确梯度可记忆,错误梯度不行
        num_train = args.batch_size * 10  # 10 个 batch 的数据量
        train_batches = make_fixed_dataset(num_train, args.batch_size, device, seed=args.seed)
        eval_batches = make_fixed_dataset(num_train, args.batch_size, device, seed=args.seed)
    else:
        train_batches = None
        eval_batches = None

    for step in range(args.steps):
        t0 = time.time()

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

    # 计算最终 accuracy
    if eval_batches is not None:
        correct, total = compute_accuracy(model, eval_batches, device)
        print(f"accuracy {correct} {total}")
        print(f"final_accuracy {correct/total*100:.1f}%")
    else:
        # 在最后一批训练数据上计算 accuracy
        if losses:
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for x, y in (train_batches or [make_batch(args.batch_size, device)]):
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            model.train()
            print(f"accuracy {correct} {total}")
            print(f"final_accuracy {correct/total*100:.1f}%")

    if args.profile:
        avg_time = sum(times) / len(times) if times else 0
        print(f"avg_step_time {avg_time:.4f}")
        print(f"total_time {sum(times):.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for custom CUDA conv2d model.

Generates synthetic data and trains a simple conv model. Reports accuracy
across different input size configurations to detect gradient bugs.

Usage:
    python train.py --input_size 28 --kernel_size 4 --stride 3 --epochs 20
    python train.py --input_size 32 --kernel_size 3 --stride 2 --epochs 20
    python train.py --input_size 28 --kernel_size 4 --stride 3 --epochs 20 --device cpu
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvModel, CPUConvModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train conv model on synthetic data")
    parser.add_argument("--input_size", type=int, default=28, help="Input spatial size (H=W)")
    parser.add_argument("--kernel_size", type=int, default=3, help="Conv kernel size")
    parser.add_argument("--stride", type=int, default=1, help="Conv stride for first layer")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=2000, help="Total training samples")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--profile", action="store_true", help="Output timing info")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate (no training)")
    parser.add_argument("--use_ref", action="store_true",
                        help="Use CPU reference model (nn.Conv2d) instead of custom CUDA")
    return parser.parse_args()


def make_synthetic_data(input_size, num_samples, num_classes, batch_size, device, seed=42):
    """Generate synthetic classification data.

    Each class has a distinct spatial pattern so the model can learn to
    distinguish them. This ensures that correct gradients lead to high accuracy.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Create class-specific patterns: each class has a unique frequency pattern
    data_list = []
    label_list = []

    samples_per_class = num_samples // num_classes
    for c in range(num_classes):
        # Class pattern: sinusoidal pattern with class-specific frequency
        freq = (c + 1) * 0.5
        phase = c * 0.3

        # Generate base pattern
        y_grid = torch.linspace(0, freq * 3.14159, input_size)
        x_grid = torch.linspace(0, freq * 3.14159, input_size)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        base_pattern = torch.sin(grid_x + phase) * torch.cos(grid_y + phase)

        # Add per-sample noise
        for _ in range(samples_per_class):
            sample = base_pattern + torch.randn(1, input_size, input_size) * 0.5
            data_list.append(sample)
            label_list.append(c)

    data = torch.stack(data_list)
    labels = torch.tensor(label_list, dtype=torch.long)

    # Shuffle
    perm = torch.randperm(len(data))
    data = data[perm]
    labels = labels[perm]

    return data, labels


def train_epoch(model, data, labels, batch_size, optimizer, loss_fn, device):
    """Train one epoch, return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    indices = torch.randperm(len(data))

    for i in range(0, len(data), batch_size):
        batch_idx = indices[i:i + batch_size]
        x = data[batch_idx].to(device)
        y = labels[batch_idx].to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(model, data, labels, batch_size, device):
    """Evaluate model accuracy on given data."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            x = data[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)

            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / max(total, 1)
    return acc, correct, total


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    print(f"Kernel size: {args.kernel_size}")
    print(f"Stride: {args.stride}")
    print(f"Epochs: {args.epochs}")

    # Create model
    if args.use_ref or device.type == "cpu":
        model = CPUConvModel(
            in_channels=1, num_classes=args.num_classes,
            kernel_size=args.kernel_size, stride=args.stride
        ).to(device)
        print("Using reference model (nn.Conv2d)")
    else:
        model = ConvModel(
            in_channels=1, num_classes=args.num_classes,
            kernel_size=args.kernel_size, stride=args.stride
        ).to(device)
        print("Using custom CUDA conv model")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Generate data
    data, labels = make_synthetic_data(
        args.input_size, args.num_samples, args.num_classes,
        args.batch_size, device, seed=args.seed
    )
    print(f"Data: {data.shape}, Labels: {labels.shape}")

    # Split into train/eval
    n_train = int(0.8 * len(data))
    train_data, train_labels = data[:n_train], labels[:n_train]
    eval_data, eval_labels = data[n_train:], labels[n_train:]

    if args.eval_only:
        # Only evaluate (useful for checking if a trained model works)
        acc, correct, total = evaluate(model, eval_data, eval_labels, args.batch_size, device)
        print(f"eval_accuracy {correct}/{total} ({acc*100:.1f}%)")
        return

    # Training loop
    t_start = time.time()
    best_eval_acc = 0.0

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_data, train_labels,
            args.batch_size, optimizer, loss_fn, device
        )
        t1 = time.time()

        eval_acc, eval_correct, eval_total = evaluate(
            model, eval_data, eval_labels, args.batch_size, device
        )

        best_eval_acc = max(best_eval_acc, eval_acc)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch={epoch:3d}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc*100:.1f}%  eval_acc={eval_acc*100:.1f}%  "
                  f"time={t1-t0:.2f}s")

    t_total = time.time() - t_start

    # Final results
    print(f"\nfinal_train_accuracy {train_acc*100:.1f}%")
    print(f"final_eval_accuracy {eval_acc*100:.1f}%")
    print(f"best_eval_accuracy {best_eval_acc*100:.1f}%")
    print(f"total_time {t_total:.1f}s")

    if args.profile:
        print(f"avg_epoch_time {t_total/args.epochs:.2f}s")


if __name__ == "__main__":
    main()

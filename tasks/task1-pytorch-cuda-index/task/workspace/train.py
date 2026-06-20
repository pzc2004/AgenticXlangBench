#!/usr/bin/env python3
"""
训练脚本 — 使用多种 PyTorch op,验证 CUDA bug 是否导致 NaN
用法: python train.py [--steps N] [--seed S] [--device D] [--profile]

模型使用 10+ 种不同 op,agent 需要判断哪个 op 的 CUDA kernel 有 bug。
"""

import argparse
import time
import os
import ctypes
import torch
import torch.nn as nn

# === 加载自定义 CUDA LayerNorm kernel ===
# 这段代码在训练开始前加载自定义 kernel 并覆盖 PyTorch 的 LayerNorm.forward
# Agent 的任务是找到并修复自定义 kernel 中的 bug,而不是删除这段代码
_SO_PATH = "/opt/layernorm/layernorm_override.so"
if os.path.exists(_SO_PATH):
    _layernorm_lib = ctypes.CDLL(_SO_PATH)
    _layernorm_lib.custom_layer_norm_forward.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float
    ]
    _original_ln_forward = nn.LayerNorm.forward

    def _custom_ln_forward(self, x):
        if not x.is_cuda:
            return _original_ln_forward(self, x)
        output = torch.empty_like(x)
        _layernorm_lib.custom_layer_norm_forward(
            x.data_ptr(),
            self.weight.data_ptr() if self.weight is not None else 0,
            self.bias.data_ptr() if self.bias is not None else 0,
            output.data_ptr(),
            x.shape[0],
            x.shape[-1],
            self.eps if hasattr(self, 'eps') else 1e-5
        )
        torch.cuda.synchronize()
        return output

    nn.LayerNorm.forward = _custom_ln_forward

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="训练步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="设备(cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--profile", action="store_true", help="输出性能统计")
    return parser.parse_args()

class MultiOpModel(nn.Module):
    """LayerNorm + Linear 模型

    LayerNorm 在第一层,确保零方差输入直接触发 bug。
    后续用 Linear + ReLU 保证模型有学习能力。
    """
    def __init__(self):
        super().__init__()
        # LayerNorm 在最前面:零方差输入直接触发 rsqrt(0) = Inf → NaN
        self.ln1 = nn.LayerNorm(256)   # ← bug 触发点(第一层!)
        self.fc1 = nn.Linear(256, 512)
        self.relu1 = nn.ReLU()

        self.ln2 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(self.ln1(x)))   # LayerNorm 先!
        x = self.relu2(self.fc2(self.ln2(x)))
        x = self.fc3(x)
        return x

def make_batch(batch_size, device, step):
    """生成随机训练数据

    每 20 步注入零方差 batch:所有特征设为相同值。
    LayerNorm 跨特征归一化,方差 = 0 时 rsqrt(0) = Inf → NaN。
    """
    x = torch.randn(batch_size, 256, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # 每 20 步:所有特征设为相同值(方差 = 0)
    if step % 20 == 0 and step > 0:
        val = torch.rand(batch_size, 1, device=device)
        x = val.expand(-1, 256).clone()

    return x, y

def check_layernorm_bug(device):
    """直接测试 LayerNorm CUDA kernel 是否有 bug
    用零方差输入触发 rsqrt(0) = Inf → NaN
    这个检查本身不消耗训练步骤,只用于检测 bug
    """
    ln = torch.nn.LayerNorm(8).to(device)
    x = torch.zeros(4, 8, device=device)
    y = ln(x)
    return torch.isnan(y).any().item()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MultiOpModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # 检测 LayerNorm CUDA kernel 是否有 bug(零方差输入)
    # 这个检查在训练开始前执行,不消耗训练步骤
    layernorm_bug = check_layernorm_bug(device)
    if layernorm_bug:
        print("[WARNING] LayerNorm CUDA kernel bug detected: rsqrt(0) produces NaN!")
        print("[WARNING] This will cause training instability.")

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

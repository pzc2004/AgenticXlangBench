"""
图像分类模型定义(用于 autograd 测试)
"""

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """简单分类器,使用多种激活函数以触发不同的 backward 路径"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),          # ← Bug 1 影响这里
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),       # ← Bug 2 影响这里
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),          # ← Bug 3 影响这里
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

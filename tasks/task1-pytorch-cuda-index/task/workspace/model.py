"""
图像分类模型定义
"""

import torch
import torch.nn as nn


class FeatureNorm(nn.Module):
    """特征归一化层 —— 对最后一个维度做归一化 + 可学习的缩放和偏移。
    内部用 PyTorch 的 F.layer_norm 实现。
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class ImageClassifier(nn.Module):
    """多层图像分类器"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            FeatureNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GroupNorm(8, 256),
            nn.Dropout(0.1),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

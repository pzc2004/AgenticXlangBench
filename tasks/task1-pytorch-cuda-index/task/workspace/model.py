"""
多 op 模型定义 —— 包含 10+ 种 PyTorch op
Agent 需要读这个文件才能知道模型用了哪些 op。
LayerNorm 藏在中间,不在第一层。
"""

import torch.nn as nn


class MultiOpModel(nn.Module):
    """使用 Conv2d + BatchNorm + ReLU + GELU + MaxPool + Linear + LayerNorm + Dropout"""

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

        # Flatten → Linear → LayerNorm
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.ln1 = nn.LayerNorm(512)       # ← bug 触发点(第 8 层)

        # Linear → LayerNorm → Dropout
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)       # ← 第 12 层
        self.dropout = nn.Dropout(0.1)

        # Linear
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.gelu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.ln1(self.fc1(x))
        x = self.dropout(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x

import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=[9, 19, 39], bottleneck=True):
        super().__init__()
        self.ks = ks
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else nn.Identity()
        self.convs = nn.ModuleList([nn.Conv1d(nf if bottleneck else ni, nf, k, padding=k//2, bias=False) for k in ks])
        self.maxpool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(x)
        out = torch.cat([c(x) for c in self.convs] + [self.maxpool(input_tensor)], dim=1)
        return self.act(self.bn(out))

class InceptionTime(nn.Module):
    def __init__(self, n_classes=5, nf=32):
        super().__init__()
        self.block1 = InceptionModule(1, nf)
        self.block2 = InceptionModule(nf * 4, nf)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(nf * 4, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.fc(x)

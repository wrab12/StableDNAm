import torch
import torch.nn as nn


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        b = x.size(0)
        c = x.size(1)
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x.squeeze(dim=2) * y.expand_as(x).squeeze(dim=2)


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sub_networks.our_blocks import ResidualBlock


class ContextNet(nn.Module):
    def __init__(self, kernel_size, nf):
        super(ContextNet, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.RB1 = ResidualBlock(kernel_size, nf)
        self.RB2 = ResidualBlock(kernel_size, nf)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        stage1 = self.RB1(x)
        stage2 = self.RB2(stage1)
        out = torch.cat((x, stage1, stage2), dim=1)

        return out
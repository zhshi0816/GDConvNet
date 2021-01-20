
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_mid = nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1, padding=0)
        self.conv_out = nn.Conv2d(in_channels//reduction_ratio, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.avg_pool(x)
        # out = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)[:, :, None, None]
        out = F.relu(self.conv_in(out))
        out = F.relu(self.conv_mid(out))
        out = torch.sigmoid(self.conv_out(out))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, reduction_ratio=2, merged_channels=1):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, merged_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(merged_channels, reduction_ratio**2*merged_channels, kernel_size, stride=reduction_ratio, padding=(kernel_size - 1) // 2)
        self.conv3 = nn.ConvTranspose2d(reduction_ratio**2*merged_channels, merged_channels, kernel_size, stride=reduction_ratio, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(out1))
        out3 = torch.sigmoid(self.conv3(out2, out1.size()))
        return out3

class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class NoSEBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class AttentionBlock(nn.Module):
    def __init__(self, inchannel, reduction_rate):
        super(AttentionBlock, self).__init__()
        self.CA = ChannelAttention(inchannel, reduction_ratio=reduction_rate)
        self.SA = SpatialAttention(inchannel, reduction_ratio=reduction_rate)

    def forward(self, x):
        out_ca = self.CA(x)
        out_sa = self.SA(x)
        out = (x*out_ca)*out_sa
        return out


class ResidualBlock(nn.Module):
    def __init__(self, kernel_size, channel):
        super(ResidualBlock, self).__init__()
        self.conv_in = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv_out = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1)//2)

    def forward(self, x):
        out = F.relu(self.conv_in(x))
        out = self.conv_out(out)
        return F.relu(out + x)

class BottleNeck(nn.Module):
    def __init__(self, kernel_size, reduction, in_channel, activation=True):
        super(BottleNeck, self).__init__()
        self.conv_in = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1)

        self.conv = nn.Conv2d(in_channel//reduction, in_channel//reduction, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

        self.activation = activation

    def forward(self, x):
        inp = F.relu(self.conv_in(x))
        inp = F.relu(self.conv(inp))
        out = self.conv_out(inp) + x
        if self.activation:
            return F.relu(out)
        else:
            return out

if __name__ == '__main__':
    a = nn.Conv2d(3,3,3,1,1)
    print(a)
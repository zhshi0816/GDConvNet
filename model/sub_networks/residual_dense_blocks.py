import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sub_networks.our_blocks import SEBlock, NoSEBlock, ChannelAttention, SpatialAttention


class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, reduction):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        if reduction != -1:
            self.SE = SEBlock(in_channels, reduction)
        else:
            self.SE = NoSEBlock()

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return self.SE(out)



class ARDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, reduction):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(ARDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        # add attention to RDB
        self.CA = ChannelAttention(in_channels, reduction_ratio=reduction)
        self.SA = SpatialAttention(in_channels, reduction_ratio=reduction)

    def forward(self, x):
        out_rdb = self.residual_dense_layers(x)
        out_rdb = self.conv_1x1(out_rdb)
        out_ca = self.CA(out_rdb)
        out_sa = self.SA(out_rdb)
        out_attention = (out_rdb*out_ca)*out_sa
        out = out_attention + x
        return out


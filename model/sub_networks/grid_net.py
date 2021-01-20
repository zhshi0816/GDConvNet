import torch.nn as nn
import torch.nn.functional as F
from model.sub_networks.our_blocks import SEBlock, NoSEBlock
from model.sub_networks.residual_dense_blocks import RDB


class DownSample(nn.Module):
    def __init__(self, in_channels, reduction, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        if reduction != -1:
            self.SE = SEBlock(stride*in_channels, reduction)
        else:
            self.SE = NoSEBlock()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return self.SE(out)


class UpSample(nn.Module):
    def __init__(self, in_channels, reduction, kernel_size=3, stride=2):
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride=stride, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels // stride, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        if reduction != -1:
            self.SE = SEBlock(in_channels // stride, reduction)
        else:
            self.SE = NoSEBlock()

    def forward(self, x, output_size):
        out = F.relu(self.deconv(x, output_size=output_size))
        out = F.relu(self.conv(out))
        return self.SE(out)


class GridNet(nn.Module):
    def __init__(self, in_channels=3, reduction = (2,2,2), depth_rate=16, kernel_size=3, stride=2, num_dense_layer=4, growth_rate=16):
        super(GridNet, self).__init__()
        self.rdb_module = nn.ModuleList()
        self.upsample_module = nn.ModuleList()
        self.downsample_module = nn.ModuleList()
        self.stride = stride
        self.depth_rate = depth_rate
        self.conv_in = nn.Conv2d(in_channels, depth_rate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv_out = nn.Conv2d(depth_rate, 3, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)

        self.downsample1_1 = DownSample(depth_rate, -1)
        self.downsample1_2 = DownSample(depth_rate, reduction[1])
        self.downsample1_3 = DownSample(depth_rate, reduction[1])

        self.downsample2_1 = DownSample(stride*depth_rate, -1)
        self.downsample2_2 = DownSample(stride*depth_rate, reduction[2])
        self.downsample2_3 = DownSample(stride*depth_rate, reduction[2])

        self.upsample3_4 = UpSample(stride * stride * depth_rate, reduction[1])
        self.upsample3_5 = UpSample(stride * stride * depth_rate, reduction[1])
        self.upsample3_6 = UpSample(stride * stride * depth_rate, reduction[1])

        self.upsample2_4 = UpSample(stride * depth_rate, reduction[0])
        self.upsample2_5 = UpSample(stride * depth_rate, reduction[0])
        self.upsample2_6 = UpSample(stride * depth_rate, reduction[0])

        self.RDB_in = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)
        self.RDB1_1 = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)
        self.RDB1_2 = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)
        self.RDB1_3 = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[0])
        self.RDB1_4 = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[0])
        self.RDB1_5 = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[0])
        self.RDB_out = RDB(depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)

        self.RDB2_1 = RDB(stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[1])
        self.RDB2_2 = RDB(stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[1])
        self.RDB2_3 = RDB(stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[1])
        self.RDB2_4 = RDB(stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[1])
        self.RDB2_5 = RDB(stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[1])

        self.RDB3_1 = RDB(stride * stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[2])
        self.RDB3_2 = RDB(stride * stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = reduction[2])
        self.RDB3_3 = RDB(stride * stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)
        self.RDB3_4 = RDB(stride * stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)
        self.RDB3_5 = RDB(stride * stride * depth_rate, num_dense_layer=num_dense_layer, growth_rate=growth_rate, reduction = -1)

    def forward(self, x):
        inp = self.conv_in(x)

        x_1_1 = self.RDB_in(inp)
        x_1_2 = self.RDB1_1(x_1_1)
        x_1_3 = self.RDB1_2(x_1_2)

        x_2_1 = self.downsample1_1(x_1_1)
        x_2_2 = self.RDB2_1(x_2_1) + self.downsample1_2(x_1_2)
        x_2_3 = self.RDB2_2(x_2_2) + self.downsample1_3(x_1_3)

        x_3_1 = self.downsample2_1(x_2_1)
        x_3_2 = self.RDB3_1(x_3_1) + self.downsample2_2(x_2_2)
        x_3_3 = self.RDB3_2(x_3_2) + self.downsample2_3(x_2_3)

        x_3_4 = self.RDB3_3(x_3_3)
        x_3_5 = self.RDB3_4(x_3_4)
        x_3_6 = self.RDB3_5(x_3_5)

        x_2_4 = self.RDB2_3(x_2_3) + self.upsample3_4(x_3_4, x_2_3.size())
        x_2_5 = self.RDB2_4(x_2_4) + self.upsample3_5(x_3_5, x_2_4.size())
        x_2_6 = self.RDB2_5(x_2_5) + self.upsample3_6(x_3_6, x_2_5.size())

        x_1_4 = self.RDB1_3(x_1_3) + self.upsample2_4(x_2_4, x_1_3.size())
        x_1_5 = self.RDB1_4(x_1_4) + self.upsample2_5(x_2_5, x_1_4.size())
        x_1_6 = self.RDB1_5(x_1_5) + self.upsample2_6(x_2_6, x_1_5.size())

        out = self.RDB_out(x_1_6)
        out = self.conv_out(out)

        return out

if __name__ == '__main__':
    net =  GridNet(9, depth_rate=16, growth_rate=16)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))


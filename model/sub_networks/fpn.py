import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sub_networks.our_blocks import SEBlock, ResidualBlock


class PPM(nn.Module):
    def __init__(self, channel):
        super(PPM, self).__init__()
        self.SE = SEBlock(channel, 16)

        self.down1 = nn.Conv2d(channel, channel//4, kernel_size=1)
        self.down2 = nn.Conv2d(channel, channel//4, kernel_size=1)
        self.down3 = nn.Conv2d(channel, channel//4, kernel_size=1)
        self.down6 = nn.Conv2d(channel, channel//4, kernel_size=1)

        self.conv_out = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        down2 = F.adaptive_avg_pool2d(x, (h//2,w//2))
        down3 = F.adaptive_avg_pool2d(x, (h//3,w//3))
        down6 = F.adaptive_avg_pool2d(x, (h//6,w//6))

        down1 = F.relu(self.down1(x))
        down2 = F.relu(self.down2(down2))
        down3 = F.relu(self.down3(down3))
        down6 = F.relu(self.down6(down6))

        up2 = F.upsample(down2, (h, w), mode='bilinear')
        up3 = F.upsample(down3, (h, w), mode='bilinear')
        up6 = F.upsample(down6, (h, w), mode='bilinear')

        out = torch.cat((down1, up2, up3, up6), dim=1)
        out = self.SE(out)
        out = F.relu(self.conv_out(out) + x)

        return out

class Offset_FPN_Concat(nn.Module):
    def __init__(self, base_channel, growth_rate=2):
        super(Offset_FPN_Concat, self).__init__()
        channel = base_channel
        self.conv_in = nn.Conv2d(12, channel, kernel_size=3, stride=1, padding=1)
        self.stage1 = nn.Sequential(
            ResidualBlock(3, channel),
            ResidualBlock(3, channel),
        )

        self.lateral11 = nn.Conv2d(channel, base_channel, kernel_size=1)


        self.down1 = nn.Conv2d(channel, int(channel*growth_rate), kernel_size=3, stride=2, padding=1)
        channel = int(channel * growth_rate)
        self.stage2 = nn.Sequential(
            ResidualBlock(3, channel),
            ResidualBlock(3, channel),
        )
        self.lateral22 = nn.Conv2d(channel, base_channel, kernel_size=1)


        self.down2 = nn.Conv2d(channel, int(channel*growth_rate), kernel_size=3, stride=2, padding=1)
        channel = int(channel * growth_rate)
        self.stage3 = nn.Sequential(
            ResidualBlock(3, channel),
            ResidualBlock(3, channel),
        )
        self.PPM = PPM(channel)
        self.lateral33 = nn.Conv2d(channel, base_channel, kernel_size=1)

        self.SE1 = SEBlock(base_channel * 3, 16)

        self.smooth11 = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)
        self.smooth22= nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)
        self.smooth33 = nn.Conv2d(base_channel, base_channel, kernel_size=3, stride=1, padding=1)

        self.pre = nn.Conv2d(3*base_channel, 3*base_channel, kernel_size=3, stride=1, padding=1)
        # self.pos = nn.Conv2d(3*base_channel, 3*base_channel, kernel_size=3, stride=1, padding=1)
        # self.smooth_out = nn.Conv2d(3*base_channel, 3*base_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h_1, w_1 = x.size(2), x.size(3)
        inp = self.conv_in(x)
        stage1 = self.stage1(inp)
        stage2 = F.relu(self.down1(stage1))

        stage2 = self.stage2(stage2)
        h_2, w_2 = stage2.size(2), stage2.size(3)
        stage3 = F.relu(self.down2(stage2))

        stage3 = self.stage3(stage3)

        top = self.PPM(stage3)

        top3 = self.lateral33(top)
        top2 = self.lateral22(stage2) + F.upsample(top3, (h_2, w_2))
        top1 = self.lateral11(stage1) + F.upsample(top2, (h_1, w_1))

        out3 = F.upsample(top3, (h_1, w_1), mode='bilinear')
        out2 = F.upsample(top2, (h_1, w_1), mode='bilinear')
        out1 = top1

        out3 = F.relu(self.smooth33(out3))
        out2 = F.relu(self.smooth22(out2))
        out1 = F.relu(self.smooth11(out1))

        ref = torch.cat((out1, out2, out3), dim=1)
        out = self.SE1(ref)

        # out = F.relu(self.smooth_out(out))
        out = F.relu(self.pre(out))
        return out
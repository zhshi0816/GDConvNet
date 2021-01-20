import torch
import torch.nn as nn
from model.sub_networks.context_net import ContextNet
from model.sub_networks.fpn import Offset_FPN_Concat
from model.sub_networks.our_blocks import SEBlock
from model.sub_networks.grid_net import GridNet


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        N, C, H, W = X.size()
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps)
        loss = torch.sum(error)
        loss = loss/(N*C*H*W)
        return loss


class Net(nn.Module):
    def __init__(self, nf, growth_rate, mode):
        super(Net, self).__init__()
        self.offset = Offset_FPN_Concat(nf//3, growth_rate)

        if mode == 'poly':
            from model.deformable.deform_conv3D_poly import DeformConv3d as DCN3D
        elif mode == '3axis':
            from model.deformable.deform_conv3D_3Dinterpolation_inverse_distance import DeformConv3d as DCN3D
        elif mode == '1axis':
            from model.deformable.deform_conv3D_1DInterpolation_inverse_distance import DeformConv3d as DCN3D

        self.dcn_image = DCN3D(3, 3, nf, kernel_size=5, padding=2, stride=1, bias=True, modulation=True)
        self.dcn_context = DCN3D(6, 6, nf, kernel_size=5, padding=2, stride=1, bias=True, modulation=True)

        self.context = nn.Sequential(ContextNet(7, 16),
                                     SEBlock(48, 3),
                                     nn.Conv2d(48, 6, kernel_size=3, stride=1, padding=1)
                                     )

        self.SE = SEBlock(9, 3)
        self.grid = GridNet(9, depth_rate=16, growth_rate=16)

    def forward(self, img1, img2, img4, img5, img_name=None):
        offset_central = self.offset(torch.cat((img1, img2, img4, img5), dim=1))

        mid_out = self.dcn_image(img1, img2, img4, img5, offset_central)

        image1_context = self.context(img1)
        image2_context = self.context(img2)
        image4_context = self.context(img4)
        image5_context = self.context(img5)

        central_context = self.dcn_context(image1_context, image2_context,
                                           image4_context, image5_context, offset_central)

        out = self.SE(torch.cat((mid_out, central_context), dim=1))
        out = self.grid(out)

        if self.training:
            return out + mid_out, mid_out
        else:
            return mid_out+out


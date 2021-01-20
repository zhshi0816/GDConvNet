import torch
from torch import nn
import torch.nn.functional as F


class DeformConv3d(nn.Module):
    def __init__(self, inc, outc, offset_source_channel, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv_1 = nn.Conv2d(offset_source_channel, 2*kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)
        self.p_conv_2 = nn.Conv2d(offset_source_channel, 2*kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)
        self.p_conv_4 = nn.Conv2d(offset_source_channel, 2*kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)
        self.p_conv_5 = nn.Conv2d(offset_source_channel, 2*kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)

        self.xy_conv = nn.Conv2d(offset_source_channel, 2*kernel_size * kernel_size, kernel_size=5, padding=2, stride=stride)
        self.h_conv = nn.Conv2d(offset_source_channel, kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(offset_source_channel, kernel_size*kernel_size, kernel_size=5, padding=2, stride=stride)

        self.init_offset()

    def init_offset(self):
        self.p_conv_1.weight.data.zero_()
        self.p_conv_2.weight.data.zero_()
        self.p_conv_4.weight.data.zero_()
        self.p_conv_5.weight.data.zero_()
        self.p_conv_1.bias.data.zero_()
        self.p_conv_2.bias.data.zero_()
        self.p_conv_4.bias.data.zero_()
        self.p_conv_5.bias.data.zero_()
        self.xy_conv.weight.data.zero_()
        self.xy_conv.bias.data.zero_()
        self.h_conv.weight.data.zero_()
        self.h_conv.bias.data.zero_()
        if self.modulation:
            self.m_conv.weight.data.zero_()
            self.m_conv.bias.data.fill_(1)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, img1, img2, img4, img5, offset_source):
        offset_1 = self.p_conv_1(offset_source)
        offset_2 = self.p_conv_2(offset_source)
        offset_4 = self.p_conv_4(offset_source)
        offset_5 = self.p_conv_5(offset_source)

        N = offset_1.size(1) // 2

        offset_mean_list = [torch.mean(torch.abs(offset_1)), torch.mean(torch.abs(offset_2)), torch.mean(torch.abs(offset_4)), torch.mean(torch.abs(offset_5))]
        if sum(offset_mean_list)/len(offset_mean_list) > 100:
            print('Offset mean is {}, larger than 100.'.format(sum(offset_mean_list)/len(offset_mean_list)))

        if self.modulation:
            m = torch.sigmoid(self.m_conv(offset_source))

        if self.padding:
            img1 = self.zero_padding(img1)
            img2 = self.zero_padding(img2)
            img4 = self.zero_padding(img4)
            img5 = self.zero_padding(img5)

        #b, N, h, w
        h_z = self.h_conv(offset_source)
        h_z = h_z.contiguous().permute(0, 2, 3, 1)
        h_z = h_z + 1.5
        h_z = torch.clamp(h_z, 0, 3)/3

        h_xy = self.xy_conv(offset_source)
        h_xy = self._get_p(h_xy, h_xy.data.type())
        h_xy = h_xy.contiguous().permute(0, 2, 3, 1)
        h_xy = torch.cat([torch.clamp(h_xy[..., :N], 0, img1.size(2) - 1), torch.clamp(h_xy[..., N:], 0, img1.size(3) - 1)], dim=-1)

        img1_feature, img1_pos = self.get_2D_output(img1, offset_1)
        img2_feature, img2_pos = self.get_2D_output(img2, offset_2)
        img4_feature, img4_pos = self.get_2D_output(img4, offset_4)
        img5_feature, img5_pos = self.get_2D_output(img5, offset_5)

        weight1 = self.get_weight(h_xy, h_z, img1_pos, 0, img1_feature.size(1), img1.size(2)-1, img1.size(3)-1)
        weight2 = self.get_weight(h_xy, h_z, img2_pos, 1/3, img1_feature.size(1), img1.size(2)-1, img1.size(3)-1)
        weight4 = self.get_weight(h_xy, h_z, img4_pos, 2/3, img1_feature.size(1), img1.size(2)-1, img1.size(3)-1)
        weight5 = self.get_weight(h_xy, h_z, img5_pos, 1, img1_feature.size(1), img1.size(2)-1, img1.size(3)-1)
        weight_norm = weight1 + weight2 + weight4 + weight5

        central_offset = (img1_feature * weight1 + img2_feature * weight2 + img4_feature * weight4 + img5_feature * weight5)/\
                         (weight_norm)

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1) # B, H, W, 2N
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(central_offset.size(1))], dim=1) # B, C, H, W, N
            central_offset *= m

        x_offset = self._reshape_x_offset(central_offset, self.kernel_size)
        out = self.conv(x_offset)

        return out

    def get_weight(self, h_xy, h_z, xy_pos, z_pos, C, x_norm, y_norm):
        N = h_xy.size(3)//2
        d_x = ((h_xy[...,:N] - xy_pos[..., :N]) / x_norm) ** 2
        d_y = ((h_xy[...,N:] - xy_pos[..., N:]) / y_norm) ** 2
        d_z = (h_z - z_pos) ** 2

        # weight = 1/(d_z + 1e-15)
        weight = 1/(d_x + d_y + d_z + 1e-15)

        weight = weight.unsqueeze(dim=1)
        weight = torch.cat([weight for _ in range(C)], dim=1)  # B, C,
        return weight

    def get_2D_output(self, x, offset):
        dtype = offset.data.type()
        N = offset.size(1) // 2

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rt = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_lb = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 - (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 + (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt


        return x_offset, p.detach()


    def _get_p_n(self, N, device, dataType):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))

        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).to(device, dtype=dataType)

        return p_n

    def _get_p_0(self, h, w, N, device, dataType):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(self.padding, h * self.stride + self.padding, self.stride),
            torch.arange(self.padding, w * self.stride + self.padding, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).to(device, dtype=dataType)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, offset.device, offset.dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, offset.device, offset.dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

if __name__ == '__main__':
    device_ids = [0]
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    net = DeformConv3d(1, 1, 2, 3, 1, 1, modulation=True)
    x1 = torch.ones((1, 1, 4, 4))
    x2 = torch.zeros(1, 1, 4, 4)
    with torch.no_grad():
        out = net(x1, x2, torch.cat((x1, x2), dim=1))
        print(out.shape)
        loss = F.mse_loss(out, x1)
        # loss.backward()


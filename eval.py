from configs.config import device_id, mode, ValData, val_batch_size, model_save_path
import torch
from model.GDConvNet import Net
import torch.nn as nn
from torch.utils.data import DataLoader
from util.utils import to_psnr
import pytorch_ssim
import torch.nn.functional as F


def to_loss(rec, gt):
    mse = F.mse_loss(rec, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    return mse_list

device_ids = device_id
device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")


# Build model
net = Net(144, growth_rate=2, mode=mode)


# multi-GPU
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(model_save_path + 'net_best_weight'), strict=False)

val_data_loader_full = DataLoader(ValData(), batch_size=val_batch_size, shuffle=False, num_workers=12)
val_data_loader_mini = DataLoader(ValData("mini"), batch_size=val_batch_size, shuffle=False, num_workers=12)

val_data_loader = val_data_loader_full

oup_psnr = []
oup_ssim = []

net.eval()

for batch_id, val_data in enumerate(val_data_loader):

    print("{}-{}".format(batch_id, len(val_data_loader)))

    with torch.no_grad():
        img1, img2, img4, img5, gt, img_name = val_data
        img1 = img1.to(device)
        img2 = img2.to(device)
        img4 = img4.to(device)
        img5 = img5.to(device)
        gt = gt.to(device)

        oup = net(img1, img2, img4, img5)

    # Calculate average PSNR
    oup_psnr.extend(to_psnr(oup, gt))
    # Calculate average SSIM
    oup_ssim.extend(pytorch_ssim.ssim(oup + 0.5, gt + 0.5, size_average=False).cpu().numpy())

    torch.cuda.empty_cache()

print('successfully test {} images'.format(len(oup_psnr)))

print('oup_PSNR:{0:.2f}, oup_SSIM:{1:.4f}'.format(sum(oup_psnr)/len(oup_psnr), sum(oup_ssim)/len(oup_ssim)))
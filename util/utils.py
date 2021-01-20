import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure
from configs.config import lr_schular, img_save_path, training_schedule
import os
import statistics
import torchvision.transforms as TF
from configs.config import mean, std
import pytorch_ssim


def findLastCheckpoint(save_dir):
    if os.path.exists(save_dir):
        file_list = os.listdir(save_dir)
        result = 0
        for file in file_list:
            try:
                num = int(file.split("_")[-1])
                result = max(result, num)
            except:
                continue
        return result
    else:
        os.mkdir(save_dir)
        return 0


def to_psnr(rec, gt):
    mse = F.mse_loss(rec, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    # ToTensor scales input images to [0.0, 1.0]
    intensity_max = 1.0
    try:
        psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    except:
        print(mse_list)
    return psnr_list

def to_ssim_skimage(rec, gt, data_range = 1):
    rec_list = torch.split(rec, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    rec_list_np = [rec_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(rec_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(rec_list))]
    ssim_list = [measure.compare_ssim(rec_list_np[ind],  gt_list_np[ind], data_range = data_range, multichannel=True) for ind in range(len(rec_list))]

    return ssim_list

def validation(net, val_data_loader, device, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    oup_psnr = []
    ssim_list = []
    net.eval()

    for batch_id, val_data in enumerate(val_data_loader):

        print("{}-{}".format(batch_id, len(val_data_loader)))

        with torch.no_grad():
            img1, img2,  img4, img5, gt, img_name = val_data
            img1 = img1.to(device)
            img2 = img2.to(device)
            img4 = img4.to(device)
            img5 = img5.to(device)
            gt = gt.to(device)

            oup = net(img1, img2, img4, img5)


        # Calculate average PSNR
        oup_psnr.extend(to_psnr(oup, gt))
        # Calculate average SSIM
        ssim_list.extend(pytorch_ssim.ssim(oup + 0.5, gt + 0.5, size_average=False).cpu().numpy())


        if save_tag:
            save_image(oup.clone()+0.5, img_name)
            # save_image(indice, img_name)

        torch.cuda.empty_cache()

    print('successfully test {} images'.format(len(oup_psnr)))

    return statistics.mean(oup_psnr), statistics.mean(ssim_list)


def save_image(recovery, image_name):
    recovery_image = torch.split(recovery, 1, dim=0)
    batch_num = len(recovery_image)

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    for ind in range(batch_num):
        utils.save_image(recovery_image[ind], img_save_path + '{}.png'.format(image_name[ind]))


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, oup_pnsr, oup_ssim, mode):
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, oup_pnsr, oup_ssim))
    # write training log
    with open('./training_log/{}_log.txt'.format(mode), 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR:{4:.2f}, Val_PSNR:{5:.2f}, Val_SSIM:{6:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, train_psnr, oup_pnsr, oup_ssim), file=f)


def adjust_learning_rate(optimizer, epoch):
    for i in range(len(training_schedule)):
        if epoch < training_schedule[i]:
            current_learning_rate = lr_schular[i]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_learning_rate
        print('Learning rate sets to {}.'.format(param_group['lr']))

def transform(forward=True):
    if forward:
        normalize1 = TF.Normalize(mean, [1.0, 1.0, 1.0])
        normalize2 = TF.Normalize([0, 0, 0], std)
        trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

    else:
        revmean = [-x for x in mean]
        revstd = [1.0 / x for x in std]
        revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
        revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])

        trans = TF.Compose([revnormalize1, revnormalize2])

    return trans

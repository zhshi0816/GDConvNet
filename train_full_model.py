import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datas.train_data import TrainData
from configs.config import learning_rate, num_epochs, train_batch_size, val_batch_size\
    , model_save_path, save_freq, device_id, mode, ValData, delta
from util.utils import to_psnr, print_log, validation, adjust_learning_rate, findLastCheckpoint
from model.GDConvNet import L1_Charbonnier_loss, Net


def main():
    # Choose Gpu device
    device_ids = device_id
    device = torch.device("cuda:{}".format(device_id[0]) if torch.cuda.is_available() else "cpu")

    # Build model
    net = Net(nf=144, growth_rate=2, mode=mode)


    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # multi-GPU
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)

    # calculate all trainable parameters in network
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    train_data_loader = DataLoader(TrainData(), batch_size=train_batch_size, shuffle=True, num_workers=12)
    val_data_loader_full = DataLoader(ValData(), batch_size=val_batch_size, shuffle=False, num_workers=12)
    val_data_loader_mini = DataLoader(ValData("mini"), batch_size=val_batch_size, shuffle=False, num_workers=12)

    print(len(train_data_loader), len(val_data_loader_full))

    # Load Network weight
    try:
        net.load_state_dict(torch.load(model_save_path + 'net_best_weight'), strict=True)
        print("Best weight has been loaded")
    except:
        print('loading best weight failed')
    # old validation PSNR
    # start_time = time.time()
    pre_val_psnr, pre_val_ssim = validation(net, val_data_loader_mini, device, False)
    print('old_val_psnr:{0:.2f}, old_val_ssim:{1:.4f}'.format(pre_val_psnr, pre_val_ssim))
    # end_time = time.time()
    # print(end_time - start_time)
    pre_val_psnr, pre_val_ssim = 0, 0

    # load the latest model
    initial_epoch = findLastCheckpoint(save_dir=model_save_path)
    if initial_epoch > 0:
        net.load_state_dict(torch.load(model_save_path + "net_epoch_{}".format(initial_epoch)), strict=False)
        print("resuming by loading epoch {}".format(initial_epoch))

    cb_loss = L1_Charbonnier_loss()
    cb_loss = cb_loss.to(device)

    # start training
    for epoch in range(initial_epoch, num_epochs):
        start_time = time.time()
        adjust_learning_rate(optimizer, epoch)
        current_psnr_list = []
        psnr_list = []

        # epoch training start
        for batch_id, train_data in enumerate(train_data_loader):
            # initialize network and optimizer parameter gradients
            net.train()
            net.zero_grad()
            optimizer.zero_grad()

            img1, img2, img4, img5, gt = train_data
            img1 = img1.to(device)
            img2 = img2.to(device)
            img4 = img4.to(device)
            img5 = img5.to(device)
            gt = gt.to(device)


            # forward + backward + optimize
            oup, mid_oup = net(img1, img2, img4, img5)

            oup_loss = cb_loss(oup, gt)
            mid_oup_loss = cb_loss(mid_oup, gt)

            # perceptual_loss = loss_network(oup, gt)
            loss = oup_loss + delta * mid_oup_loss

            current_psnr_list.extend(to_psnr(oup, gt))
            psnr_list.extend(current_psnr_list[-train_batch_size:])

            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            # print out
            if not (batch_id % 100):
                if batch_id == 0 and epoch !=0:
                    continue
                print('Epoch:{0}, Iteration:{1}, central_psnr:{2:.2f}, oup_loss:{3:.4f}, mid_oup_loss:{4:.4f}'.format(
                    epoch, batch_id, sum(current_psnr_list)/len(current_psnr_list), oup_loss, mid_oup_loss))
                current_psnr_list = []

        # Average PSNR on one epoch train_data
        train_psnr = sum(psnr_list)/len(psnr_list)
        train_one_epoch_time = time.time() - start_time
        print("Training one epoch costs {}s".format(train_one_epoch_time))

        # use evaluation model during the net evaluating
        if (epoch + 1) % save_freq == 0:
            start_time = time.time()
            oup_pnsr, oup_ssim = validation(net, val_data_loader_full, device)
            val_time = time.time() - start_time
            torch.save(net.state_dict(), model_save_path + "net_epoch_{}".format(epoch + 1))
            print_log(epoch + 1, num_epochs, train_one_epoch_time * save_freq + val_time, train_psnr,
                      oup_pnsr, oup_ssim, 'full')

            if oup_pnsr >= pre_val_psnr:
                torch.save(net.state_dict(), model_save_path + 'net_best_weight')
                pre_val_psnr = oup_pnsr
        else:
            print_log(epoch + 1, num_epochs, train_one_epoch_time, train_psnr, 0, 0, 'full')
            torch.save(net.state_dict(), model_save_path + "net_epoch_{}".format(epoch + 1))



if __name__ == "__main__":
    main()

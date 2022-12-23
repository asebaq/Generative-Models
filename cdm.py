import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from SR3.SR3_pytorch import SR3
from DDPM.DDPM_pytorch import DDPM
import matplotlib
import cv2 as cv
import random

sample = random.randrange(0, 10 ** 6)


def main():
    batch_size = 1
    root = '/home/asebaq/SAC/healthy_aug_22_good'
    testroot = '/home/asebaq/SAC/healthy_aug_22_good'

    transforms_ = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataloader = DataLoader(torchvision.datasets.ImageFolder(root, transform=transforms_),
                            batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(torchvision.datasets.ImageFolder(testroot, transform=transforms_),
                            batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    schedule_opt_sr3 = {'schedule': 'linear', 'n_timestep': 2000,
                        'linear_start': 1e-4, 'linear_end': 0.05}
    schedule_opt_ddpm = {'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 0.05}

    print('[INFO] Load SR3 ...')
    sr3 = SR3(device, img_size=256, LR_size=64, loss_type='l1',
              dataloader=dataloader, testloader=testloader, schedule_opt=schedule_opt_sr3,
              save_path='./SR3.pt', load_path='./SR3/SR3_396.pt', load=True, inner_channel=96,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0.2, res_blocks=2, lr=1e-5, distributed=False)

    transforms_ = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data = torchvision.datasets.ImageFolder(root, transform=transforms_)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print('[INFO] Load DDPM ...')
    ddpm = DDPM(device, dataloader=dataloader, schedule_opt=schedule_opt_ddpm,
                save_path='ddpm.ckpt', load_path='./DDPM/ddpm.ckpt', load=True, img_size=64,
                inner_channel=128,
                norm_groups=32, channel_mults=[1, 2, 2, 2], res_blocks=2, dropout=0.2, lr=5 * 1e-5, distributed=False)

    fixed_noise = torch.randn(batch_size, 3, 64, 64).to(device)
    print('[INFO] Generate image using DDPM ...')
    gen_imgs = ddpm.test(fixed_noise)
    saved_img = np.transpose(torchvision.utils.make_grid(gen_imgs.detach().cpu(), nrow=1, normalize=True),
                             (1, 2, 0))
    saved_img = saved_img.numpy()
    matplotlib.image.imsave(f'generated_image_{sample}.jpg', saved_img)

    # return
    print('[INFO] Super resolve image using SR3 (256) ...')
    imgs_lr = transforms.Resize(256)(transforms.Resize(64)(gen_imgs))
    result_SR = sr3.sr3.super_resolution(imgs_lr)
    saved_img = np.transpose(torchvision.utils.make_grid(result_SR.detach().cpu(), nrow=1, normalize=True),
                             (1, 2, 0))
    saved_img = saved_img.numpy()
    matplotlib.image.imsave(f'super_resolved_image_256_{sample}.jpg', saved_img)

    # print('[INFO] Super resolve image using SR3 (512) ...')
    # imgs_lr = transforms.Resize(512)(transforms.Resize(64)(gen_imgs))
    # result_SR = sr3.sr3.super_resolution(imgs_lr)
    # saved_img = np.transpose(torchvision.utils.make_grid(result_SR.detach().cpu(), nrow=1, normalize=True),
    #                          (1, 2, 0))
    # saved_img = saved_img.numpy()
    # matplotlib.image.imsave('super_resolved_image_512.jpg', saved_img)


if __name__ == "__main__":
    main()

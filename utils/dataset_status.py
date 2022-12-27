import os
from glob import glob
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json


def cal_stats(dataloader, batch_size=4, img_size=64):
    psum = torch.zeros((1, 3))
    psum_sq = torch.zeros((1, 3))

    # loop through images
    for inputs in tqdm(dataloader):
        inputs = inputs[0]
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])

    # pixel count
    count = len(dataloader) * batch_size * img_size * img_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    return {'mean': total_mean[0].tolist(), 'std': total_std[0].tolist()}


def plot_hist(dataloader, img_size=64):
    all_images = torch.zeros((3, img_size, img_size))
    # loop through images
    for inputs in tqdm(dataloader):
        inputs = inputs[0]
        all_images += inputs.sum(axis=[0])
    colors = ('red', 'green', 'blue')
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            all_images[channel_id, :, :], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.title('Histogram')
    plt.xlabel('Color value')
    plt.ylabel('Pixel count')
    plt.savefig(dataset_path.split('/')[-1] + '.png')
    plt.show()


def main(dataset_path):
    img_size = 64
    batch_size = 4
    transforms_ = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    data = torchvision.datasets.ImageFolder(dataset_path, transform=transforms_)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    stats = cal_stats(dataloader)

    print('mean: ' + str(stats['mean']))
    print('std:  ' + str(stats['std']))
    with open('stats.json', 'w') as js:
        json.dump(stats, js, indent=4)

    # plot_hist(dataloader)


if __name__ == '__main__':
    dataset_path = '/home/asebaq/SAC/healthy_aug_22_good'
    # dataset_path = '/home/asebaq/CholecT50_sample/data/images'
    main(dataset_path)

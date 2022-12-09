import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.utils as vutils
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from DCGAN_pytorch import Generator, Discriminator, Train


def main():
    # Prepare dataloader for training
    root = '/home/asebaq/SAC/healthy_aug_22_good'

    batch_size = 64
    img_size = 32
    in_channel = 3
    # transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(img_size), transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataloader = DataLoader(torchvision.datasets.ImageFolder(root, transform=transform),
                            batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    img_shape = (in_channel, img_size, img_size)
    print(f'Input size is {img_shape}')

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:8], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    # Training
    latent_dim = 100
    lr = 0.001
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    # Initialize generator and discriminator
    G = Generator(img_size=img_size, latent_dim=latent_dim, in_channels=in_channel).to(device)
    D = Discriminator(img_size=img_size, in_channels=in_channel).to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    Train(epoch=500, dataloader=dataloader, device=device, G=G, D=D,
          optimizer_G=optimizer_G, optimizer_D=optimizer_D)

    # Save trained model if needed
    torch.save(G, './DC_G.pt')
    torch.save(D, './DC_D.pt')

    # Load trained model if needed
    G = torch.load('./DC_G.pt')
    D = torch.load('./DC_D.pt')

    G.eval()
    z = torch.rand(16, latent_dim).to(device)
    fake = G(z).detach().cpu()
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()

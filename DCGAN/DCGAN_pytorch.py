import torch
import numpy as np
from torch import nn
from tqdm.notebook import tqdm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, in_channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 4
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.in_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, in_channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def Train(epoch, dataloader, device, G, D, optimizer_G, optimizer_D):
    Tensor = torch.FloatTensor
    dim_latent = G.latent_dim
    adversarial_loss = torch.nn.BCELoss()

    # Initialize weights
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    for j in tqdm(range(epoch)):
        for _, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)

            # Adversarial ground truths
            y_valid = torch.ones(batch_size, 1).to(device)
            y_fake = torch.zeros(batch_size, 1).to(device)

            # Configure input
            real_imgs = imgs.type(Tensor).to(device)

            # Sample noise as generator input
            z = torch.rand(batch_size, dim_latent).to(device)

            # Generate a batch of images
            gen_imgs = G(z)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(D(real_imgs), y_valid)
            fake_loss = adversarial_loss(D(gen_imgs.detach()), y_fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(D(gen_imgs), y_valid)
            g_loss.backward()
            optimizer_G.step()

        if (j + 1) % 5 == 0:
            print(f"Epoch {j + 1} / D loss: {d_loss.item():.4f} / G loss: {g_loss.item():.4f}")

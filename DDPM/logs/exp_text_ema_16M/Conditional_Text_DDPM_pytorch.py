import math
import os
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange
from einops.layers.torch import Rearrange

from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
from PIL import Image
from transformers import BertTokenizer, BertModel
# from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from shutil import copyfile
import sys
import random
import time

sys.path.append('.')
from utils.seed_everything import seed_everything
from utils import log
from torchmetrics.image.inception import InceptionScore


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # Input tensor should be shape of [B, 1] with value of time
        count = self.dim // 2
        embed = torch.arange(count, dtype=time.dtype, device=time.device) / count
        embed = time.unsqueeze(1) * torch.exp(-math.log(1e4) * embed.unsqueeze(0))
        embed = torch.cat([embed.sin(), embed.cos()], dim=-1)
        return embed


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Mish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), 'b (qkv heads c) h w -> (qkv) b heads c (h w)', heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, norm_groups=32, num_heads=8, dropout=0.0, att=True):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.att = att
        self.attn = SelfAtt(dim_out, num_heads=num_heads, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        y = self.block1(x)
        y += self.mlp(time_emb).view(x.shape[0], -1, 1, 1)
        y = self.block2(y)
        x = y + self.res_conv(x)
        if self.att:
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, inner_channel=32, norm_groups=32,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, img_size=128, dropout=0.0,
                 text_embed_dim=768, max_text_len=256, text=None):
        super().__init__()

        noise_level_channel = inner_channel
        self.max_text_len = max_text_len
        self.time_embed = nn.Sequential(
            TimeEmbed(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Mish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = img_size

        # Downsampling stage of U-net
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(pre_channel, channel_mult, time_emb_dim=noise_level_channel,
                                      norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, time_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, time_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResBlock(pre_channel + feat_channels.pop(), channel_mult, time_emb_dim=noise_level_channel,
                                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

        if text is not None:
            self.txt_emb = nn.Linear(text_embed_dim, inner_channel)

    def forward(self, x, t, y):
        # Embedding of time step with noise coefficient alpha
        t = self.time_embed(t)

        if y is not None:
            text_tokens = self.txt_emb(y)
            text_tokens = text_tokens[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            text_tokens = text_tokens.mean(dim=1)
            t += torch.unsqueeze(text_tokens, dim=1)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


class Diffusion(nn.Module):
    def __init__(self, model, device, channels=3):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.device = device
        self.loss_func = nn.L1Loss(reduction='sum')

    def make_beta_schedule(self, schedule, n_timestep):
        if schedule == 'linear':
            scale = 1000 / n_timestep
            beta_start = scale * 1e-4
            beta_end = scale * 2e-2
            betas = np.linspace(beta_start, beta_end, n_timestep, dtype=np.float64)
        elif schedule == 'cosine':
            betas = self.cosine_beta_schedule(n_timestep)
        else:
            raise NotImplementedError(schedule)
        return betas

    def cosine_beta_schedule(self, n_timestep):
        betas = []
        max_beta = 0.999
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        for i in range(n_timestep):
            t1 = i / n_timestep
            t2 = (i + 1) / n_timestep
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))

        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log.py calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log.py variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal,
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, txt, clip_denoised: bool):
        batch_size = x.shape[0]

        pred_noise = self.model(x, torch.full((batch_size, 1), t, device=x.device), txt)

        x_recon = self.predict_start(x, t, pred_noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log.py variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, txt, clip_denoised=True):
        mean, log_variance = self.p_mean_variance(x=x, t=t, txt=txt, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def generate(self, x_in, txt):
        img = torch.rand_like(x_in, device=x_in.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, txt)
        return img

    # Compute loss to train the model
    def p_losses(self, x_start, txt):
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t - 1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(x_start.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(x_start.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The model predict actual noise added at time step t
        pred_noise = self.model(x_noisy, torch.full((b, 1), t, device=x_start.device), txt)

        return self.loss_func(noise, pred_noise)

    def forward(self, x, txt, *args, **kwargs):
        return self.p_losses(x, txt, *args, **kwargs)


class SatDataset(torch.utils.data.Dataset):
    def __init__(self, df, root, split='train'):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.root = root
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased',
                                               output_hidden_states=True,
                                               )
        self.transforms_ = transforms.Compose([transforms.Resize(64), transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.df)

    def embed_text(self, text):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings_sum = torch.zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
        for i in range(token_embeddings.shape[0]):
            token_embeddings_sum[i] = torch.sum(token_embeddings[i, -4:], dim=0)
        return token_embeddings_sum

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.root, self.df.filename[idx])
        img = Image.open(img_path)
        img = self.transforms_(img)

        txt = self.df.sent1[idx]
        tokens = self.embed_text(txt)

        data = dict()
        data['image'] = img
        data['txt'] = txt
        data['tokens'] = tokens
        return data


# Class to train & test desired model
class DDPM:
    def __init__(self, device, dataloader, schedule_opt, save_path,
                 load_path=None, load=False, in_channel=3, out_channel=3, inner_channel=32,
                 norm_groups=16, channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0.0,
                 img_size=64, lr=1e-4, distributed=False):
        super(DDPM, self).__init__()
        self.dataloader = dataloader
        self.device = device
        self.save_path = save_path
        self.in_channel = in_channel
        self.img_size = img_size

        model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, img_size,
                     text=True)

        params = sum(p.numel() for p in model.parameters())
        logger.info(f'Number of UNet model parameters : {params:,}')
        self.ddpm = Diffusion(model, device, out_channel)

        # Apply weight initialization & set loss & set noise schedule
        self.ddpm.apply(self.weights_init_orthogonal)
        self.ddpm.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.ddpm = nn.DataParallel(self.ddpm)

        self.optimizer = torch.optim.Adam(self.ddpm.parameters(), lr=lr)
        self.current_epoch = 0
        self.train_loss = 0

        params = sum(p.numel() for p in self.ddpm.parameters())
        # print(f'Number of model parameters : {params}')
        logger.info(f'Number of DDPM model parameters : {params:,}')

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):
        ema = EMA(0.995)
        ema_model = copy.deepcopy(self.ddpm).eval().requires_grad_(False)
        fid = FrechetInceptionDistance(feature=64, normalize=True)
        inception = InceptionScore()

        for i in range(self.current_epoch, epoch):
            logger.info(f'Epoch: {i + 1}/{epoch}')
            self.train_loss = 0
            start = time.time()
            for _, data in enumerate(tqdm(self.dataloader)):
                imgs = data['image'].to(self.device)
                tokens = data['tokens'].to(self.device)
                b, c, h, w = imgs.shape

                self.optimizer.zero_grad()
                loss = self.ddpm(imgs, tokens)
                loss = loss.sum() / int(b * c * h * w)
                loss.backward()
                self.optimizer.step()
                ema.step_ema(ema_model, self.ddpm)
                self.train_loss += loss.item() * b
            self.current_epoch += 1
            current_loss = self.train_loss / len(self.dataloader)
            logger.info(f'Epoch: {i + 1} / loss:{current_loss:.3f}, Time: {round(time.time() - start, 3)} sec')

            if (i + 1) % verbose == 0:
                data = next(iter(self.dataloader))
                real_imgs = data['image'][0]
                tokens = data['tokens'][0].to(self.device)
                txt = data['txt'][0]

                real_imgs = torch.unsqueeze(real_imgs, 0)
                tokens = torch.unsqueeze(tokens, 0)

                txt = txt.replace(' ', '_')[:-1]
                image_save_path = os.path.join('/'.join(self.save_path.split('/')[:-1]), 'generated_images')
                # fid.update(real_imgs, real=True)
                # inception.update(real_imgs)

                # Save example of test images to check training
                fixed_noise = torch.randn(1, self.in_channel, self.img_size, self.img_size).to(self.device)
                start = time.time()
                gen_imgs = self.test(self.ddpm, fixed_noise, tokens)
                logger.info(f'Inference time: {round(time.time() - start, 3)} sec')
                # fid.update(gen_imgs.detach().cpu(), real=False)

                # inception.update(torch.unsqueeze(gen_imgs.detach().cpu(), 0).type(torch.uint8))
                gen_imgs = np.transpose(torchvision.utils.make_grid(
                    gen_imgs.detach().cpu(), nrow=1, padding=2, normalize=True), (1, 2, 0))
                matplotlib.image.imsave(os.path.join(image_save_path,
                                                     f'{txt}_ddpm_{self.current_epoch}.jpg'),
                                        gen_imgs.numpy())
                # logger.info(f'FID: {fid.compute().item()}')
                # logger.info(f'Inception: {inception.compute().item()}')
                # writer.add_scalar('FID/Model', fid.compute().item(), self.current_epoch)
                # writer.add_scalar('Inception/Model', inception.compute().item(), self.current_epoch)

                # Save example of test images to check training
                gen_imgs = self.test(ema_model, fixed_noise, tokens)
                # fid.update(gen_imgs.detach().cpu(), real=False)

                # inception.update(torch.unsqueez(gen_imgs.detach().cpu(), 0))
                gen_imgs = np.transpose(torchvision.utils.make_grid(
                    gen_imgs.detach().cpu(), nrow=1, padding=2, normalize=True), (1, 2, 0))
                matplotlib.image.imsave(os.path.join(image_save_path,
                                                     f'{txt}_ddpm_ema_{self.current_epoch}.jpg'),
                                        gen_imgs.numpy())
                # logger.info(f'EMA FID: {fid.compute().item()}')
                # logger.info(f'EMA Inception: {inception.compute().item()}')
                # writer.add_scalar('FID/EMA Model', fid.compute().item(), self.current_epoch)
                # writer.add_scalar('Inception/EMA Model', inception.compute().item(), self.current_epoch)

                # Save model weight
                self.save(self.save_path, self.ddpm)
                self.save(self.save_path.replace('.ckpt', '_ema.ckpt'), ema_model)
                writer.add_scalar('Epoch', self.current_epoch, self.current_epoch)
                writer.add_scalar('Loss/train', current_loss, self.current_epoch)

    def test(self, model, imgs, txt):
        model.eval()
        with torch.no_grad():
            if isinstance(model, nn.DataParallel):
                gen_imgs = model.module.generate(imgs, txt)
            else:
                gen_imgs = model.generate(imgs, txt)
        model.train()
        return gen_imgs

    def save(self, save_path, network):
        if isinstance(self.ddpm, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.train_loss,
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_rgn_state': random.getstate()
        }, save_path)

    def load(self, load_path):
        # network = self.ddpm
        checkpoint = torch.load(load_path)
        if isinstance(self.ddpm, nn.DataParallel):
            self.ddpm = self.ddpm.module
        self.ddpm.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_loss = checkpoint['loss']
        # Set rng state
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['random_rgn_state'])
        logger.info('Model loaded successfully')


if __name__ == '__main__':
    unet = UNet(channel_mults=[1, 2, 4, 8], res_blocks=2, text=True)  # 2,315,715 parameters
    # val = sum([p.numel() for p in unet.parameters()])
    # print(f'unet {val:,} all parameters')
    # val = sum([p.numel() for p in unet.parameters() if p.requires_grad])
    # print(f'unet {val:,} trainable parameters')

    batch_size = 64
    img_size = 64
    root = '/home/a.sebaq/RSICD_optimal'
    log_dir = '/home/a.sebaq/Generative-Models/DDPM/logs/exp_text_ema_16M'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_images'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, os.path.realpath(__file__).split('/')[-1]))
    logger = log.setup_custom_logger(log_dir, 'root')
    logger.debug('main')
    # Seed
    seed_everything()
    writer = SummaryWriter(os.path.join(log_dir, 'runs'))

    df = pd.read_csv(os.path.join(root, 'dataset_rsicd.csv'))
    data = SatDataset(df, os.path.join(root, 'RSICD_images'))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    schedule_opt = {'schedule': 'cosine', 'n_timestep': 1000}

    ddpm = DDPM(device, dataloader=dataloader, schedule_opt=schedule_opt,
                save_path=os.path.join(log_dir, 'c_txt_ddpm.ckpt'), load_path=os.path.join(log_dir, 'c_txt_ddpm.ckpt'),
                load=True, img_size=img_size,
                inner_channel=128,
                norm_groups=32, channel_mults=[1, 2, 4, 8], res_blocks=2, dropout=0.2, lr=5 * 1e-5,
                distributed=False)

    # val = sum([p.numel() for p in ddpm.ddpm.parameters()])
    # print(f'ddpm {val:,} all parameters')
    # val = sum([p.numel() for p in ddpm.ddpm.parameters() if p.requires_grad])
    # print(f'ddpm {val:,} trainable parameters')
    ddpm.train(epoch=1000, verbose=5)

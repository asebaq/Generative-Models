import torch, torchvision
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from einops import rearrange, repeat
from tqdm.notebook import tqdm
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math, os, copy


"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels*(1+self.use_affine_level)))

    def forward(self, x, noise_embed):
        batch_size = x.shape[0]
        noise = self.noise_func(noise_embed).view(batch_size, -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
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
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_head, norm_groups=32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        # 1x1 convolution is used & Input tensor shape is [B, C, H, W]
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.projection = nn.Conv2d(channel_dim, channel_dim, 1)

    def forward(self, x):
        height = x.shape[2]
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (c n qkv) h w -> (qkv) b n c h w", n=self.num_heads, qkv=3)
        # Each tensor(query, key, value) shape is [batch, num_heads, splitted_channel_dim, height, width]
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        att = torch.einsum('bnchw, bncyx -> bnhwyx', queries, keys).contiguous()
        att = rearrange(att, 'b n h w y x -> b n h w (y x)')
        scaling = self.channel_dim ** (1/2)
        att = F.softmax(att, dim=-1) / scaling
        att = rearrange(att, 'b n h w (y x) -> b n h w y x', y=height)

        out = torch.einsum('bnhwyx, bncyx -> bnchw', att, values).contiguous()
        # Transform output tensor to shape [B, C, H, W] again
        out = rearrange(out, "b n c h w -> b (n c) h w")
        out = self.projection(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        return y + self.res_conv(x)

class ResBlockWithAtt(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.res_block = ResBlock(dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, inner_channel=32, norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(), 
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
                downs.append(ResBlockWithAtt(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlockWithAtt(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                            norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                        norm_groups=norm_groups, dropout=dropout)
        ])

        # Upsampling stage of U-net
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResBlockWithAtt(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, noise_level):
        t = self.noise_level_mlp(noise_level)  # Embedding of time step with noise step
        
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlockWithAtt):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResBlockWithAtt):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""

class Diffusion(nn.Module):
    def __init__(self, model, img_size, device, channels=3):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.model = model.to(device)
        self.device = device

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(self.device)
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, warmup_frac=0.1, 
                            linear_start=1e-4, linear_end=2e-2):
        if schedule == 'quad':
            betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
        elif schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == 'const':
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            warmup_frac=schedule_opt['warmup_frac'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
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
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start_from_noise(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # Note that posterior q for reverse diffusion process is conditioned Gaussian distribution q(x_{t-1}|x_t, x_0)
    # Thus to compute desired posterior q, we need original image x_0 in ideal, 
    # but it's impossible for actual training procedure -> Thus we reconstruct desired x_0 and use this for posterior
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(self.device)
        x_recon = self.predict_start_from_noise(x, t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        mean, log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, x_in):
        img = torch.rand_like(x_in, device=self.device)
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in)
        return img

    # Compute loss to train the model
    def p_losses(self, x_in):
        x_start = x_in['HR']
        b, c, h, w = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        sqrt_alpha = torch.FloatTensor(
            np.random.uniform(self.sqrt_alphas_cumprod_prev[t-1], self.sqrt_alphas_cumprod_prev[t], size=b)
        ).to(self.device)
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1, 1)

        noise = torch.randn_like(x_start).to(self.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * x_start + (1 - sqrt_alpha**2).sqrt() * noise
        # The model predict actual noise added at time step t
        x_recon = self.model(torch.cat([x_in['LR'], x_noisy], dim=1), noise_level=sqrt_alpha)
        
        return self.loss_func(noise, x_recon)

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


# Class to train & test desired model
class SR3():
    def __init__(self, device, loss_type, dataloader, testloader, schedule_opt, save_path, 
                    load_path=None, load=False, in_channel=6, out_channel=3, inner_channel=32, 
                    norm_groups=8, channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, 
                    img_size=64, lr=1e-4):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path

        model = UNet(in_channel, out_channel, inner_channel, norm_groups, channel_mults, res_blocks, dropout, img_size)
        self.sr3 = Diffusion(model, img_size, device, out_channel)
        # Apply weight initialization
        self.sr3.apply(self.weights_init_orthogonal)
        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

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
        fixed_imgs = copy.deepcopy(next(iter(self.testloader)))
        fixed_imgs['LR'] = fixed_imgs['LR'][:4]

        for i in tqdm(range(epoch)):
            for _, imgs in enumerate(self.dataloader):
                imgs['HR'] = imgs['HR'].to(self.device)
                imgs['LR'] = imgs['LR'].to(self.device)
                b, c, h, w = imgs['HR'].shape
    
                self.optimizer.zero_grad()
                loss = self.sr3(imgs)
                loss = loss.sum() / int(b*c*h*w)
                loss.backward()
                self.optimizer.step()

            if (i+1) % verbose == 0:
                test_imgs = next(iter(self.testloader))
                test_imgs['HR'] = test_imgs['HR'].to(self.device)
                test_imgs['LR'] = test_imgs['LR'].to(self.device)
                
                self.sr3.eval()
                with torch.no_grad():
                    val_loss = self.sr3(test_imgs)
                    val_loss = val_loss.sum() / int(b*c*h*w)
                self.sr3.train()
                print(f'Epoch: {i+1} / loss:{loss.item():.3f} / val_loss:{val_loss.item():.3f}')

                # Save example of test images to check training
                result_SR = self.test(fixed_imgs)

                plt.figure(figsize=(15,10))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Low-Resolution Inputs")
                plt.imshow(np.transpose(torchvision.utils.make_grid(fixed_imgs['LR'], padding=1, normalize=True).cpu(),(1,2,0)))

                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Super-Resolution Results")
                plt.imshow(np.transpose(torchvision.utils.make_grid(result_SR.detach().cpu(), padding=1, normalize=True),(1,2,0)))
                plt.savefig('SuperResolution_Result.jpg')
                plt.close()

                # Save model weight
                self.save(self.save_path)


    def test(self, imgs):
        imgs['LR'] = imgs['LR'].to(self.device)
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(imgs['LR'])
            else:
                result_SR = self.sr3.super_resolution(imgs['LR'])
        self.sr3.train()
        return result_SR

    
    def save(self, save_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")


##################################
###  Utils for data loading
##################################

class Dataset_SR3(Dataset):
    def __init__(self, root, transforms_HR=None, transforms_LR=None):
        self.transforms_HR = transforms.Compose(transforms_HR)
        self.transforms_LR = transforms.Compose(transforms_LR)

        self.total_files = []
        for path, _, files in os.walk(root):
            for file in files:
                self.total_files.append(os.path.join(path,file))

    def __getitem__(self, index):
        img_HR = self.transforms_HR(Image.open(self.total_files[index]))
        img_LR = self.transforms_LR(Image.open(self.total_files[index]))
        return {'HR': img_HR, 'LR': img_LR}

    def __len__(self):
        return len(self.total_files)


####################################
###  Execute Training at terminal
####################################

if __name__ == "__main__":
    batch_size = 16
    low_res_size = 32
    img_size = 128
    root = './data/ffhq_thumb'
    testroot = './data/celeba_hq_256'

    transforms_HR = [transforms.Resize(img_size), transforms.ToTensor(), 
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    transforms_LR = [transforms.Resize(low_res_size), transforms.Resize(img_size), transforms.ToTensor(), 
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    dataloader = DataLoader(Dataset_SR3(root, transforms_HR=transforms_HR, transforms_LR=transforms_LR), 
                            batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(Dataset_SR3(testroot, transforms_HR=transforms_HR, transforms_LR=transforms_LR), 
                            batch_size=batch_size, shuffle=True, num_workers=2)

    # Save train data example
    imgs = next(iter(dataloader))
    plt.figure(figsize=(15,10))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Low-Resolution Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(imgs['LR'][:4], padding=1, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("High-Resolution Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(imgs['HR'][:4], padding=1, normalize=True).cpu(),(1,2,0)))
    plt.savefig('Train_Examples.jpg')
    plt.close()
    print("Example train images were saved")

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    schedule_opt = {'schedule':'linear', 'n_timestep':2000, 'warmup_frac':0.1, 'linear_start':1e-6, 'linear_end':1e-2}

    sr3 = SR3(device, loss_type='l1', dataloader=dataloader, testloader=testloader, schedule_opt=schedule_opt, 
                save_path='./SR3.pt', load_path='./SR3.pt', load=False, img_size=img_size, inner_channel=32, 
                norm_groups=16, channel_mults=(1, 2, 4, 8, 8), dropout=0.2, res_blocks=2)
    sr3.train(epoch=250, verbose=10)

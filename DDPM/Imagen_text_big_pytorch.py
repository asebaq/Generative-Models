import os.path
import time

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from PIL import Image
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from shutil import copyfile

import sys
sys.path.append('.')

from utils.seed_everything import seed_everything
from utils import log
from torch.utils.tensorboard import SummaryWriter


class SatDataset(Dataset):
    def __init__(self, df, root, image_size=64, transform=None, split='train'):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.root = root
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.df.filename[idx])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        txt = self.df.sent1[idx]
        return img, txt


def config(log_dir, data_root):
    os.makedirs(log_dir, exist_ok=True)
    logger = log.setup_custom_logger(log_dir, 'root')
    logger.debug('main')
    writer = SummaryWriter(os.path.join(log_dir, 'runs'))
    df = pd.read_csv(os.path.join(data_root, 'dataset_rsicd.csv'))
    os.makedirs(os.path.join(log_dir, 'generated_images'), exist_ok=True)
    copyfile(os.path.realpath(__file__), os.path.join(log_dir, os.path.realpath(__file__).split('/')[-1]))
    return logger, writer, df


def build_models():
    # unets for unconditional imagen
    unet_gen = Unet(
        dim=128,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True)
    )

    
    # imagen, which contains the unet above
    imagen = Imagen(
        text_encoder_name='t5-base',
        unets=unet_gen,
        image_sizes=256,
        timesteps=1000,
        cond_drop_prob=0.1
    )

    return imagen, unet_gen


def build_dataloaders(df, data_root, image_size=64):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])

    train_dataset = SatDataset(df, os.path.join(data_root, 'RSICD_images'), image_size=image_size, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    test_dataset = SatDataset(df, os.path.join(data_root, 'RSICD_images'), image_size=image_size, transform=transform,
                              split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    return train_dataloader, test_dataloader


def train(imagen, df, data_root, logger, writer, log_dir):
    

    # working training loop
    epochs = 1000
    model_path = os.path.join(log_dir, 'checkpoint.pt')

    train_dataloader, test_dataloader = build_dataloaders(df, data_root, 128)
    trainer = ImagenTrainer(imagen=imagen)
    
    
    # Load model
    if os.path.isfile(model_path):
        trainer.load(model_path)
        
    for j in range(277, epochs):
        loss = 0
        start = time.time()
        for _, (imgs, txts) in enumerate(tqdm(train_dataloader)):
            loss += trainer(
                imgs,
                texts=txts,
                unet_number=1,
                max_batch_size=4
            )
            trainer.update(unet_number=1)
        loss = loss / (len(train_dataloader) / 64)
        writer.add_scalar(f'Imagen Model', round(loss, 3), j)

        logger.info(
            f'Finished epoch {j} with loss: {round(loss, 3)} in {round(time.time() - start, 3)} sec')

        if not (j % 5):
            data = next(iter(test_dataloader))
            txt = data[1][0]
            start = time.time()
            images = trainer.sample(texts=[txt], batch_size=1, return_pil_images=True)
            logger.info(f'Sampling time: {round(time.time() - start, 3)} sec')
            image_path = os.path.join(log_dir, 'generated_images',
                                        f"sample-{j}-text-{'_'.join(txt.replace('.', '').split())}.png")
            images[0].save(image_path)
        trainer.save(model_path)


def main():
    data_root = '/home/a.sebaq/RSICD_optimal'
    # log_dir = '/home/a.sebaq/Generative-Models/DDPM/logs/exp_imagen_text_t5_base_bs64_ts50'
    log_dir = '/home/a.sebaq/Generative-Models/DDPM/logs/exp_imagen_text_t5_base_bs64_big'

    logger, writer, df = config(log_dir, data_root)
    seed_everything()

    imagen, unet_gen = build_models()
    params = sum(p.numel() for p in unet_gen.parameters())
    logger.info(f'Number of generation UNet model parameters : {params:,}')
    params = sum(p.numel() for p in imagen.parameters())
    logger.info(f'Number of Imagen model parameters : {params:,}')

    train(imagen, df, data_root, logger, writer, log_dir)


if __name__ == '__main__':
    main()

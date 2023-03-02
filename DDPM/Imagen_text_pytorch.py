import os.path
import time

import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5Model

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from utils.seed_everything import seed_everything
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


# class SatDataset(torch.utils.data.Dataset):
class SatDataset(Dataset):
    def __init__(self, df, root, image_size=128, transform=None, split='train'):
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything()
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    inception = InceptionScore()
    image_size = 224
    # unets for unconditional imagen
    unet = Unet(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=1,
        layer_attns=(False, False, False, True),
        layer_cross_attns=False
    )

    params = sum(p.numel() for p in unet.parameters())
    print(f'Number of UNet model parameters : {params:,}')

    # imagen, which contains the unet above
    imagen = Imagen(
        text_encoder_name='t5-small',
        unets=unet,
        image_sizes=image_size,
        timesteps=1000,
        cond_drop_prob=0.1
    )

    params = sum(p.numel() for p in imagen.parameters())
    print(f'Number of Imagen model parameters : {params:,}')

    trainer = ImagenTrainer(imagen=imagen).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images,
    # text embeddings, then text masks.
    # in this case, only images is returned as it is unconditional training
    root = '/home/asebaq/RSICD_optimal'
    log_dir = './logs/exp_imagen_text_bs16_sr'
    df = pd.read_csv(os.path.join(root, 'dataset_rsicd.csv'))

    transform = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])

    train_dataset = SatDataset(df, os.path.join(root, 'RSICD_images'), image_size=128, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])
    test_dataset = SatDataset(df, os.path.join(root, 'RSICD_images'), image_size=128, transform=transform, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    # Load model
    model_path = './checkpoint_text.pt'
    # if os.path.isfile(model_path):
    #     trainer.load(model_path)
    epochs = 1000
    # working training loop
    for i in (1, 2):
        for j in range(epochs):
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

            print(f'Finished epoch {i} with loss: {loss / (len(train_dataloader) / 16)} in {round(time.time()-start, 3)} sec')

            if not (i % 1):
                data = next(iter(test_dataloader))
                # real_img = data[0]
                # fid.update(real_img, real=True)
                txt = data[1][0]
                start = time.time()
                images = trainer.sample(texts=[txt], batch_size=1, return_pil_images=True)  # returns List[Image]
                print(f'Sampling time: {round(time.time() - start, 3)} sec')
                # fake_img = T.ToTensor()(images[0])
                # fake_img = torch.unsqueeze(fake_img, dim=0)
                # fid.update(fake_img, real=False)
                # fid_score = fid.compute().item()
                # print('FID:', fid_score)
                images[0].save(f"./sample-{i}-text-{'_'.join(txt.replace('.', '').split())}.png")
            trainer.save('./checkpoint_text.pt')


if __name__ == '__main__':
    main()

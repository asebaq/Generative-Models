import torch
from torch.utils.data import DataLoader
import torchvision

import numpy as np
from DDPM.Conditional_Text_DDPM_pytorch import DDPM, SatDataset
import matplotlib
import pandas as pd
import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

sys.path.append('.')
from utils.seed_everything import seed_everything


def show_image(img, title='img', ctype='gray'):
    plt.figure(figsize=(10, 10))
    if ctype == 'bgr':
        b, g, r = cv.split(img)
        rgb_img = cv.merge([r, g, b])
        plt.imshow(rgb_img)
    elif ctype == 'hsv':
        rgb = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        plt.imshow(rgb)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    elif ctype == 'rgb':
        plt.imshow(img)
    else:
        raise Exception('Unknown colour type')
    plt.title(title)
    plt.show()


def embed_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,
                                      )

    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings_sum = torch.zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
    for i in range(token_embeddings.shape[0]):
        token_embeddings_sum[i] = torch.sum(token_embeddings[i, -4:], dim=0)
    return token_embeddings_sum


def main():
    root = '/home/asebaq/RSICD_optimal'
    batch_size = 1
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    schedule_opt_ddpm = {'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 0.05}

    # Seed
    seed_everything()

    df = pd.read_csv(os.path.join(root, 'dataset_rsicd.csv'))
    data = SatDataset(df, os.path.join(root, 'RSICD_images'))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print('[INFO] Load DDPM ...')
    ddpm = DDPM(device, dataloader=dataloader, schedule_opt=schedule_opt_ddpm,
                save_path='ddpm.ckpt',
                load_path='/home/asebaq/Generative-Models/DDPM/logs/exp_text_ema/c_txt_ddpm.ckpt', load=True,
                img_size=64,
                inner_channel=128,
                norm_groups=32, channel_mults=[1, 2, 2, 2], res_blocks=2, dropout=0.2, lr=5 * 1e-5, distributed=False)

    data = next(iter(dataloader))
    real_imgs = data['image']
    txt = data['txt']
    tokens = data['tokens'].to(device)

    # txt = 'some boats are in a parking lot near two parallel roads .'
    # tokens = embed_text(txt).to(device)
    # tokens = tokens.unsqueeze(0)

    fixed_noise = torch.randn(batch_size, 3, 64, 64).to(device)
    show_image(np.transpose(fixed_noise.cpu().numpy()[0], (1, 2, 0)), 'Noise')

    print('[INFO] Generate image using DDPM ...')
    gen_imgs = ddpm.test(ddpm.ddpm, fixed_noise, tokens)
    saved_img = np.transpose(torchvision.utils.make_grid(gen_imgs.detach().cpu(), nrow=1, normalize=True),
                             (1, 2, 0))
    saved_img = saved_img.numpy()
    show_image(saved_img, 'txt[0]')

    matplotlib.image.imsave(txt[0] + '.jpg', saved_img)


if __name__ == "__main__":
    main()

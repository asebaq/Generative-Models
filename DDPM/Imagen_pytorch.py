import os.path

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset


def main():
    # unets for unconditional imagen
    unet = Unet(
        dim=32,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=1,
        layer_attns=(False, False, False, True),
        layer_cross_attns=False
    )

    params = sum(p.numel() for p in unet.parameters())
    print(f'Number of UNet model parameters : {params:,}')

    # imagen, which contains the unet above
    imagen = Imagen(
        condition_on_text=False,  # this must be set to False for unconditional Imagen
        unets=unet,
        image_sizes=128,
        timesteps=1000
    )

    params = sum(p.numel() for p in imagen.parameters())
    print(f'Number of Imagen model parameters : {params:,}')

    trainer = ImagenTrainer(
        imagen=imagen,
        split_valid_from_train=True  # whether to split the validation dataset from the training
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images,
    # text embeddings, then text masks.
    # in this case, only images is returned as it is unconditional training
    dataset = Dataset('/home/asebaq/RSICD_optimal/RSICD_images', image_size=128)

    trainer.add_train_dataset(dataset, batch_size=16)

    # Load model
    model_path = './checkpoint.pt'
    if os.path.isfile(model_path):
        trainer.load(model_path)
    sampling = 3000
    # working training loop
    for i in range(600000):
        trainer.train_step(unet_number=1, max_batch_size=4)
        # loss = trainer.train_step(unet_number=1, max_batch_size=4)
        # print(f'loss: {loss}')

        if not (i % sampling):
            valid_loss = trainer.valid_step(unet_number=1, max_batch_size=4)
            print(f'valid loss: {valid_loss} for iteration: {i}')

        # if not (i % 100) and trainer.is_main:  # is_main makes sure this can run in distributed
        if not (i % sampling):
            images = trainer.sample(batch_size=1, return_pil_images=True)  # returns List[Image]
            images[0].save(f'./sample-{i // sampling}.png')
        trainer.save('./checkpoint.pt')


if __name__ == '__main__':
    main()

Implemented SR3(Image Super-Resolution via Iterative Refinement) model which applied diffusion model framework to Super Resolution task

Used 2D Linear Attention instead of vanilla self-attention for reasonable memory usage

* Trained with FFHQ thumbnails(128x128) dataset : https://github.com/NVlabs/ffhq-dataset

* Tested with CelebA-HQ 256x256 resized dataset : https://www.kaggle.com/badasstechie/celebahq-resized-256x256

#### Examples of images pairs of training dataset
<img src="https://user-images.githubusercontent.com/48702949/136547999-45a613aa-67eb-42d8-8cbf-16b931164659.jpg" width="866" height="123"/>

#### Result from Trained model
Trained to implement Super Resolution task of 32x32 to 128x128 resolution
* Total training epoch = 250
* Random images were selected from the above CelebA-HQ and resized to 32x32 low resolution inputs to test the trained model
<img src="https://user-images.githubusercontent.com/48702949/136547491-cb8dc04c-c52e-446d-84ee-c315558581a4.jpg" width="866" height="123"/>

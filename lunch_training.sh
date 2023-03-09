#!/bin/bash

#SBATCH --job-name=sentinel_fb_gen # job name
#SBATCH --output=/home/a.sebaq/Generative-Models/DDPM/output/%j_%x.out # output log file
#SBATCH --error=/home/a.sebaq/Generative-Models/DDPM/output/%j_%x.err  # error file
#SBATCH --time=23:55:00  # 24 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu  # gpu partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# python /home/a.sebaq/Generative-Models/DDPM/Conditional_Text_DDPM_pytorch.py 
# python /home/a.sebaq/Generative-Models/DDPM/Imagen_text_sr_pytorch.py
python /home/a.sebaq/Generative-Models/DDPM/Imagen_text_pytorch.py
# accelerate launch /home/a.sebaq/Generative-Models/DDPM/Imagen_text_pytorch.py


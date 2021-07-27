# ZSL_Generative
An open source Zero Shot Classification toolbox based on PyTorch.

# Installation
The model is built in PyTorch 1.6.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA11.0, cuDNN7.5).

For installing, follow these intructions
```
conda create -n pytorch160 python=3.7
conda activate pytorch160
conda install pytorch=1.6 torchvision cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```

# Data preparation
Download CUB, AWA2, FLO and SUN features using `downlaod.sh` inside datasets folder.
```
cd datasets; sh download.sh; cd ../
```

# Training and Evaluation
## Zero-Shot Image Classification
To train and evaluate ZSL and GZSL models on CUB, AWA2, FLO and SUN, please run:
```
CUB: python train_images.py -opt options/Tfvaegan/CUB.yml
AWA2: python train_images.py -opt options/Tfvaegan/AWA2.yml
FLO: python train_images.py -opt options/Tfvaegan/FLO.yml
SUN: python train_images.py -opt options/Tfvaegan/SUN.yml

```
# Finetuning Inductive
Download finetuned weights for the CUB, AWA2, FLO and SUN features from the drive link shared below.

```
link: https://drive.google.com/drive/folders/13-eyljOmGwVRUzfMZIf_19HmCj1yShf1?usp=sharing
```

# Training and Evaluation
To train and evaluate ZSL and GZSL models for the finetune inductive setting on CUB, AWA2, FLO and SUN, please run:
```
CUB: python train_images.py -opt options/Tfvaegan/CUB_ft.yml
AWA2: python train_images.py -opt options/Tfvaegan/AWA2_ft.yml
FLO: python train_images.py -opt options/Tfvaegan/FLO_ft.yml
SUN: python train_images.py -opt options/Tfvaegan/SUN_ft.yml
```

# Wandb Logger

[wandb](https://www.wandb.com/) can be viewed as a cloud version of tensorboard. One can easily view training processes and curves in wandb.  
To enable wandb logging edit the configuration file.

```yml
wandb: True
```

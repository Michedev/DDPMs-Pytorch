# DDPM Pytorch

Pytorch implementation of "_Improved Denoising Diffusion Probabilistic Models_", 
"_Denoising Diffusion Probabilistic Models_" and "_Classifier-free Diffusion Guidance_"

![](https://hojonathanho.github.io/diffusion/assets/img/pgm_diagram_xarrow.png)

# Project structure

      .
      ├── ddpm_pytorch  # Source files
      │   ├── config    # YAML config files with all the hyperparameters
      │   ├── distributions.py
      │   ├── ema.py  # Pytorch-Lightning implementation of exponential moving average
      │   ├── generate.py  # Entry point to generate a new batch of images given the checkpoint path
      │   ├── __init__.py
      │   ├── model  # UNet and DDPM train/generation processes are here
      │   ├── paths.py  # Path constants
      │   ├── train.py # Entry point to train a new DDPM model
      │   └── variance_scheduler  # DDPM variance scheduler like linear, cosine
      ├── anaconda-project.yml  # anaconda project file
      ├── anaconda-project-lock.yml  # anaconda project lock file
      └── readme.md   # This file



# How to train

1. Install [anaconda](https://www.anaconda.com/) 

2. Install all the dependencies with the command

       anaconda-project prepare

3. Train the model

       anaconda-project run train-gpu 

   By default, the model trained is the DDPM from "Improved Denoising Diffusion Probabilistic Models" paper on MNIST dataset.
   You can switch to the original DDPM by disabling the vlb with the following command:
      
       anaconda-project run train model.vlb=False
   You can also train the DDPM with the Classifier-free Diffusion Guidance by changing the model:

       anaconda-project run train model=unet_class_conditioned

# How to generate

1. Train a model (See previous section)

2. Generate a new batch of images

       anaconda-project run generate -r RUN

   The other options are: `[--seed SEED] [--device DEVICE] [--batch-size BATCH_SIZE] [-w W] [--scheduler {linear,cosine,tan}] [-T T]`

# Configure the training

Under _ddpm_pytorch/config_ there are several yaml files containing the training parameters 
such as model class and paramters, noise steps, scheduler and so on. 
Note that the hyperparameters in such files are taken from 
the papers "_Improved Denoising Diffusion Probabilistic Models_" 
and "_Denoising Diffusion Probabilistic Models_"

    defaults:
      - model: unet_paper  # take the model config from model/unet_paper.yaml
      - scheduler: cosine  # use the cosine scheduler from scheduler/cosine.yaml
      - dataset: mnist
      - optional model_dataset: ${model}-${dataset}  # set particular hyper parameters for specific couples (model, dataset)
      - optional model_scheduler: ${model}-${scheduler} # set particular hyper parameters for specific couples (model, scheduler)

    batch_size: 128 # train batch size
    noise_steps: 4_000  # noising steps; the T in "Improved Denoising Diffusion Probabilistic Models" and "Denoising Diffusion Probabilistic Models"
    accelerator: null  # training hardware; for more details see pytorch lightning
    devices: null  # training devices to use; for more details see pytorch lightning
    gradient_clip_val: 0.0  # 0.0 means gradient clip disabled
    gradient_clip_algorithm: norm  # gradient clip has two values: 'norm' or 'value
    ema: true  # use Exponential Moving Average implemented in ddpm_pytorch/ema.py
    ema_decay: 0.99  # decay factor of EMA

    hydra:
      run:
        dir: saved_models/${now:%Y_%m_%d_%H_%M_%S}

### Add custom dataset

To add a custom dataset, you need to create a new class that inherits from torch.utils.data.Dataset
and implement the __len__ and __getitem__ methods. 
Then, you need to add the config file to the _ddpm_pytorch/config/dataset_ folder with a similar
structure of mnist.yaml

      width: 28  # meta info about the dataset
      height: 28
      channels: 1   # number of image channels
      num_classes: 10  # number of classes
      files_location: ~/.cache/torchvision_dataset  # location where to store the dataset, in case to be downloaded
      train:  #dataset.train is instantiated with this config
        _target_: torchvision.datasets.MNIST  # Dataset class. Following arguments are passed to the dataset class constructor
        root: ${dataset.files_location}
        train: true
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor
      val:  #dataset.val is instantiated with this config
        _target_: torchvision.datasets.MNIST # Same dataset of train, but the validation split
        root: ${dataset.files_location}
        train: false
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor

### Examples of custom training

__Disable the variational lower bound__, hence training like in "_Denoising Diffusion Probabilistic Models_" with __linear__ scheduler and in __GPU__

    anaconda-project run train scheduler=linear accelerator='gpu' model.vlb=False noise_steps=1000


### Classifier-free Guidance

Use the labels for __Diffusion Guidance__, as in "_Classifier-free Diffusion Guidance_" with the following command

    anaconda-project run train model=unet_class_conditioned noise_steps=1000

# Anaconda-project
## mac-os lock file

1. Remove cudatoolkit from _anaconda-project.yml_ file at the bottom of the file, 
under `env_specs -> default -> packages`
2. Decomment `- osx-64`  under `env_specs -> default -> platforms`
3. Delete _anaconda-project-lock.yml_ file
4. Run `anaconda-project prepare` to generate the new lock file

## CPU-only environment

To have an alternative a PyTorch CPU-only environment, de-comment the following lines at the bottom of _anaconda-project.yml_

    #  pytorch-cpu:
    #    packages:
    #    - cpuonly
    #    channels:
    #    - pytorch
    #    platforms:
    #    - linux-64
    #    - win-64


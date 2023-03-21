[![PyPI - Downloads](https://img.shields.io/pypi/dm/ddpm)](https://pypi.org/project/ddpm/)
[![PyPI](https://img.shields.io/pypi/v/ddpm)](https://pypi.org/project/ddpm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ddpm)](https://pypi.org/project/ddpm/)

# DDPM Pytorch

Pytorch implementation of "_Improved Denoising Diffusion Probabilistic Models_", 
"_Denoising Diffusion Probabilistic Models_" and "_Classifier-free Diffusion Guidance_"

![](https://hojonathanho.github.io/diffusion/assets/img/pgm_diagram_xarrow.png)


# How to use

1.  Install pip package containing the pytorch lightning model, which includes also the training step

        pip install ddpm

2. Clone the repository

        git clone git@github.com:Michedev/VAE_anomaly_detection.git

# How to train

1. Install [anaconda](https://www.anaconda.com/) or miniconda, but install anaconda-project `conda install anaconda-project`


2. Train the model

       anaconda-project run train 

   By default, the version of trained DDPM is from "Improved Denoising Diffusion Probabilistic Models" paper on MNIST dataset.
   You can switch to the original DDPM by disabling the variational lower bound with the following command:
      
       anaconda-project run train model.vlb=False
   
   You can also train the DDPM with the Classifier-free Diffusion Guidance by changing the model:

       anaconda-project run train model=unet_class_conditioned
    
    or via the shortcut

       anaconda-project run train-class-conditioned

    Finally, under saved_models/{train-datetime} you can find the trained model, the tensorboard logs, the training config

# How to generate

1. Train a model (See previous section)

2. Generate a new batch of images

       anaconda-project run generate -r RUN

   The other options are: `[--seed SEED] [--device DEVICE] [--batch-size BATCH_SIZE] [-w W] [--scheduler {linear,cosine,tan}] [-T T]`

# Configure the training

Under _config_ there are several yaml files containing the training parameters 
such as model class and paramters, noise steps, scheduler and so on. 
Note that the hyperparameters in such files are taken from 
the papers "_Improved Denoising Diffusion Probabilistic Models_" 
and "_Denoising Diffusion Probabilistic Models_". Down below the explaination of the config file for train the model:

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
    ema: true  # use Exponential Moving Average implemented in ema.py
    ema_decay: 0.99  # decay factor of EMA

    hydra:
      run:
        dir: saved_models/${now:%Y_%m_%d_%H_%M_%S}

# Project structure

      .
      ├── anaconda-project-lock.yml  # lock file for anaconda-project
      ├── anaconda-project.yml  # project specs
      ├── callbacks  # Pytorch Lightning callbacks for training
      │   ├── ema.py  # exponential moving average callback
      ├── config  # config files for training for hydra
      │   ├── dataset  # dataset config files
      │   ├── model  # model config files
      │   ├── model_dataset  # specific (model, dataset) config
      │   ├── model_scheduler  # specific (model, scheduler) config
      │   ├── scheduler  # scheduler config files
      │   └── train.yaml  # training config file
      ├── generate.py  # script for generating images
      ├── model  # model files
      │   ├── classifier_free_ddpm.py  # Classifier-free Diffusion Guidance
      │   ├── ddpm.py  # Denoising Diffusion Probabilistic Models
      │   ├── distributions.py  # distributions functions for diffusion
      │   ├── unet_class.py  # UNet model for Classifier-free Diffusion Guidance
      │   └── unet.py  # UNet model for Denoising Diffusion Probabilistic Models
      ├── pyproject.toml  # setuptool file to publish model/ to pypi
      ├── readme.md   # this file
      ├── readme_pip.md  # readme for pypi
      ├── train.py  # script for training
      ├── utils  # utility functions
      └── variance_scheduler  # variance scheduler files
          ├── cosine.py  # cosine variance scheduler
          └── linear.py  # linear variance scheduler

### Add custom dataset

To add a custom dataset, you need to create a new class that inherits from torch.utils.data.Dataset
and implement the __len__ and __getitem__ methods. 
Then, you need to add the config file to the _config/dataset_ folder with a similar
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

# Examples of custom training

### Disable the variational lower bound, use Linear scheduler, use 1000 noise steps, train in GPU

    anaconda-project run train scheduler=linear accelerator='gpu' model.vlb=False noise_steps=1000


## Classifier-free Guidance

Use the labels for __Diffusion Guidance__, as in "_Classifier-free Diffusion Guidance_" with the following command

    anaconda-project run train model=unet_class_conditioned noise_steps=1000

## Add your scheduler

1. Add a new class (preferabily under `variance_scheduler/`) which subclasses `Scheduler` class or just copy the same methods syntax of `Scheduler`
2. Define a new config under `config/scheduler` with the name _my-scheduler.yaml_ containing the following fields

```   
 _target_: {your scheduler import path} (e.g. variance_scheduler.Linear)
... // your scheduler additional parameters
```

Finally train with the following command

    anaconda-project run train scheduler=my-scheduler

## Add your dataset

1. Add a new class which subclasses `torch.utils.data.Dataset`

2. Define a new config under `config/dataset` with the name _my-dataset.yaml_ containing the following fields

```   
width: ???
height: ???
channels: ???
train:
  _target_: {your dataset import path} (e.g. torchvision.datasets.MNIST)
  // your dataset additional parameters
val:
  _target_: {your dataset import path} (e.g. torchvision.datasets.MNIST)
  // your dataset additional parameters
```

Finally train with the following command

    anaconda-project run train dataset=my-dataset
    
### Anaconda-project
#### mac-os lock file

1. Remove cudatoolkit from _anaconda-project.yml_ file at the bottom of the file, 
under `env_specs -> default -> packages`
2. De-comment `- osx-64`  under `env_specs -> default -> platforms`
3. Delete _anaconda-project-lock.yml_ file
4. Run `anaconda-project prepare` to generate the new lock file

## CPU-only environment

To have an alternative a PyTorch CPU-only environment, 
de-comment the following lines at the bottom of _anaconda-project.yml_
under `env_specs`

    #  pytorch-cpu:
    #    packages:
    #    - cpuonly
    #    channels:
    #    - pytorch
    #    platforms:
    #    - linux-64
    #    - win-64


# DDPM Pytorch

Pytorch implementation of "_Improved Denoising Diffusion Probabilistic Models_", 
"_Denoising Diffusion Probabilistic Models_" and "_Classifier-free Diffusion Guidance_"

![](https://hojonathanho.github.io/diffusion/assets/img/pgm_diagram_xarrow.png)


# How to train

1. Install [Poetry](https://python-poetry.org/) 

2. Install all the dependencies with the command

        poetry install
  
3. Train the model

       poetry run python train.py 


# Configure the training

Under _ddpm_pytorch/config_ there are several yaml files containing the training parameters such as model class and paramters, noise steps, scheduler and so on. Note that the hyperparameters in such files are taken from the papers "_Improved Denoising Diffusion Probabilistic Models_" and "_Denoising Diffusion Probabilistic Models_"

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

### Examples of custom training

__Disable the variational lower bound__, hence training like in "_Denoising Diffusion Probabilistic Models_" with __linear__ scheduler and in __GPU__

      poetry run python ddpm_pytorch/train.py scheduler=linear accelerator='gpu' model.vlb=False noise_steps=1000

Use the labels as __Diffusion Guidance__, as in "_Classifier-free Diffusion Guidance_"

      poetry run python ddpm_pytorch/train.py model=unet_class_conditioned noise_steps=1000

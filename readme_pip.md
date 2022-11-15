# DDPMs Pytorch Implementation

Pytorch implementation of "_Improved Denoising Diffusion Probabilistic Models_", 
"_Denoising Diffusion Probabilistic Models_" and "_Classifier-free Diffusion Guidance_"

## Install

```bash
pip install ddpm
```

# Usage

## Gaussian plain DDPM
```python

from ddpm import GaussianDDPM, UNetTimeStep
from ddpm.variance_scheduler import LinearScheduler

T = 1_000
width = 32
height = 32
channels = 3

# Create a Gaussian DDPM with 1000 noise steps
scheduler = LinearScheduler(T=T, beta_start=1e-5, beta_end=1e-2)
denoiser = UNetTimeStep(channels=[3, 128, 256, 256, 384],
                        kernel_sizes=[3, 3, 3, 3],
                        strides=[1, 1, 1, 1],
                        paddings=[1, 1, 1, 1],
                        p_dropouts=[0.1, 0.1, 0.1, 0.1],
                        time_embed_size=100, 
                        downsample=True)
model = GaussianDDPM(denoiser, T, scheduler, vlb=False, lambda_variational=1.0, width=width, 
                     height=height, input_channels=channels, logging_freq=1_000)  # pytorch lightning module

```

## Gaussian "Improved" DDPM

```python

from ddpm import GaussianDDPM, UNetTimeStep
from ddpm.variance_scheduler import CosineScheduler

T = 1_000
width = 32
height = 32
channels = 3

# Create a Gaussian DDPM with 1000 noise steps
scheduler = CosineScheduler(T=T)
denoiser = UNetTimeStep(channels=[3, 128, 256, 256, 384],
                        kernel_sizes=[3, 3, 3, 3],
                        strides=[1, 1, 1, 1],
                        paddings=[1, 1, 1, 1],
                        p_dropouts=[0.1, 0.1, 0.1, 0.1],
                        time_embed_size=100, 
                        downsample=True)
model = GaussianDDPM(denoiser, T, scheduler, vlb=True, lambda_variational=0.0001, width=width, 
                     height=height, input_channels=channels, logging_freq=1_000)  # pytorch lightning module

```

## Classifier-free Diffusion Guidance

```python

from ddpm import GaussianDDPMClassifierFreeGuidance, UNetTimeStep
from ddpm.variance_scheduler import CosineScheduler

T = 1_000
width = 32
height = 32
channels = 3
num_classes = 10

# Create a Gaussian DDPM with 1000 noise steps
scheduler = CosineScheduler(T=T)
denoiser = UNetTimeStep(channels=[3, 128, 256, 256, 384],
                        kernel_sizes=[3, 3, 3, 3],
                        strides=[1, 1, 1, 1],
                        paddings=[1, 1, 1, 1],
                        p_dropouts=[0.1, 0.1, 0.1, 0.1],
                        time_embed_size=100, 
                        downsample=True)
model = GaussianDDPMClassifierFreeGuidance(denoiser, T, w=0.3, v=0.2, variance_scheduler=scheduler, width=width, 
                                           height=height, input_channels=channels, logging_freq=1_000, p_uncond=0.2, 
                                           num_classes=num_classes)  # pytorch lightning module

```


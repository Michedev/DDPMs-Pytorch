import hydra
import pkg_resources
import torch
from omegaconf import DictConfig


@hydra.main(pkg_resources.resource_filename("ddpm_pytorch", 'config'), 'train.yaml')
def train(config: DictConfig):
    pass
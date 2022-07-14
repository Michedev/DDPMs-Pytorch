import hydra
import pkg_resources
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


@hydra.main(pkg_resources.resource_filename("ddpm_pytorch", 'config'), 'train.yaml')
def train(config: DictConfig):
    scheduler = hydra.utils.instantiate(config.scheduler)
    model: pl.LightningModule = hydra.utils.instantiate(config.model, variance_scheduler=scheduler)
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset.train)
    val_dataset: Dataset = hydra.utils.instantiate(config.dataset.val)
    pin_memory = 'cuda' in config.device
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    trainer = pl.Trainer()
    trainer.fit(model, train_dl, val_dl)
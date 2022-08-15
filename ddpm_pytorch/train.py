import hydra
import pkg_resources
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
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
    ckpt_callback = ModelCheckpoint('./', monitor='loss/val_loss', every_n_train_steps=config.checkpoint_steps)
    trainer = pl.Trainer(checkpoint_callback=ckpt_callback, accelerator=config.accelerator, devices=config.devices)
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    train()

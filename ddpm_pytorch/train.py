import hydra
import pkg_resources
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import omegaconf

from ema import EMA


@hydra.main(pkg_resources.resource_filename("ddpm_pytorch", 'config'), 'train.yaml')
def train(config: DictConfig):
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)
    scheduler = hydra.utils.instantiate(config.scheduler)
    model: pl.LightningModule = hydra.utils.instantiate(config.model, variance_scheduler=scheduler)
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset.train)
    val_dataset: Dataset = hydra.utils.instantiate(config.dataset.val)

    model.save_hyperparameters(config)

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    ckpt_callback = ModelCheckpoint('./', 'epoch={epoch}-val_loss={loss/val_loss}', monitor='loss/val_loss',
                                    auto_insert_metric_name=False)
    callbacks = [ckpt_callback]
    if config.ema:
        callbacks.append(EMA(config.ema_decay))
    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val,
                         gradient_clip_algorithm=config.gradient_clip_algorithm)
    trainer.fit(model, train_dl, val_dl)


if __name__ == '__main__':
    train()

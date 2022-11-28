import hydra
import pkg_resources
from omegaconf import DictConfig, OmegaConf
from path import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import omegaconf
import os

from callbacks.ema import EMA


@hydra.main(pkg_resources.resource_filename("ddpm_pytorch", 'config'), 'train.yaml')
def train(config: DictConfig):
    ckpt = None
    pl.seed_everything(config.seed)
    if config.ckpt is not None:
        os.chdir(Path(__file__).parent.parent.abspath())
        assert Path(config.ckpt).exists()
        ckpt = Path(config.ckpt)
        config = OmegaConf.load(ckpt.parent / 'config.yaml')
        os.chdir(ckpt.parent.abspath())
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)
    scheduler = hydra.utils.instantiate(config.scheduler)
    model: pl.LightningModule = hydra.utils.instantiate(config.model, variance_scheduler=scheduler)
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset.train)
    val_dataset: Dataset = hydra.utils.instantiate(config.dataset.val)

    if ckpt is not None:
        model.load_from_checkpoint(ckpt.basename(), variance_scheduler=scheduler)

    model.save_hyperparameters(OmegaConf.to_object(config)['model'])

    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    ckpt_callback = ModelCheckpoint('./', 'epoch={epoch}-valid_loss={loss/valid_loss_epoch}', monitor='loss/valid_loss_epoch',
                                    auto_insert_metric_name=False, save_last=True)
    callbacks = [ckpt_callback]
    if config.ema:
        callbacks.append(EMA(config.ema_decay))
    if config.early_stop:
        callbacks.append(EarlyStopping('loss/valid_loss_epoch', min_delta=config.min_delta,
                                       patience=config.patience))
    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val,
                         gradient_clip_algorithm=config.gradient_clip_algorithm)
    trainer.fit(model, train_dl, val_dl, )


if __name__ == '__main__':
    train()

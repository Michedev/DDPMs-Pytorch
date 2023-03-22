import hydra
from omegaconf import DictConfig, OmegaConf
from path import Path
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import omegaconf
import os

from callbacks.ema import EMA
from utils.paths import MODEL


# This function is the entry point for the training script. It takes a DictConfig object as an argument, which contains
# the configuration for the training run. The configuration is loaded from a YAML file using Hydra.
@hydra.main('config', 'train.yaml')
def train(config: DictConfig):
    # Initialize checkpoint to None
    ckpt = None

    # Set random seeds for reproducibility
    pl.seed_everything(config.seed)

    # If a checkpoint is specified in the config, load it and update the config accordingly
    if config.ckpt is not None:
        # Change the current working directory to the parent directory of the checkpoint file
        os.chdir(os.path.dirname(config.ckpt))

        # Assert that the checkpoint file exists
        assert os.path.exists(config.ckpt)

        # Set ckpt to the path of the checkpoint file
        ckpt = config.ckpt

        # Load the configuration file associated with the checkpoint file
        config = OmegaConf.load(os.path.join(os.path.dirname(ckpt), 'config.yaml'))

    # Save the updated configuration to a file called 'config.yaml'
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)

    Path.getcwd().joinpath('gen_images').makedirs_p()
    # copy paste model/ folder
    MODEL.copytree(Path.getcwd().joinpath('model'))

    # Create the variance scheduler and a deep generative model using Hydra
    scheduler = hydra.utils.instantiate(config.scheduler)
    opt = hydra.utils.instantiate(config.optimizer)
    model: pl.LightningModule = hydra.utils.instantiate(config.model, variance_scheduler=scheduler, opt=opt)

    # Create training and validation datasets using Hydra
    train_dataset: Dataset = hydra.utils.instantiate(config.dataset.train)
    val_dataset: Dataset = hydra.utils.instantiate(config.dataset.val)

    # If a checkpoint is specified, load the model weights from the checkpoint
    if ckpt is not None:
        model.load_from_checkpoint(ckpt, variance_scheduler=scheduler)

    # Save the hyperparameters of the model to a file called 'hparams.yaml'
    model.save_hyperparameters(OmegaConf.to_object(config)['model'])

    # Create PyTorch dataloaders for the training and validation datasets
    pin_memory = 'gpu' in config.accelerator
    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, pin_memory=pin_memory)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, pin_memory=pin_memory)

    # Create a ModelCheckpoint callback that saves the model weights to disk during training
    ckpt_callback = ModelCheckpoint('./', 'epoch={epoch}-valid_loss={loss/val_loss_epoch}', 
                                     monitor='loss/val_loss_epoch', auto_insert_metric_name=False, save_last=True)
    callbacks = [ckpt_callback]

    # Add additional callbacks if specified in the configuration file
    if config.ema:
        # Create an Expontential Moving Average callback
        callbacks.append(EMA(config.ema_decay))  
    if config.early_stop:
        callbacks.append(EarlyStopping('loss/val_loss_epoch', min_delta=config.min_delta, patience=config.patience))

    trainer = pl.Trainer(callbacks=callbacks, accelerator=config.accelerator, devices=config.devices,
                         gradient_clip_val=config.gradient_clip_val, gradient_clip_algorithm=config.gradient_clip_algorithm)
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    import sys
    sys.path.append(Path(__file__).parent.abspath())
    train()

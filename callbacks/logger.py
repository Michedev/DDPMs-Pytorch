from typing import Any
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torchvision
import torch

class LoggerCallback(Callback):

    def __init__(self, freq_train_log: int, freq_train_norm_gradients: int, batch_size_gen_images: int) -> None:
        super().__init__()
        self.freq_train = freq_train_log
        self.freq_train_norm_gradients = freq_train_norm_gradients
        self.batch_size_gen_images = batch_size_gen_images

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.freq_train == 0:
            pl_module.log("train/loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True, logger=True)
            pl_module.log("train/noise_loss", outputs["noise_loss"], on_step=True, on_epoch=False, prog_bar=True, logger=True)
            if outputs['vlb_loss'] is not None:
                pl_module.log("train/vlb_loss", outputs["vlb_loss"], on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.global_step % self.freq_train_norm_gradients == 0:
            norm_grad = 0
            for p in pl_module.parameters():
                if p.grad is not None:
                    norm_grad += p.grad.norm(2).item()
            pl_module.log("train/norm_grad", norm_grad, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        pl_module.log("val/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("val/noise_loss", outputs["noise_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if outputs['vlb_loss'] is not None:
            pl_module.log("val/vlb_loss", outputs["vlb_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        gen_images = pl_module.generate(batch_size=self.batch_size_gen_images) # Generate images
        gen_images = torchvision.utils.make_grid(gen_images)  # Convert to grid
        pl_module.logger.experiment.add_image('gen_val_images', gen_images, trainer.current_epoch)  # Log the images
        torchvision.utils.save_image(gen_images, f'gen_images/epoch={pl_module.current_epoch}.png')  # Save the images

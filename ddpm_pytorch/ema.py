from typing import Any

import pytorch_lightning as pl
import torch


class EMA(pl.Callback):

    def __init__(self, decay_factor: float):
        assert 0.0 <= decay_factor <= 1.0
        self.decay_factor = decay_factor
        self.dict_params = dict()

    @torch.no_grad()
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for n, p in pl_module.named_parameters():
            self.dict_params[n] = p

    @torch.no_grad()
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        for n, p in pl_module.named_parameters():
            self.dict_params[n] = self.dict_params[n] * (1.0 - self.decay_factor) + p * self.decay_factor
            torch.fill_(p, self.dict_params[n])

from typing import Any

import pytorch_lightning as pl
import torch


class EMA(pl.Callback):
    """
    Exponential Moving Average
    Let \beta the smoothing parameter, p the current parameter value and v the accumulated value, the EMA is calculated
    as follows

    v_t = \beta * p_{t-1} + (1 - \beta) * v_{t-1}
    p_t = v_t
    """

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
            p.fill_(self.dict_params[n])

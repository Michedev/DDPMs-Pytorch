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
        """
        For each parameter in the model, we add the parameter to the dictionary
        
        :param trainer: The trainer object
        :type trainer: "pl.Trainer"
        :param pl_module: The LightningModule that is being trained
        :type pl_module: "pl.LightningModule"
        """
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
        """
        For each parameter in the model, we multiply the parameter by a decay factor and add the current
        parameter multiplied by the decay factor to the parameter in the dictionary
        
        :param trainer: The trainer object
        :type trainer: "pl.Trainer"
        :param pl_module: The LightningModule that is being trained
        :type pl_module: "pl.LightningModule"
        :param batch: The batch of data that is being passed to the model
        :type batch: Any
        :param batch_idx: the index of the batch within the current epoch
        :type batch_idx: int
        :param unused: int = 0, defaults to 0
        :type unused: int (optional)
        """
        for n, p in pl_module.named_parameters():
            self.dict_params[n] = self.dict_params[n] * (1.0 - self.decay_factor) + p * self.decay_factor
            p[:] = self.dict_params[n][:]

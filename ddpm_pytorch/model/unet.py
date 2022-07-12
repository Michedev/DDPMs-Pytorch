from typing import Dict

import pytorch_lightning as pl
from torch import nn

class ConvBlock(nn.Module):

    def __init__(self, **conv_params):
        super().__init__()
        self.conv = nn.Conv2d(**conv_params)
        self.groupnorm = nn.


    def forward(self, x):


class DDPMUNet(pl.LightningModule):

    def forward(self, x) -> Dict:
        pass

    def training_step(self, batch, batch_idx):
        pass
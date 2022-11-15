from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from model.unet import ResBlockTimeEmbed, ImageSelfAttention
import tensorguard as tg


class ResBlockTimeEmbedClassConditioned(ResBlockTimeEmbed):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 time_embed_size: int, p_dropout: float, num_classes: int, class_embed_size: int,
                 assert_shapes: bool = True):
        super().__init__(in_channels + class_embed_size, out_channels, kernel_size, stride, padding,
                         time_embed_size, p_dropout)
        self.linear_map_class = nn.Sequential(
            nn.Linear(num_classes, class_embed_size),
            nn.ReLU(),
            nn.Linear(class_embed_size, class_embed_size)

        )

        self.assert_shapes = assert_shapes

    def forward(self, x, time_embed, c):
        emb_c = self.linear_map_class(c)
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])
        if self.assert_shapes: tg.guard(emb_c, "B, C, W, H")
        x = torch.cat([x, emb_c], dim=1)
        return super().forward(x, time_embed)


class UNetTimeStepClassConditioned(nn.Module):
    """
    UNet architecture with class and time embedding injected in every residual block, both in downsample and upsample.
    Both information are mapped via an 2-layers MLP to a fixed embedding size.
    After the third downsample block a self-attention layer is applied.
    """

    def __init__(self, channels: List[int], kernel_sizes: List[int], strides: List[int], paddings: List[int],
                 downsample: bool, p_dropouts: List[float], time_embed_size: int, num_classes: int,
                 class_embed_size: int,
                 assert_shapes: bool = True):
        super().__init__()
        assert len(channels) == (len(kernel_sizes) + 1) == (len(strides) + 1) == (len(paddings) + 1) == \
               (len(p_dropouts) + 1), f'{len(channels)} == {(len(kernel_sizes) + 1)} == ' \
                                      f'{(len(strides) + 1)} == {(len(paddings) + 1)} == \
                                                              {(len(p_dropouts) + 1)}'
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.assert_shapes = assert_shapes
        self.num_classes = num_classes
        self.time_embed_size = time_embed_size
        self.class_embed_size = class_embed_size
        self.downsample_blocks = nn.ModuleList([
            ResBlockTimeEmbedClassConditioned(channels[i], channels[i + 1], kernel_sizes[i], strides[i],
                                              paddings[i], time_embed_size, p_dropouts[i], num_classes,
                                              class_embed_size, assert_shapes) for i in range(len(channels) - 1)
        ])

        self.use_downsample = downsample
        self.downsample_op = nn.MaxPool2d(kernel_size=2)
        self.middle_block = ResBlockTimeEmbedClassConditioned(channels[-1], channels[-1], kernel_sizes[-1], strides[-1],
                                                              paddings[-1], time_embed_size, p_dropouts[-1],
                                                              num_classes, class_embed_size, assert_shapes)
        self.upsample_blocks = nn.ModuleList([
            ResBlockTimeEmbedClassConditioned((2 if i != 0 else 1) * channels[-i - 1], channels[-i - 2],
                                              kernel_sizes[-i - 1],
                                              strides[-i - 1], paddings[-i - 1], time_embed_size, p_dropouts[-i - 1],
                                              num_classes,
                                              class_embed_size, assert_shapes) for i in range(len(channels) - 1)
        ])
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in p_dropouts])
        self.p_dropouts = p_dropouts
        self.self_attn = ImageSelfAttention(channels[2])
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_size),
            nn.GELU(),
            nn.Linear(self.time_embed_size, self.time_embed_size),
        )

    def forward(self, x: torch.FloatTensor, t: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_channels = x.shape[1]
        if self.assert_shapes: tg.guard(x, "B, C, W, H")
        if self.assert_shapes: tg.guard(c, "B, NUMCLASSES")
        time_embedding = self.time_embed(t)
        if self.assert_shapes: tg.guard(time_embedding, "B, TE")
        h = self.forward_unet(x, time_embedding, c)
        x_recon = h
        if self.assert_shapes: tg.guard(x_recon, "B, C, W, H")
        return x_recon

    def forward_unet(self, x, time_embedding, c):
        hs = []
        h = x
        for i, downsample_block in enumerate(self.downsample_blocks):
            h = downsample_block(h, time_embedding, c)
            if i == 2:
                h = self.self_attn(h)
            h = self.dropouts[i](h)
            if i != (len(self.downsample_blocks) - 1): hs.append(h)
            if self.use_downsample and i != (len(self.downsample_blocks) - 1):
                h = self.downsample_op(h)
        h = self.middle_block(h, time_embedding, c)
        for i, upsample_block in enumerate(self.upsample_blocks):
            if i != 0:
                h = torch.cat([h, hs[-i]], dim=1)
            h = upsample_block(h, time_embedding, c)
            if self.use_downsample and (i != (len(self.upsample_blocks) - 1)):
                h = F.interpolate(h, size=hs[-i - 1].shape[-1], mode='nearest')
        return h

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Callable
from efficientnet_mbconv import MBConvBlock

class Activation(nn.Module):
    def __init__(self, activation_fn):
        super(Activation, self).__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(x)

class EfficientNetB0(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 **mb_kwargs):
        """EfficientNetB0 model.

        Args:
            num_classes (int): Number of output classes.
            **mb_kwargs: Additional keyword arguments. see `MBConvBlock`
        """
        super(EfficientNetB0, self).__init__()

        # get activation from kw
        activation_fn = mb_kwargs.get("activation", F.silu) # default value is silu

        # Stem
        self.init_conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Changed stride to 2 for downsampling
            nn.BatchNorm2d(32),
            Activation(activation_fn)
        )

        # MBConv Blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, kernel_size=3, stride=1, expansion_factor=1, **mb_kwargs),
            MBConvBlock(16, 24, kernel_size=3, stride=2, expansion_factor=6, **mb_kwargs),
            MBConvBlock(24, 24, kernel_size=3, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(24, 40, kernel_size=5, stride=2, expansion_factor=6, **mb_kwargs),
            MBConvBlock(40, 40, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(40, 80, kernel_size=3, stride=2, expansion_factor=6, **mb_kwargs),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(80, 80, kernel_size=3, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(80, 112, kernel_size=5, stride=2, expansion_factor=6, **mb_kwargs),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(112, 112, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(112, 192, kernel_size=5, stride=2, expansion_factor=6, **mb_kwargs),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(192, 192, kernel_size=5, stride=1, expansion_factor=6, **mb_kwargs),
            MBConvBlock(192, 320, kernel_size=3, stride=1, expansion_factor=6, **mb_kwargs),
            nn.Conv2d(320, 320, kernel_size=1, stride=1, padding=0),
        )

        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, 1280),
            nn.LayerNorm(1280),
            Activation(activation_fn),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.init_conv_block(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

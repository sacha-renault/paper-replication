from typing import Union, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MBConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, Tuple[int, int]], 
                 stride: Union[int, Tuple[int, int]], 
                 expansion_factor: int, 
                 alpha: float = 0.25,
                 activation: Callable = F.silu,
                 drop_rate: float = 0.0):
        super(MBConvBlock, self).__init__()
        
        # Compute Exp and SE channels
        expanded_channels = int(in_channels * expansion_factor)
        se_channels = max(1, int(in_channels * alpha))
        
        # Expansion phase, only if the expansion factor isn't 1
        if expansion_factor > 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        # otherwise it's legit a non op
        else:
            self.expand_conv = None
            self.expand_bn = None
        
        # Depthwise conv
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=kernel_size // 2, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # SE phase
        self.se_global_pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Conv2d(expanded_channels, se_channels, kernel_size=1, stride=1, padding=0)
        self.se_expand = nn.Conv2d(se_channels, expanded_channels, kernel_size=1, stride=1, padding=0)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # Dropout if not 0
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
        
        # If channel num is same for in and out (AND strides = 1), we can use skip connection
        self.use_skip = (stride == 1 and in_channels == out_channels)
        
        # Activation function used
        self.activation = activation
        
    def forward(self, input_tensor):
        # Keep a reference to input for skip connection
        x = input_tensor

        # Expansion phase (only if expansion_factor > 1)
        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.activation(x)
        
        # Depthwise Convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)
        
        # Squeeze-and-Excitation
        se = self.se_global_pool(x)
        se = self.activation(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se
        
        # Projection phase
        x = self.project_conv(x)
        x = self.project_bn(x)
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Skip connection if applicable
        if self.use_skip:
            x = x + input_tensor
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeedForward(nn.Module):
    def __init__(self, input_features: int, hidden_dim: int, output_features: int):
        super(FeedForward, self).__init__()

        self.w1 = nn.Linear(input_features, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.w1(x)
        x = F.relu(x)
        return self.w2(x)
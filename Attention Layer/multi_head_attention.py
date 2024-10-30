import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from single_head_attention import SingleHeadAttentionLayer

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Linear projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Final linear for output
        self.linear_o = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        # Start with linear projection
        # Shape: (batch_size, seq_length, d_model)
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Reshaping
        # Shape: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Perform attention scores
        x = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        if mask is not None:
            x = x.masked_fill(mask == 0, float('-inf'))

        # softmax to convert logit scores to probs
        x = torch.softmax(x, dim=-1)

        # perform output
        x = x @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linear_o(x)

if __name__ == "__main__":
    # Dummy data
    batch_size = 2
    seq_length = 3
    d_model = 64
    q = torch.rand(batch_size, seq_length, d_model)
    k = torch.rand(batch_size, seq_length, d_model)
    v = torch.rand(batch_size, seq_length, d_model)

    attention_layer = MultiHeadAttentionLayer(8, d_model)
    attention_layer(q, k, v)


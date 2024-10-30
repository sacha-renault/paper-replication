import torch
import torch.nn as nn
import torch.nn.functional as F
from single_head_attention import SingleHeadAttentionLayer

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_heads: int, d_model: int):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)

        self.attention_layer = SingleHeadAttentionLayer(self.head_dim)

    def forward(self, q, k, v, mask=None):
        # Shape: (batch_size, seq_length, d_model)
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Shape: (batch_size, num_heads, seq_length, head_dim)
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        headi = []
        for i in range(self.num_heads):
            head_output = self.attention_layer(q[:, i], k[:, i], v[:, i], mask)
            headi.append(head_output)

        headn = torch.cat(headi, dim=-1)  # Shape: (batch_size, seq_length, d_model)
        return self.linear_o(headn)

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


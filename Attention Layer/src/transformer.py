import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from feed_forward import FeedForward
from multi_head_attention import MultiHeadAttentionLayer

class AddAndNorm(nn.Module):
    def __init__(self, size: int):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x1, x2) -> Tensor:
        return self.norm(x1 + x2)

class NxEncoderBlock(nn.Module):
    def __init__(self,
                 mha_heads: int = 8,
                 d_model: int = 512,
                 ffn_hidden_dim: int = 2048):
        super(NxEncoderBlock, self).__init__()
        self.ffn = FeedForward(d_model, ffn_hidden_dim, d_model)
        self.mha = MultiHeadAttentionLayer(mha_heads, d_model)
        self.add_and_norm1 = AddAndNorm(d_model)
        self.add_and_norm2 = AddAndNorm(d_model)

    def forward(self, x: Tensor, mask = None) -> Tensor:
        # Self-attention with add & norm
        attention_output = self.mha(x, x, x, mask)
        x = self.add_and_norm1(x, attention_output)

        # Feed-forward with add & norm
        ffn_output = self.ffn(x)
        return self.add_and_norm2(x, ffn_output)

class NxDecoderBlock(nn.Module):
    def __init__(self,
                 mha_heads: int = 8,
                 d_model: int = 512,
                 ffn_hidden_dim: int = 2048):
        super(NxDecoderBlock, self).__init__()
        self.ffn = FeedForward(d_model, ffn_hidden_dim, d_model)
        self.self_attention = MultiHeadAttentionLayer(mha_heads, d_model)
        self.cross_attention = MultiHeadAttentionLayer(mha_heads, d_model)
        self.add_and_norm1 = AddAndNorm(d_model)
        self.add_and_norm2 = AddAndNorm(d_model)
        self.add_and_norm3 = AddAndNorm(d_model)

    def forward(self, x: Tensor, encoder_output: Tensor, mask = None) -> Tensor:
        # Self-attention with add & norm
        self_attention_output = self.self_attention(x, x, x, mask)
        x = self.add_and_norm1(x, self_attention_output)

        # Cross-attention with add & norm
        cross_attention_output = self.cross_attention(x, encoder_output, encoder_output, mask)
        x = self.add_and_norm2(x, cross_attention_output)

        # Feed-forward with add & norm
        ffn_output = self.ffn(x)
        return self.add_and_norm3(x, ffn_output)


class Transformer(nn.Module):
    def __init__(self,
                 mha_heads: int = 8,
                 d_model: int = 512,
                 ffn_hidden_dim: int = 2048,
                 n_encoder_block: int = 6,
                 n_decoder_block: int = 6,
                 vocab_size: int = 512):
        super(Transformer, self).__init__() # forgot this previously

        self.encoder = nn.ModuleList([
            NxEncoderBlock(mha_heads, d_model, ffn_hidden_dim) for _ in range(n_encoder_block)])

        self.decoder = nn.ModuleList([
            NxDecoderBlock(mha_heads, d_model, ffn_hidden_dim) for _ in range(n_decoder_block)])

        self.linear_output = nn.Linear(d_model, vocab_size)

    def forward(self, inputs: Tensor, outputs: Tensor, mask = None) -> Tensor:
        # encoder part
        x = inputs
        for block in self.encoder:
            x = block(x, mask)

        # keep a reference on the output of decoder
        encoder_output = x

        # decoder part, it takes the output of encoder as query
        # for the cross attention part
        x = outputs
        for block in self.decoder:
            x = block(x, encoder_output, mask)

        # apply a final linear layer
        x = self.linear_output(x)
        return F.softmax(x, dim=-1)

if __name__ == "__main__":
    model = Transformer()

    with torch.no_grad():
        output = model(torch.randn(1, 25, 512), torch.randn(1, 25, 512))
        print(output.shape)
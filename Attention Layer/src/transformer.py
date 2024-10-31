import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .feed_forward import FeedForward
from .multi_head_attention import MultiHeadAttentionLayer

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
                 vocab_size: int = 512,
                 max_sequence_len: int = 100):
        super(Transformer, self).__init__() # forgot this previously
        self.d_model = d_model
        self._max_len = max_sequence_len
        self._set_positional_encoding()

        # encoder
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList([
            NxEncoderBlock(mha_heads, d_model, ffn_hidden_dim) for _ in range(n_encoder_block)])

        # decoder
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = nn.ModuleList([
            NxDecoderBlock(mha_heads, d_model, ffn_hidden_dim) for _ in range(n_decoder_block)])

        # output
        self.linear_output = nn.Linear(d_model, vocab_size)

    def _set_positional_encoding(self) -> None:
        # positional encoding
        positions = torch.arange(0, self._max_len).unsqueeze(1).repeat(1, self.d_model)
        dimensions = torch.arange(0, self.d_model).unsqueeze(0)
        angle_rates = 1 / (10_000 ** (dimensions // 2 * 2 / self.d_model))
        positional_encoding = torch.zeros((self._max_len, self.d_model))
        positional_encoding[:, 0::2] = torch.sin(positions * angle_rates[:, 0::2])
        positional_encoding[:, 1::2] = torch.cos(positions * angle_rates[:, 1::2])
        self.positional_encoding = positional_encoding.unsqueeze(0)

    @property
    def max_sequence_length(self) -> int:
        return self._max_len

    @max_sequence_length.setter
    def max_sequence_length(self, new_value: int) -> None:
        self._max_len = new_value
        self._set_positional_encoding()

    def forward(self, inputs: Tensor, outputs: Tensor, encoder_mask = None, decoder_mask = None) -> Tensor:
        # ensure positional encoding isn't smaller than max len
        assert inputs.size(1) <= self._max_len and outputs.size(1) <= self._max_len (
            "Input size is greater than max sequence length"
            "Considere truncating or increase max seq size")

        # encoder part
        x = self.encoder_embedding(inputs) * (self.d_model ** 0.5)
        x += self.positional_encoding[:, :inputs.size(1), :]
        for block in self.encoder:
            x = block(x, encoder_mask)

        # keep a reference on the output of decoder
        encoder_output = x

        # decoder part, it takes the output of encoder as query
        # for the cross attention part
        x = self.decoder_embedding(outputs) * (self.d_model ** 0.5)
        x += self.positional_encoding[:, :outputs.size(1), :]
        for block in self.decoder:
            x = block(x, encoder_output, decoder_mask)

        # apply a final linear layer
        x = self.linear_output(x)

        # in training mode, we just wanna return
        # the raw logit because we will use CrossEntropy afterward
        # that already contains softmax
        if self.training:
            return x

        # else, we return the probability of token
        else:
            return F.softmax(x, dim=-1)

if __name__ == "__main__":
    model = Transformer()

    with torch.no_grad():
        output = model(torch.randn(1, 25, 512), torch.randn(1, 25, 512))
        print(output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttentionLayer(nn.Module):
    def __init__(self):
        super(SingleHeadAttentionLayer, self).__init__()

        # init softmax layer
        self.softmax = nn.Softmax(2)

    def forward(self, q, k, v, mask = None):
        x = q @ k.transpose(-2, -1)     # matmul QxK.T
        x = x / (k.size(-1) ** 0.5)     # scaling using kqdim
        if mask is not None:
            mask = mask.unsqueeze(1)
            x += mask                   # masks need to be -inf to have outputs of 0 where masked
        x = self.softmax(x)
        return x @ v

    def forward_workaround(self, q, k, v, mask = None):
        x = q @ k.transpose(-2, -1)     # matmul QxK.T
        x = x / (k.size(-1) ** 0.5)     # scaling using kqdim
        x = self.softmax(x)             # softmaxing before masking
        if mask is not None:
            mask = mask.unsqueeze(1)    # this time masks is a boolean tensor (0 where masked) x isn't a probability
            x *= mask                   # anymore since the sums of x over axis 0 aren't equal to 0

        x /= torch.sum(x, axis = -1, keepdim=True)     # retrieving a probability score
        return x @ v

if __name__ == "__main__":
    # Dummy data
    batch_size = 2
    seq_length = 3
    d_model = 4
    q = torch.rand(batch_size, seq_length, d_model)
    k = torch.rand(batch_size, seq_length, d_model)
    v = torch.rand(batch_size, seq_length, d_model)

    # Dummy mask
    mask = torch.tensor([[1, 0, 1],  # First sequence
                        [1, 1, 0]], dtype=torch.float32)  # Second sequence

    # Initialize the attention layer
    attention_layer = SingleHeadAttentionLayer()

    # Compute outputs
    output_workaround = attention_layer.forward_workaround(q, k, v, mask)

    # Transform mask to -inf for masked positions for masking addition
    mask = mask.masked_fill(mask == 0, float('-inf'))   # Set 0s to -inf
    mask = mask.masked_fill(mask == 1, 0)               # Set 1s to 0
    output_normal = attention_layer(q, k, v, mask)


    # Print results
    print("Mean err between the two forward implementation")
    print(torch.mean(torch.abs(output_workaround - output_normal)))
# Attention is all your need

It's been a long time i said i would implement this so ... here we go.

## Implemented paper
- [Attention is all you need](https://arxiv.org/pdf/1706.03762) Basically one of the most important paper in AI, base of all transformer architectures.

## Step by step

- [Single Head Attention Layer]("./single_head_attention.py"): Basic single head attention layer, as explained in page 4. I also tried to workaround the mask thing, using a ew product instead of sum. (see `forward_workaround()`)
# Attention is all your need

It's been a long time i said i would implement this so ... here we go.

## Implemented paper
- [Attention is all you need](https://arxiv.org/pdf/1706.03762) Basically one of the most important paper in AI, base of all transformer architectures.

## Step by step

- [Single Head Attention Layer](./src/single_head_attention.py): Basic single head attention layer, as explained in page 4. I also tried to workaround the mask thing, using a ew product instead of sum. (see `forward_workaround()`). `forward_workaround()` uses mutliplication instead of addition for masking, it allows to use directly the boolean mask without having any transformations. It wasn't used later but was helpful to understand what was coming next so i kept it.

- [Multi Heads Attention Layer](./src/multi_head_attention.py): My implementation of MHA layer.

- [Feed Forward Layer](./src/feed_forward.py): A simple implementation of feed forward layers.

- [Transformer](./src/transformer.py): My attempt to recreate the basic tranformer architecture described in the paper.

- [Training ipynb file](./training_tranformers.ipynb): a notebook were basic training is implemented for sentence translation.

## Known issue
- Transformer training: i know current training loop is bugged (it runs but will not work as expected). Currently, the teacher forcing isn't implemented well (it only removes padding or EOS token) ...

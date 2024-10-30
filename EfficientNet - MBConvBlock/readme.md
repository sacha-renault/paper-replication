# EfficientNet - MBConvBlock

This folder contains my implementation of the MBConvBlock for the paper:

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le.
- Also, MBConv was introduced here [MobileNetV2: Inverted Residuals and Linear Bottlenecks
  ](https://arxiv.org/abs/1801.04381)

## Files

- [efficientnet_mbconv.py](./efficientnet_mbconv.py): Contains my implementation of the core MBConvBlock class, with squeeze-and-excitation.
- [efficientnet_b0.py](./efficientnet_b0.py): Contains my implementation of efficientnet b0 model.

## WORK IN PROGRESS

TODOs:

- Test a full training on any cv dataset (Imagenet ?).
- make a more flexible efficientnet class that can be any b0 to b7 model.

# MobileNetV2
This is my implementation of [MobileNetV2](https://arxiv.org/abs/1801.04381) in pyTorch

MobileNetV2 is an efficient convolutional neural network architecture designed for mobile and resource-constrained environments. Developed by researchers at Google, MobileNetV2 improves upon the original MobileNet architecture, offering significant enhancements in terms of accuracy and computational efficiency. 

# The key features of MobileNetv2 architecture:
- Inverted Residuals and Linear Bottlenecks: allows for a more efficient feature extraction process
  - Each block consists of three layers: a 1x1 convolution (expansion), a depthwise 3x3 convolution, and another 1x1 convolution (projection).
  - The expansion layer increases the number of channels, the depthwise convolution processes each channel separately, and the projection layer reduces the number of channels back to the original size.
- Depthwise Separable Convolutions: reduces the number of parameters and computations, making the network more efficient for mobile and embedded applications
  - The final 1x1 convolution uses a linear activation function instead of a non-linear one, preserving the representational power of the network and improving its ability to generalize.

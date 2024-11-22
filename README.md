# MobileNetV2
This is my implementation of [MobileNetV2](https://arxiv.org/abs/1801.04381) in pyTorch

MobileNetV2 is an efficient convolutional neural network architecture designed for mobile and resource-constrained environments. Developed by researchers at Google, MobileNetV2 improves upon the original MobileNet architecture, offering significant enhancements in terms of accuracy and computational efficiency. 

# The key features of MobileNetv2 architecture:
- Inverted Residuals and Linear Bottlenecks: allow a more efficient feature extraction process
  - Each block consists of three layers: a 1x1 convolution (expansion), a depthwise 3x3 convolution, and another 1x1 convolution (projection).
  - The expansion layer increases the number of channels, the depthwise convolution processes each channel separately, and the projection layer reduces the number of channels back to the original size.
- Depthwise Separable Convolutions: reduces the number of parameters and computations, making the network more efficient for mobile and embedded applications
  - The final 1x1 convolution uses a linear activation function instead of a non-linear one, preserving the representational power of the network and improving its ability to generalize.
 
# Personal modification and optimisation:
- Since the images data CIFAR have low resolution (32x32), a Deep architecture (19 layers of Convolution) where 5 of the layers have their stride superior than 1, this would cause loss of information, this would lead to under-fitting. I change the stride the first 3 convolution layers out of those 5 into 1, in order to conserve the flow of information.

# Training result
The result from training indicates a sign of overfitting
<img width="701" alt="Screenshot 2024-11-14 at 23 07 44" src="https://github.com/user-attachments/assets/185f4ac8-fdef-4f70-9a20-d2bf66b84a05">

!TODO:
- Tunning the complexity of models
- Image augmenetation et pre-processing

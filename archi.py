import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class InvertedResidualBlock(nn.Module):
  def __init__(self, in_c, out_c, expansion_factor, stride, isFirst=False):
    super().__init__()
    # First block vs later blocks
    self.stride = stride
    self.isFirst = isFirst
    # Seq
    self.seq = nn.Sequential(
        # Conv2d
        nn.Conv2d(in_c, in_c*expansion_factor, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(in_c*expansion_factor),
        nn.ReLU6(inplace=True),
        # DW-Conv2d
        nn.Conv2d(in_c*expansion_factor, in_c*expansion_factor, kernel_size=3,\
                  stride=self.stride,\
                  padding=1,\
                  groups=in_c*expansion_factor),
        nn.BatchNorm2d(in_c*expansion_factor),
        nn.ReLU6(inplace=True),
        # Linear
        nn.Conv2d(in_c*expansion_factor, out_c, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_c)
    )

  def forward(self, x):
    if self.stride == 2 or self.isFirst:
      return self.seq(x)
    else:
      return x + self.seq(x)

class BottleNeckBlock(nn.Module):
  def __init__(self, in_c, out_c, stride, n, expansion_factor):
    super().__init__()
    self.first_block = InvertedResidualBlock(in_c, out_c, expansion_factor, stride, isFirst=True)
    self.laterblock = nn.ModuleList()
    for i in range(n-1):
      self.laterblock.append(InvertedResidualBlock(out_c, out_c, expansion_factor, stride=1))

  def forward(self, x):
    x = self.first_block(x)
    for block in self.laterblock:
      x = block(x)
    return x
  
  class BackboneMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # original stride 2
        self.bottleneck1 = BottleNeckBlock(32, 16, n=1, stride=1, expansion_factor=6)
        self.bottleneck2 = BottleNeckBlock(16, 24, n=2, stride=1, expansion_factor=6) # original stride 2
        self.bottleneck3 = BottleNeckBlock(24, 32, n=3, stride=1, expansion_factor=6) # original stride 2
        self.bottleneck4 = BottleNeckBlock(32, 64, n=4, stride=2, expansion_factor=6)
        self.bottleneck5 = BottleNeckBlock(64, 96, n=3, stride=1, expansion_factor=6)
        self.bottleneck6 = BottleNeckBlock(96, 160, n=3, stride=2, expansion_factor=6)
        self.bottleneck7 = BottleNeckBlock(160, 320, n=1, stride=1, expansion_factor=6)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        # init convo
        x = self.init_conv(x)
        # BottleNeck
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        # final convo
        x = self.conv2(x)
        x = self.avg_pool(x)
        # flatten the extracted features
        x = self.flatten(x) # (Batch, 1280)
        return x
    
class MyNet(nn.Module):
  def __init__(self, nb_classes):
    super().__init__()
    self.backbone = BackboneMobileNetV2()
    self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, nb_classes),
        )

    # weight initialization
    self._initialize_weights()

  def forward(self, x):
    x = self.backbone(x)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            # m.weight.data.normal_(0, 0.01)
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
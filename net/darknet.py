import math
import torch
from config import *
from torch import nn
from collections import OrderedDict

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, planes):  # in_channels:输入残差块的通道数  planes:残差块间的通道数变化例如[32, 64]通过残差块后通道数由32变为64
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes[0], kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out + residual


class Darknet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.in_channels = 32  # 图像的输入通道数
        # Convoluional
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        # --------------------------------
        # 残差和下采样
        self.residual1 = self._make_layer([32, 64], layers[0])
        self.residual2 = self._make_layer([64, 128], layers[1])
        self.residual3 = self._make_layer([128, 256], layers[2])
        self.residual4 = self._make_layer([256, 512], layers[3])
        self.residual5 = self._make_layer([512, 1024], layers[4])
        # ---------------------------------------------------------------

        # 对模型权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):  # planes:残差块间的通道数变化例如[32, 64]通过残差块后通道数由32变为64  blocks:要连接的残差块数量
        layers = []
        layers.append(('ds_conv', nn.Conv2d(self.in_channels, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(('ds_bn', nn.BatchNorm2d(planes[1])))
        layers.append(('ds_relu', nn.LeakyReLU(0.1)))

        self.in_channels = planes[1]

        # 加入残差块
        for i in range(blocks):
            layers.append((f'res_{i}', ResidualBlock(self.in_channels, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        result = self.conv1(x)
        result = self.bn1(result)
        result = self.relu1(result)
        result = self.residual1(result)
        result = self.residual2(result)
        out_52 = self.residual3(result)
        out_26 = self.residual4(out_52)
        out_13 = self.residual5(out_26)
        return out_13, out_26, out_52

def darknet53(pretrained=False):
    model = Darknet(residual_num)
    # 如果有权重就加载权重
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(pretrained)
        else:
            raise Exception("darknet request a pretrained path.got[{}]".format(pretrained))
    return model

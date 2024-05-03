import torch
from torch import nn
from darknet import *
from config import *

# convolutional块
def Convolutional(in_channels, out_channels, kernel_size):  # in_channels:输入通道数 channels_list:中间通道数 kernel_size:卷积核大小
    layers = []
    padding = kernel_size // 2
    layers.append(('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)))
    layers.append(('bn', nn.BatchNorm2d(out_channels)))
    layers.append(('relu', nn.LeakyReLU(0.1)))
    return nn.Sequential(OrderedDict(layers))

def make_list_layer(in_channels, channels_list, out_channels):  # in_channels:输入通道数 channels_list:中间通道数 out_channels:输出通道数
    return nn.Sequential(
        Convolutional(in_channels, channels_list[0], kernel_size=1),
        Convolutional(channels_list[0], channels_list[1], kernel_size=3),
        Convolutional(channels_list[1], channels_list[0], kernel_size=1),
        Convolutional(channels_list[0], channels_list[1], kernel_size=3),
        Convolutional(channels_list[1], channels_list[0], kernel_size=1),
        Convolutional(channels_list[0], channels_list[1], kernel_size=3),
        nn.Conv2d(channels_list[1], out_channels, kernel_size=1)
    )

class YoLov3net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = darknet53()

        self.out_channels = 3*(1+4+CLASS_SUM)
        self.set_13 = make_list_layer(set_input[-1], [512, 1024], self.out_channels)
        self.set_26 = make_list_layer(set_input[-2]+set_input[-3], [set_input[-2]+set_input[-3], 512], self.out_channels)
        self.set_52 = make_list_layer(set_input[-3]+set_input[-4], [set_input[-3]+set_input[-4], 256], self.out_channels)

        # 下采样
        self.downsample_13_2_26 = Convolutional(set_input[-2], set_input[-3], 1)
        self.downsample_26_2_52 = Convolutional(set_input[-2]+set_input[-3], set_input[-4], 1)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        out_13, out_26, out_52 = self.net(x)  # 拿到darknet53的三个输出
        def _branch(layer_in, layer_out):
            for i, e in enumerate(layer_in):
                layer_out = e(layer_out)
                if i == 4:
                    out_branch = layer_out
            return layer_out, out_branch

        output_13, out_branch_13 = _branch(self.set_13, out_13)

        out_branch_13 = self.downsample_13_2_26(out_branch_13)
        out_branch_13 = self.upsample(out_branch_13)
        out_26 = torch.concat([out_26, out_branch_13], dim=1)

        output_26, out_branch_26 = _branch(self.set_26, out_26)
        out_branch_26 = self.downsample_26_2_52(out_branch_26)
        out_branch_26 = self.upsample(out_branch_26)
        out_52 = torch.concat([out_52, out_branch_26], dim=1)

        output_52, out_branch_52 = _branch(self.set_52, out_52)
        return output_13, output_26, output_52

if __name__ == '__main__':
    x = torch.rand(1, 3, 416, 416)
    net = YoLov3net()
    a, b, c = net(x)
    print(a.shape)

import torch
from torch import nn
from config import *


class Decodebox(nn.Module):
    def __init__(self, anchors):

        self.anchors = anchors  # 拿到先验框
        self.len_anchors = len(anchors)  # 拿到先验框个数
        self.class_num = CLASS_SUM  # 分类总数
        self.an_length = 5+self.class_num
        self.img_w = img_w  # 输入网络图像的宽度
        self.img_h = img_h  # 输入网络图像的高度
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input):
        # input是网络输出的三个特征图shape分别为: [batch, channels, 13, 13]  [batch, channels, w, h]
        #                                   [batch, channels, 26, 26]
        #                                   [batch, channels, 52, 52]
        input_batch = input.shape[0]  # 网络输出的batch
        input_w = input.shape[2]  # 网络输出的宽
        input_h = input.shape[3]  # 网络输出的高

        stride_W = self.img_w / input_w  # 从（416*416）到(13*13)图像高的缩小倍数
        stride_h = self.img_h / input_h  # 从（416*416）到(13*13)图像高的缩小倍数

        scaled_anchors = torch.tensor([(anchors_w/stride_W, anchors_h/stride_h) for (anchors_w, anchors_h) in self.anchors])  # 先验框从(416*416)映射到(13*13)中的大小

        prediction = input.reshape(input.shape[0], 3, self.an_length, input_w, input_h).permute(0, 1, 3, 4, 2)  # 先将input的shape变为[batch, 3, 13, 13, 25]
        pre_x = torch.sigmoid(prediction[..., 0])  # 拿到预测中心坐标x
        pre_y = torch.sigmoid(prediction[..., 1])  # 拿到预测中心坐标y
        pre_tw = prediction[..., 2]  # 拿到预测先验框缩放因子
        pre_th = prediction[..., 3]  # 拿到预测先验框缩放因子
        conf = torch.sigmoid(prediction[..., 4])   # 置信度
        pre_class = torch.sigmoid(prediction[..., 5:])  # 预测类别

        grad_x = (torch.linspace(0, input_w-1, input_w).repeat(input_h, 1).
                  repeat(input_batch*self.len_anchors, 1, 1).reshape(pre_x.shape).type(torch.FloatTensor).to(self.device))
        grad_y = (torch.linspace(0, input_h-1, input_h).repeat(input_w, 1).t().
                  repeat(input_batch*self.len_anchors, 1, 1).reshape(pre_y.shape).to(self.device))

        anchors_w = (scaled_anchors.float()).index_select(1, torch.tensor([0]))
        anchors_h = (scaled_anchors.float()).index_select(1, torch.tensor([1]))
        anchors_w = anchors_w.repeat(input_batch, 1).repeat(1, 1, input_h*input_w).reshape(pre_tw.shape).to(self.device)
        anchors_h = anchors_h.repeat(input_batch, 1).repeat(1, 1, input_w*input_h).reshape(pre_th.shape).to(self.device)

        prediction_box = torch.zeros(prediction[..., :4].shape, dtype=torch.float32)
        prediction_box[..., 0] = pre_x.data+grad_x
        prediction_box[..., 1] = pre_y.data+grad_y
        prediction_box[..., 2] = torch.exp(pre_tw)*anchors_w
        prediction_box[..., 3] = torch.exp(pre_th)*anchors_h

        scale = torch.tensor([stride_h, stride_W]*2).float().to(self.device)
        output = torch.cat((prediction_box.reshape(input_batch, -1, 4)*scale, conf.reshape(input_batch, -1, 1), pre_class.reshape(input_batch, -1, self.class_num)), dim=-1)
        return output


if __name__ == '__main__':
    input = torch.rand(2, 75, 13, 13)
    d = Decodebox([[116, 90], [158, 198], [373, 326]])
    r = d.forward(input)
